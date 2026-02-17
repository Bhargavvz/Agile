"""
evaluate.py
===========
Comprehensive evaluation of the trained Agile Cost Estimator.

Produces:
    - Metrics: R², MAPE, MAE, RMSE, MedAE, calibration error, PICP
    - 8 publication-quality plots saved to results/plots/
    - Per-complexity and per-domain breakdown tables
    - MC Dropout uncertainty estimates
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.amp import autocast

import config
from dataset import AgileCostDataset, get_dataloaders
from model import AgileCostEstimator

# plot style
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})
sns.set_style("whitegrid")


# ──────────────────────────── load checkpoint ────────────────────────────

def load_model(device: torch.device) -> Tuple[AgileCostEstimator, dict]:
    """Load the best checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AgileCostEstimator(
        num_numeric_features=checkpoint["config"]["num_numeric_features"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # restore scalers
    AgileCostDataset.get_feature_scaler().load_state_dict(checkpoint["feature_scaler"])
    AgileCostDataset.get_target_scaler().load_state_dict(checkpoint["target_scaler"])

    print(f"  ✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"     Val metrics at save: {checkpoint['metrics']}")
    return model, checkpoint


# ──────────────────────────── MC Dropout inference ───────────────────────

def predict_with_uncertainty(
    model: AgileCostEstimator,
    dataloader,
    device: torch.device,
    n_passes: int = config.MC_DROPOUT_PASSES,
) -> Dict[str, np.ndarray]:
    """
    Run MC Dropout inference for uncertainty estimation.

    Returns dict with:
        mu_mean      : mean predicted cost (normalised)
        mu_std       : epistemic uncertainty (std across MC passes)
        sigma_mean   : mean aleatoric uncertainty
        targets      : ground-truth targets (normalised)
        raw_costs    : original dollar costs
        domains      : domain labels
        complexities : complexity labels
    """
    model.train()  # keep dropout active for MC Dropout

    all_mu_passes = []    # [n_passes, N]
    all_sigma_passes = [] # [n_passes, N]
    all_targets = []
    all_raw_costs = []
    all_domains = []
    all_complexities = []

    # collect domain/complexity metadata from dataset
    test_ds = dataloader.dataset
    for row in test_ds.rows:
        all_domains.append(row["domain"])
        all_complexities.append(row["complexity_level"])

    with torch.no_grad():
        for pass_idx in range(n_passes):
            pass_mu = []
            pass_sigma = []
            pass_targets = []
            pass_raw = []

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                features = batch["numeric_features"].to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16, enabled=config.USE_AMP):
                    mu, sigma = model(input_ids, attention_mask, features)

                pass_mu.append(mu.float().cpu().numpy())
                pass_sigma.append(sigma.float().cpu().numpy())

                if pass_idx == 0:
                    pass_targets.append(batch["target"].numpy())
                    pass_raw.append(batch["raw_cost"].numpy())

            all_mu_passes.append(np.concatenate(pass_mu))
            all_sigma_passes.append(np.concatenate(pass_sigma))

            if pass_idx == 0:
                all_targets = np.concatenate(pass_targets)
                all_raw_costs = np.concatenate(pass_raw)

    mu_passes = np.stack(all_mu_passes, axis=0)     # [n_passes, N]
    sigma_passes = np.stack(all_sigma_passes, axis=0)

    return {
        "mu_mean": mu_passes.mean(axis=0),
        "mu_std": mu_passes.std(axis=0),        # epistemic uncertainty
        "sigma_mean": sigma_passes.mean(axis=0), # aleatoric uncertainty
        "targets": all_targets,
        "raw_costs": all_raw_costs,
        "domains": all_domains[:len(all_targets)],
        "complexities": all_complexities[:len(all_targets)],
    }


# ──────────────────────────── metric computation ─────────────────────────

def compute_all_metrics(
    preds_usd: np.ndarray,
    targets_usd: np.ndarray,
    total_uncertainty_usd: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    # R²
    ss_res = ((targets_usd - preds_usd) ** 2).sum()
    ss_tot = ((targets_usd - targets_usd.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    # MAPE
    mape = np.mean(np.abs((targets_usd - preds_usd) / (targets_usd + 1e-8))) * 100

    # MAE
    mae = np.mean(np.abs(targets_usd - preds_usd))

    # RMSE
    rmse = np.sqrt(np.mean((targets_usd - preds_usd) ** 2))

    # Median Absolute Error
    medae = np.median(np.abs(targets_usd - preds_usd))

    # Prediction Interval Coverage Probability (PICP) at 90%
    z = 1.645  # 90% CI
    lower = preds_usd - z * total_uncertainty_usd
    upper = preds_usd + z * total_uncertainty_usd
    picp = np.mean((targets_usd >= lower) & (targets_usd <= upper))

    # Mean Prediction Interval Width (MPIW)
    mpiw = np.mean(upper - lower)

    return {
        "r2": round(float(r2), 4),
        "mape_pct": round(float(mape), 2),
        "mae_usd": round(float(mae), 2),
        "rmse_usd": round(float(rmse), 2),
        "medae_usd": round(float(medae), 2),
        "picp_90": round(float(picp), 4),
        "mpiw_usd": round(float(mpiw), 2),
    }


# ──────────────────────────── plotting ───────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(config.PLOTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    📈 Saved: {path}")


def plot_loss_curves(history: dict):
    """Plot 1: Training & validation loss curves."""
    fig, ax = plt.subplots()
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax.axvline(x=config.NUM_EPOCHS_PHASE1, color="gray", linestyle="--",
               alpha=0.7, label="Phase 1→2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    _save(fig, "01_loss_curves")


def plot_predicted_vs_actual(preds_usd, targets_usd, r2):
    """Plot 2: Predicted vs Actual scatter."""
    fig, ax = plt.subplots()
    ax.scatter(targets_usd, preds_usd, alpha=0.3, s=10, color="steelblue")
    lims = [
        min(targets_usd.min(), preds_usd.min()),
        max(targets_usd.max(), preds_usd.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Cost (USD)")
    ax.set_ylabel("Predicted Cost (USD)")
    ax.set_title(f"Predicted vs Actual Cost — R² = {r2:.4f}")
    ax.legend()
    _save(fig, "02_predicted_vs_actual")


def plot_residual_distribution(preds_usd, targets_usd):
    """Plot 3: Residual histogram."""
    residuals = preds_usd - targets_usd
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (Predicted - Actual) USD")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    _save(fig, "03_residual_distribution")


def plot_error_by_complexity(preds_usd, targets_usd, complexities):
    """Plot 4: Error by complexity box plot."""
    errors = np.abs(preds_usd - targets_usd)
    data = {"Low": [], "Medium": [], "High": []}
    for e, c in zip(errors, complexities):
        data[c].append(e)

    fig, ax = plt.subplots()
    labels = ["Low", "Medium", "High"]
    box_data = [data[l] for l in labels]
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Absolute Error (USD)")
    ax.set_title("Prediction Error by Complexity Level")
    _save(fig, "04_error_by_complexity")


def plot_error_by_domain(preds_usd, targets_usd, domains):
    """Plot 5: Error by domain box plot."""
    errors = np.abs(preds_usd - targets_usd)
    domain_set = sorted(set(domains))
    data = {d: [] for d in domain_set}
    for e, d in zip(errors, domains):
        data[d].append(e)

    fig, ax = plt.subplots(figsize=(12, 7))
    box_data = [data[d] for d in domain_set]
    ax.boxplot(box_data, labels=domain_set, patch_artist=True)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Absolute Error (USD)")
    ax.set_title("Prediction Error by Domain")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "05_error_by_domain")


def plot_cumulative_error(preds_usd, targets_usd):
    """Plot 6: Cumulative error distribution."""
    pct_errors = np.abs((preds_usd - targets_usd) / (targets_usd + 1e-8)) * 100
    sorted_errors = np.sort(pct_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    fig, ax = plt.subplots()
    ax.plot(sorted_errors, cumulative * 100, linewidth=2, color="steelblue")
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, label="90th percentile")
    ax.set_xlabel("Absolute Percentage Error (%)")
    ax.set_ylabel("Cumulative % of Samples")
    ax.set_title("Cumulative Error Distribution")
    ax.set_xlim(0, min(50, sorted_errors.max()))
    ax.legend()
    _save(fig, "06_cumulative_error")


def plot_uncertainty_calibration(preds_usd, targets_usd, total_unc_usd):
    """Plot 7: Uncertainty calibration plot."""
    expected_coverages = np.linspace(0.05, 0.95, 19)
    observed_coverages = []

    for ec in expected_coverages:
        z = abs(np.percentile(np.random.standard_normal(100000), ((1 + ec) / 2) * 100))
        lower = preds_usd - z * total_unc_usd
        upper = preds_usd + z * total_unc_usd
        coverage = np.mean((targets_usd >= lower) & (targets_usd <= upper))
        observed_coverages.append(coverage)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax.plot(expected_coverages, observed_coverages, "o-", color="steelblue",
            linewidth=2, markersize=6, label="Model")
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Uncertainty Calibration Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "07_uncertainty_calibration")


def plot_prediction_intervals(preds_usd, targets_usd, total_unc_usd):
    """Plot 8: Prediction intervals for a subset of samples."""
    n_show = min(100, len(preds_usd))
    # sort by actual cost for clean visualisation
    sort_idx = np.argsort(targets_usd)[:n_show]
    p = preds_usd[sort_idx]
    t = targets_usd[sort_idx]
    u = total_unc_usd[sort_idx]

    z = 1.645  # 90% CI
    x = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(x, p - z * u, p + z * u, alpha=0.25, color="steelblue",
                    label="90% CI")
    ax.plot(x, p, "o", color="steelblue", markersize=3, label="Predicted")
    ax.plot(x, t, "x", color="red", markersize=4, label="Actual")
    ax.set_xlabel("Sample (sorted by actual cost)")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Prediction Intervals with 90% Confidence")
    ax.legend()
    _save(fig, "08_prediction_intervals")


# ──────────────────────────── main evaluation ────────────────────────────

def evaluate():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    _, _, test_loader = get_dataloaders()
    target_scaler = AgileCostDataset.get_target_scaler()

    # load model
    model, checkpoint = load_model(device)

    # load training history for loss curves
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_loss_curves(history)
    else:
        print("  ⚠ training_history.json not found — skipping loss curves")

    # MC Dropout inference
    print(f"\n  Running MC Dropout inference ({config.MC_DROPOUT_PASSES} passes)...")
    results = predict_with_uncertainty(model, test_loader, device)

    # convert to USD scale
    preds_usd = target_scaler.inverse_transform(results["mu_mean"])
    targets_usd = results["raw_costs"]
    preds_usd = np.maximum(preds_usd, 0)

    # uncertainty in USD scale
    aleatoric_usd = results["sigma_mean"] * target_scaler.std
    epistemic_usd = results["mu_std"] * target_scaler.std
    total_unc_usd = np.sqrt(aleatoric_usd ** 2 + epistemic_usd ** 2)

    # compute metrics
    metrics = compute_all_metrics(preds_usd, targets_usd, total_unc_usd)

    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*60}\n")

    # per-complexity breakdown
    print("  Per-Complexity Breakdown:")
    for level in ["Low", "Medium", "High"]:
        mask = [c == level for c in results["complexities"]]
        if sum(mask) > 0:
            m = np.array(mask)
            level_preds = preds_usd[m]
            level_targets = targets_usd[m]
            r2_l = 1 - ((level_targets - level_preds) ** 2).sum() / (
                (level_targets - level_targets.mean()) ** 2
            ).sum()
            mape_l = np.mean(np.abs((level_targets - level_preds) / (level_targets + 1e-8))) * 100
            print(f"    {level:8s}: R²={r2_l:.4f}, MAPE={mape_l:.2f}%, n={sum(mask)}")

    # save metrics
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  📊 Metrics saved → {metrics_path}")

    # generate all plots
    print(f"\n  Generating plots...")
    plot_predicted_vs_actual(preds_usd, targets_usd, metrics["r2"])
    plot_residual_distribution(preds_usd, targets_usd)
    plot_error_by_complexity(preds_usd, targets_usd, results["complexities"])
    plot_error_by_domain(preds_usd, targets_usd, results["domains"])
    plot_cumulative_error(preds_usd, targets_usd)
    plot_uncertainty_calibration(preds_usd, targets_usd, total_unc_usd)
    plot_prediction_intervals(preds_usd, targets_usd, total_unc_usd)

    print(f"\n  ✅ Evaluation complete! All results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    evaluate()
