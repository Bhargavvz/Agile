"""
analysis.py
============
Dataset analysis and feature importance visualization.

Generates:
    - Feature distribution plots
    - Correlation heatmap
    - Gradient-based feature importance from the trained model
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.cuda.amp import autocast

import config
from dataset import AgileCostDataset, get_dataloaders
from model import AgileCostEstimator

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "figure.dpi": 150,
})
sns.set_style("whitegrid")

ANALYSIS_DIR = os.path.join(config.RESULTS_DIR, "analysis")


def _save(fig, name: str):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(ANALYSIS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    📊 Saved: {path}")


# ──────────────────────────── dataset distributions ──────────────────────

def plot_distributions():
    """Plot distributions of key features and targets."""
    df = pd.read_csv(config.LABELS_CSV)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Continuous features
    continuous = [
        ("actual_cost_usd", "Cost Distribution (USD)"),
        ("actual_effort_hours", "Effort Distribution (Hours)"),
        ("team_size", "Team Size Distribution"),
        ("duration_months", "Duration Distribution (Months)"),
        ("total_user_stories", "User Stories Distribution"),
        ("avg_story_points", "Avg Story Points Distribution"),
        ("velocity_per_sprint", "Velocity per Sprint"),
    ]

    for idx, (col, title) in enumerate(continuous):
        row, col_idx = divmod(idx, 3)
        ax = axes[row, col_idx]
        ax.hist(df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Categorical counts
    ax = axes[2, 1]
    df["complexity_level"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_title("Complexity Level Distribution")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[2, 2]
    df["domain"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Domain Distribution")
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Dataset Feature Distributions", fontsize=16, y=1.01)
    fig.tight_layout()
    _save(fig, "feature_distributions")


def plot_correlation_heatmap():
    """Plot correlation heatmap of numeric features and target."""
    df = pd.read_csv(config.LABELS_CSV)

    numeric_cols = [
        "team_size", "duration_months", "num_sprints",
        "total_user_stories", "avg_story_points", "velocity_per_sprint",
        "actual_effort_hours", "actual_cost_usd",
    ]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "correlation_heatmap")


def plot_cost_by_category():
    """Plot cost distributions by complexity and domain."""
    df = pd.read_csv(config.LABELS_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By complexity
    for level, color in zip(["Low", "Medium", "High"], ["#2ecc71", "#f39c12", "#e74c3c"]):
        subset = df[df["complexity_level"] == level]["actual_cost_usd"]
        axes[0].hist(subset, bins=40, alpha=0.5, label=level, color=color)
    axes[0].set_title("Cost Distribution by Complexity")
    axes[0].set_xlabel("Cost (USD)")
    axes[0].legend()

    # By domain
    domain_means = df.groupby("domain")["actual_cost_usd"].mean().sort_values()
    domain_means.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Mean Cost by Domain")
    axes[1].set_xlabel("Mean Cost (USD)")

    fig.tight_layout()
    _save(fig, "cost_by_category")


# ──────────────────────────── feature importance ─────────────────────────

def compute_feature_importance():
    """
    Gradient-based feature importance.
    Measures how much each numeric input feature influences the predicted cost.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print("  ⚠ No trained model found — skipping feature importance")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AgileCostEstimator(
        num_numeric_features=checkpoint["config"]["num_numeric_features"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    # accumulate gradients w.r.t. numeric features
    grad_importance = np.zeros(config.NUM_NUMERIC_FEATURES)
    n_samples = 0

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["numeric_features"].to(device).requires_grad_(True)
        targets = batch["target"].to(device)

        mu, sigma = model(input_ids, attention_mask, features)
        loss = mu.sum()  # use sum to get gradient for each sample
        loss.backward()

        grad_importance += features.grad.abs().sum(dim=0).cpu().numpy()
        n_samples += features.shape[0]

    grad_importance /= n_samples

    # build feature names
    feature_names = list(config.NUMERIC_FEATURES)
    for col, categories in config.CATEGORICAL_FEATURES.items():
        for cat in categories:
            feature_names.append(f"{col}={cat}")

    # plot
    sort_idx = np.argsort(grad_importance)[::-1]
    sorted_names = [feature_names[i] for i in sort_idx]
    sorted_vals = grad_importance[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    ax.barh(range(len(sorted_names)), sorted_vals[::-1], color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.set_xlabel("Mean |Gradient|")
    ax.set_title("Feature Importance (Gradient-Based Attribution)")
    _save(fig, "feature_importance")

    # save as JSON
    importance_dict = {name: float(val) for name, val in zip(sorted_names, sorted_vals)}
    imp_path = os.path.join(ANALYSIS_DIR, "feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"    📊 Feature importance saved: {imp_path}")


# ──────────────────────────── main ───────────────────────────────────────

def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset Analysis & Feature Importance")
    print(f"{'='*60}\n")

    print("  Generating distribution plots...")
    plot_distributions()

    print("  Generating correlation heatmap...")
    plot_correlation_heatmap()

    print("  Generating cost-by-category plots...")
    plot_cost_by_category()

    print("  Computing feature importance...")
    compute_feature_importance()

    print(f"\n  ✅ Analysis complete! All plots in: {ANALYSIS_DIR}\n")


if __name__ == "__main__":
    main()
