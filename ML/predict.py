"""
predict.py
==========
Inference script for predicting software project cost from a report.

Supports:
    - Single report prediction with uncertainty
    - Batch prediction on a directory of reports

Usage:
    python predict.py path/to/report.txt
    python predict.py path/to/reports_dir/ --batch
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoTokenizer

from ML import config
from ML.dataset import AgileCostDataset, FeatureScaler, TargetScaler
from ML.model import AgileCostEstimator


# ──────────────────────────── load model ─────────────────────────────────

def load_model_for_inference(device: torch.device):
    """Load trained model and scalers from checkpoint."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print("  Run train.py first to train the model.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = AgileCostEstimator(
        num_numeric_features=checkpoint["config"]["num_numeric_features"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["config"]["model_name"])

    target_scaler = TargetScaler()
    target_scaler.load_state_dict(checkpoint["target_scaler"])

    feature_scaler = FeatureScaler()
    feature_scaler.load_state_dict(checkpoint["feature_scaler"])

    return model, tokenizer, target_scaler, feature_scaler


# ──────────────────────────── extract features from text ─────────────────

def _extract_features_from_text(text: str) -> np.ndarray:
    """
    Heuristically extract structured features from report text.
    Falls back to median values if parsing fails.
    """
    import re

    defaults = {
        "team_size": 6,
        "duration_months": 6,
        "num_sprints": 12,
        "total_user_stories": 18,
        "avg_story_points": 5.0,
        "velocity_per_sprint": 20.0,
    }

    # Try to extract from report header / body
    patterns = {
        "team_size": r"Team Size\s*:\s*(\d+)",
        "duration_months": r"Duration\s*:\s*(\d+)\s*months",
        "num_sprints": r"(\d+)\s*sprints",
        "total_user_stories": r"backlog contains\s*(\d+)\s*user stories",
        "avg_story_points": r"average.*?(\d+\.?\d*)",
        "velocity_per_sprint": r"velocity.*?(\d+\.?\d*)",
    }

    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        values[key] = float(match.group(1)) if match else defaults[key]

    # Detect categorical features from text
    text_lower = text.lower()

    # Complexity
    if "high" in text_lower and "complexity" in text_lower:
        complexity = "High"
    elif "medium" in text_lower and "complexity" in text_lower:
        complexity = "Medium"
    else:
        complexity = "Low"

    # Tech difficulty
    if any(kw in text_lower for kw in ["ml", "ai", "real-time", "encryption"]):
        tech_diff = "Advanced"
    elif any(kw in text_lower for kw in ["api gateway", "oauth", "ci/cd"]):
        tech_diff = "Intermediate"
    else:
        tech_diff = "Basic"

    # Volatility
    if "high" in text_lower and "volatility" in text_lower:
        volatility = "High"
    elif "medium" in text_lower and "volatility" in text_lower:
        volatility = "Medium"
    else:
        volatility = "Low"

    # Risk
    if "high" in text_lower and "risk" in text_lower:
        risk = "High"
    elif "medium" in text_lower and "risk" in text_lower:
        risk = "Medium"
    else:
        risk = "Low"

    # Build feature vector (same order as dataset.py)
    feat = []
    for col in config.NUMERIC_FEATURES:
        feat.append(values.get(col, defaults.get(col, 0)))

    # One-hot categoricals
    cat_vals = {
        "complexity_level": complexity,
        "tech_stack_difficulty": tech_diff,
        "requirement_volatility_score": volatility,
        "risk_level": risk,
    }
    for col, categories in config.CATEGORICAL_FEATURES.items():
        val = cat_vals[col]
        for cat in categories:
            feat.append(1.0 if val == cat else 0.0)

    return np.array([feat], dtype=np.float32)


# ──────────────────────────── single prediction ─────────────────────────

def predict_single(
    report_path: str,
    model: AgileCostEstimator,
    tokenizer,
    target_scaler: TargetScaler,
    feature_scaler: FeatureScaler,
    device: torch.device,
) -> Dict[str, float]:
    """
    Predict cost for a single report with uncertainty.
    """
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    # tokenise
    encoding = tokenizer(
        text,
        max_length=config.MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # features
    raw_features = _extract_features_from_text(text)
    scaled_features = feature_scaler.transform(raw_features)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # MC Dropout inference
    model.train()  # keep dropout active

    mu_samples = []
    sigma_samples = []

    with torch.no_grad():
        for _ in range(config.MC_DROPOUT_PASSES):
            with autocast("cuda", dtype=torch.bfloat16, enabled=config.USE_AMP and device.type == "cuda"):
                mu, sigma = model(input_ids, attention_mask, features_tensor)
            mu_samples.append(mu.float().cpu().item())
            sigma_samples.append(sigma.float().cpu().item())

    mu_mean = np.mean(mu_samples)
    mu_std = np.std(mu_samples)          # epistemic uncertainty
    sigma_mean = np.mean(sigma_samples)  # aleatoric uncertainty

    # convert to USD
    predicted_cost = target_scaler.inverse_transform(np.array([mu_mean]))[0]
    predicted_cost = max(0, predicted_cost)

    # uncertainty in USD (approximate via scaling)
    aleatoric_usd = abs(sigma_mean * target_scaler.std) * predicted_cost * 0.01
    epistemic_usd = abs(mu_std * target_scaler.std) * predicted_cost * 0.01
    total_usd = np.sqrt(aleatoric_usd ** 2 + epistemic_usd ** 2)

    # 90% CI
    z90 = 1.645
    lower = max(0, predicted_cost - z90 * total_usd)
    upper = predicted_cost + z90 * total_usd

    return {
        "predicted_cost_usd": round(predicted_cost, 2),
        "aleatoric_uncertainty_usd": round(aleatoric_usd, 2),
        "epistemic_uncertainty_usd": round(epistemic_usd, 2),
        "total_uncertainty_usd": round(total_usd, 2),
        "ci_90_lower_usd": round(lower, 2),
        "ci_90_upper_usd": round(upper, 2),
    }


# ──────────────────────────── batch prediction ──────────────────────────

def predict_batch(
    reports_dir: str,
    model: AgileCostEstimator,
    tokenizer,
    target_scaler: TargetScaler,
    feature_scaler: FeatureScaler,
    device: torch.device,
):
    """Predict cost for all .txt files in a directory."""
    files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".txt")])

    print(f"\n  Predicting {len(files)} reports from: {reports_dir}\n")
    print(f"  {'File':<25} {'Predicted Cost':>18} {'±Uncertainty':>18} {'90% CI':>30}")
    print(f"  {'─'*25} {'─'*18} {'─'*18} {'─'*30}")

    for fname in files:
        path = os.path.join(reports_dir, fname)
        result = predict_single(path, model, tokenizer, target_scaler, feature_scaler, device)
        ci = f"${result['ci_90_lower_usd']:,.0f} – ${result['ci_90_upper_usd']:,.0f}"
        print(
            f"  {fname:<25} "
            f"${result['predicted_cost_usd']:>15,.2f} "
            f"±${result['total_uncertainty_usd']:>14,.2f} "
            f"{ci:>30}"
        )


# ──────────────────────────── CLI ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict Agile software project cost from report text."
    )
    parser.add_argument(
        "path",
        help="Path to a .txt report file or directory of reports",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch prediction (path must be a directory)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, target_scaler, feature_scaler = load_model_for_inference(device)

    if args.batch:
        assert os.path.isdir(args.path), f"Not a directory: {args.path}"
        predict_batch(args.path, model, tokenizer, target_scaler, feature_scaler, device)
    else:
        assert os.path.isfile(args.path), f"Not a file: {args.path}"
        result = predict_single(args.path, model, tokenizer, target_scaler, feature_scaler, device)

        print(f"\n{'='*60}")
        print(f"  COST PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"  Report         : {os.path.basename(args.path)}")
        print(f"  Predicted Cost : ${result['predicted_cost_usd']:,.2f}")
        print(f"  ────────────────────────────────────")
        print(f"  Aleatoric unc. : ±${result['aleatoric_uncertainty_usd']:,.2f}")
        print(f"  Epistemic unc. : ±${result['epistemic_uncertainty_usd']:,.2f}")
        print(f"  Total unc.     : ±${result['total_uncertainty_usd']:,.2f}")
        print(f"  90% CI         : ${result['ci_90_lower_usd']:,.2f} – ${result['ci_90_upper_usd']:,.2f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
