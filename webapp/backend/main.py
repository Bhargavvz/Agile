"""
Agile Cost Estimator — FastAPI Backend
=======================================
Serves ML results, dataset statistics, plot images, and cost predictions
using the trained BERT-Large model.
"""

import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─── paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ML"
DATASET_DIR = ML_DIR / "agile_cost_dataset"
LABELS_CSV = DATASET_DIR / "labels.csv"
RESULTS_DIR = ML_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
CHECKPOINT_PATH = ML_DIR / "checkpoints" / "best_model.pt"

# ─── add ML directory to sys.path for imports ────────────────────────────
# ML code uses `import ML.config` then references `config.XXX`
# So we need BASE_DIR (for `ML.config`) AND ML_DIR (for `config` shorthand)
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ML_DIR))

# ─── try loading the trained model ───────────────────────────────────────
MODEL_LOADED = False
model = None
tokenizer = None
target_scaler = None
feature_scaler = None
device = None

try:
    import torch
    from torch.amp import autocast
    from transformers import AutoTokenizer

    # ML code does `import ML.config` then uses bare `config.XXX`.
    # `import ML.config` only puts `ML` in namespace, not `config`.
    # Fix: pre-import config.py as standalone module into sys.modules
    # so that when model.py class body evaluates `config.XXX`, Python
    # finds `config` as a real module.
    import importlib.util
    _config_path = str(ML_DIR / "config.py")
    _spec = importlib.util.spec_from_file_location("config", _config_path)
    _config_mod = importlib.util.module_from_spec(_spec)
    sys.modules["config"] = _config_mod
    _spec.loader.exec_module(_config_mod)
    config = _config_mod  # also available locally

    from ML.model import AgileCostEstimator
    from ML.dataset import FeatureScaler, TargetScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CHECKPOINT_PATH.exists():
        print(f"  ⏳ Loading trained model from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)

        model = AgileCostEstimator(
            num_numeric_features=checkpoint["config"]["num_numeric_features"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = AutoTokenizer.from_pretrained(checkpoint["config"]["model_name"])

        target_scaler = TargetScaler()
        target_scaler.load_state_dict(checkpoint["target_scaler"])

        feature_scaler = FeatureScaler()
        feature_scaler.load_state_dict(checkpoint["feature_scaler"])

        MODEL_LOADED = True
        print(f"  ✅ Model loaded successfully on {device}")
    else:
        print(f"  ⚠️  Checkpoint not found at {CHECKPOINT_PATH}")
        print("  Falling back to heuristic prediction")

except ImportError as e:
    print(f"  ⚠️  Could not import ML dependencies: {e}")
    print("  Falling back to heuristic prediction")
except Exception as e:
    print(f"  ⚠️  Error loading model: {e}")
    traceback.print_exc()
    print("  Falling back to heuristic prediction")


# ─── feature extraction from report text ─────────────────────────────────
def extract_features_from_text(text: str) -> dict:
    """
    Extract structured features from a project report using regex.
    Returns both parsed values and the raw feature array.
    """
    defaults = {
        "team_size": 6,
        "duration_months": 6,
        "num_sprints": 12,
        "total_user_stories": 18,
        "avg_story_points": 5.0,
        "velocity_per_sprint": 20.0,
    }

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

    text_lower = text.lower()

    # Domain detection
    domain_map = {
        "e-commerce": "E-commerce", "healthcare": "Healthcare",
        "fintech": "FinTech", "edtech": "EdTech", "saas": "SaaS",
        "ai platform": "AI Platform", "iot": "IoT", "cloud system": "Cloud System",
    }
    domain = "SaaS"
    for key, val in domain_map.items():
        if key in text_lower:
            domain = val
            break

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

    return {
        "numeric": values,
        "domain": domain,
        "complexity_level": complexity,
        "tech_stack_difficulty": tech_diff,
        "requirement_volatility": volatility,
        "risk_level": risk,
    }


def build_feature_vector(parsed: dict) -> np.ndarray:
    """Build the feature vector matching the model's expected input format."""
    numeric_features = [
        "team_size", "duration_months", "num_sprints",
        "total_user_stories", "avg_story_points", "velocity_per_sprint",
    ]
    feat = [parsed["numeric"].get(col, 0) for col in numeric_features]

    # One-hot encode categoricals (must match training order)
    cat_config = {
        "complexity_level": ["High", "Low", "Medium"],
        "tech_stack_difficulty": ["Advanced", "Basic", "Intermediate"],
        "requirement_volatility_score": ["High", "Low", "Medium"],
        "risk_level": ["High", "Low", "Medium"],
    }
    cat_mapping = {
        "complexity_level": parsed["complexity_level"],
        "tech_stack_difficulty": parsed["tech_stack_difficulty"],
        "requirement_volatility_score": parsed["requirement_volatility"],
        "risk_level": parsed["risk_level"],
    }

    for col, categories in cat_config.items():
        val = cat_mapping[col]
        for cat in categories:
            feat.append(1.0 if val == cat else 0.0)

    return np.array([feat], dtype=np.float32)


def predict_with_model(text: str) -> dict:
    """Run prediction using the trained BERT model with MC Dropout."""
    encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    parsed = extract_features_from_text(text)
    raw_features = build_feature_vector(parsed)
    scaled_features = feature_scaler.transform(raw_features)
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    # MC Dropout inference
    model.train()  # keep dropout active
    mu_samples = []
    sigma_samples = []

    # 5 passes on CPU (~45s), 20 on GPU (~2s)
    n_passes = 20 if device.type == "cuda" else 5
    with torch.no_grad():
        for _ in range(n_passes):
            use_amp = device.type == "cuda"
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                mu, sigma = model(input_ids, attention_mask, features_tensor)
            mu_samples.append(mu.float().cpu().item())
            sigma_samples.append(sigma.float().cpu().item())

    mu_mean = np.mean(mu_samples)
    mu_std = np.std(mu_samples)
    sigma_mean = np.mean(sigma_samples)

    # Convert to USD
    predicted_cost = target_scaler.inverse_transform(np.array([mu_mean]))[0]
    predicted_cost = max(0, predicted_cost)

    aleatoric_usd = abs(sigma_mean * target_scaler.std) * predicted_cost * 0.01
    epistemic_usd = abs(mu_std * target_scaler.std) * predicted_cost * 0.01
    total_usd = np.sqrt(aleatoric_usd ** 2 + epistemic_usd ** 2)

    z90 = 1.645
    lower = max(0, predicted_cost - z90 * total_usd)
    upper = predicted_cost + z90 * total_usd

    return {
        "predicted_cost": round(predicted_cost, 2),
        "aleatoric_uncertainty": round(aleatoric_usd, 2),
        "epistemic_uncertainty": round(epistemic_usd, 2),
        "total_uncertainty": round(total_usd, 2),
        "confidence_interval_low": round(lower, 2),
        "confidence_interval_high": round(upper, 2),
        "parsed_features": parsed,
        "model_used": "BERT-Large (trained)",
        "device": str(device),
        "mc_passes": n_passes,
    }


def predict_heuristic(text: str) -> dict:
    """Fallback prediction using the deterministic formula."""
    parsed = extract_features_from_text(text)
    v = parsed["numeric"]

    complexity_mult = {"Low": 1.0, "Medium": 1.35, "High": 1.65}.get(parsed["complexity_level"], 1.0)
    tech_mult = {"Basic": 1.0, "Intermediate": 1.25, "Advanced": 1.5}.get(parsed["tech_stack_difficulty"], 1.0)
    volatility_mult = {"Low": 1.0, "Medium": 1.10, "High": 1.25}.get(parsed["requirement_volatility"], 1.0)

    base_effort = v["team_size"] * v["duration_months"] * 160
    story_effort = v["total_user_stories"] * v["avg_story_points"] * 4
    final_effort = (base_effort + story_effort) * complexity_mult * tech_mult * volatility_mult

    base_rate = {
        "E-commerce": 42, "Healthcare": 48, "FinTech": 50,
        "EdTech": 38, "SaaS": 45, "AI Platform": 52,
        "IoT": 44, "Cloud System": 46,
    }.get(parsed["domain"], 45)
    risk_mult = {"Low": 0.95, "Medium": 1.0, "High": 1.1}.get(parsed["risk_level"], 1.0)
    cost_per_hour = base_rate * risk_mult

    predicted_cost = final_effort * cost_per_hour
    aleatoric = predicted_cost * 0.05
    epistemic = predicted_cost * 0.03
    total_unc = math.sqrt(aleatoric**2 + epistemic**2)

    return {
        "predicted_cost": round(predicted_cost, 2),
        "aleatoric_uncertainty": round(aleatoric, 2),
        "epistemic_uncertainty": round(epistemic, 2),
        "total_uncertainty": round(total_unc, 2),
        "confidence_interval_low": round(predicted_cost - 1.645 * total_unc, 2),
        "confidence_interval_high": round(predicted_cost + 1.645 * total_unc, 2),
        "parsed_features": parsed,
        "model_used": "Heuristic (formula-based)",
        "device": "cpu",
        "mc_passes": 0,
    }


# ─── app ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agile Cost Estimator API",
    description="API for Software Cost Estimation in Agile Methodology using Deep Learning",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── helper: load JSON file ─────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    with open(path, "r") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════
# RESULTS ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.get("/api/results/metrics")
def get_metrics():
    return _load_json(RESULTS_DIR / "metrics.json")

@app.get("/api/results/history")
def get_training_history():
    return _load_json(RESULTS_DIR / "training_history.json")

@app.get("/api/results/feature-importance")
def get_feature_importance():
    return _load_json(ANALYSIS_DIR / "feature_importance.json")


# ═════════════════════════════════════════════════════════════════════════
# PLOT ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.get("/api/plots/eval/{filename}")
def get_eval_plot(filename: str):
    path = PLOTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
    return FileResponse(path, media_type="image/png")

@app.get("/api/plots/analysis/{filename}")
def get_analysis_plot(filename: str):
    path = ANALYSIS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found")
    return FileResponse(path, media_type="image/png")

@app.get("/api/plots/list")
def list_plots():
    eval_plots = sorted([f.name for f in PLOTS_DIR.glob("*.png")]) if PLOTS_DIR.exists() else []
    analysis_plots = sorted([f.name for f in ANALYSIS_DIR.glob("*.png")]) if ANALYSIS_DIR.exists() else []
    return {"eval": eval_plots, "analysis": analysis_plots}


# ═════════════════════════════════════════════════════════════════════════
# DATASET ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.get("/api/dataset/stats")
def get_dataset_stats():
    if not LABELS_CSV.exists():
        raise HTTPException(status_code=404, detail="labels.csv not found")
    df = pd.read_csv(LABELS_CSV)
    cost = df["actual_cost_usd"]
    effort = df["actual_effort_hours"]
    stats = {
        "total_projects": len(df),
        "cost": {
            "min": round(float(cost.min()), 2),
            "max": round(float(cost.max()), 2),
            "mean": round(float(cost.mean()), 2),
            "median": round(float(cost.median()), 2),
            "std": round(float(cost.std()), 2),
        },
        "effort": {
            "min": round(float(effort.min()), 2),
            "max": round(float(effort.max()), 2),
            "mean": round(float(effort.mean()), 2),
            "median": round(float(effort.median()), 2),
        },
        "domains": df["domain"].value_counts().to_dict(),
        "complexity_levels": df["complexity_level"].value_counts().to_dict(),
        "tech_stack_difficulty": df["tech_stack_difficulty"].value_counts().to_dict(),
        "risk_levels": df["risk_level"].value_counts().to_dict(),
        "team_size": {
            "min": int(df["team_size"].min()),
            "max": int(df["team_size"].max()),
            "mean": round(float(df["team_size"].mean()), 1),
        },
        "duration_months": {
            "min": int(df["duration_months"].min()),
            "max": int(df["duration_months"].max()),
            "mean": round(float(df["duration_months"].mean()), 1),
        },
        "domain_avg_cost": {
            k: round(v, 2)
            for k, v in df.groupby("domain")["actual_cost_usd"].mean().sort_values(ascending=False).to_dict().items()
        },
        "complexity_avg_cost": {
            k: round(v, 2)
            for k, v in df.groupby("complexity_level")["actual_cost_usd"].mean().to_dict().items()
        },
    }
    return stats

@app.get("/api/dataset/metadata")
def get_dataset_metadata():
    return _load_json(DATASET_DIR / "metadata.json")

@app.get("/api/dataset/sample")
def get_dataset_sample(n: int = 20):
    if not LABELS_CSV.exists():
        raise HTTPException(status_code=404, detail="labels.csv not found")
    df = pd.read_csv(LABELS_CSV)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return sample.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS  (Document-based — the core feature)
# ═════════════════════════════════════════════════════════════════════════

class ReportPredictionRequest(BaseModel):
    report_text: str = Field(description="Full text of the Agile project report")


@app.post("/api/predict")
def predict_from_report(req: ReportPredictionRequest):
    """
    Predict software cost from a project report document.
    Uses the trained BERT-Large model if available, otherwise falls back
    to heuristic formula.
    """
    text = req.report_text.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Report text too short (minimum 50 characters)")

    if MODEL_LOADED:
        try:
            return predict_with_model(text)
        except Exception as e:
            print(f"  ⚠️  Model prediction failed, falling back: {e}")
            return predict_heuristic(text)
    else:
        return predict_heuristic(text)


@app.post("/api/predict/upload")
async def predict_from_file(file: UploadFile = File(...)):
    """
    Upload a .txt project report file and predict its cost.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    content = await file.read()
    text = content.decode("utf-8").strip()

    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Report text too short (minimum 50 characters)")

    if MODEL_LOADED:
        try:
            return predict_with_model(text)
        except Exception as e:
            print(f"  ⚠️  Model prediction failed, falling back: {e}")
            return predict_heuristic(text)
    else:
        return predict_heuristic(text)


@app.get("/api/predict/sample-report")
def get_sample_report():
    """Return a sample report for testing."""
    sample_path = DATASET_DIR / "reports" / "project_00001.txt"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample report not found")
    with open(sample_path, "r", encoding="utf-8") as f:
        return {"filename": "project_00001.txt", "text": f.read()}


# ═════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_device": str(device) if device else "none",
        "dataset_available": LABELS_CSV.exists(),
        "results_available": (RESULTS_DIR / "metrics.json").exists(),
        "plots_available": PLOTS_DIR.exists(),
        "checkpoint_available": CHECKPOINT_PATH.exists(),
    }


# ─── entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
