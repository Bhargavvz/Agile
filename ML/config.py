"""
config.py
=========
Centralised configuration for the Agile Cost Estimation training pipeline.
All hyperparameters, paths, and H200-optimised settings live here.
"""

import os

# ──────────────────────────── paths ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "agile_cost_dataset")
REPORTS_DIR = os.path.join(DATASET_DIR, "reports")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# ──────────────────────────── reproducibility ────────────────────────────
SEED = 42

# ──────────────────────────── data splits ────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ──────────────────────────── model ──────────────────────────────────────
MODEL_NAME = "bert-large-uncased"          # 340 M params, 1024-dim output
MAX_SEQ_LENGTH = 512

# Structured numeric features extracted from labels.csv
NUMERIC_FEATURES = [
    "team_size",
    "duration_months",
    "num_sprints",
    "total_user_stories",
    "avg_story_points",
    "velocity_per_sprint",
]

# Categorical features to be one-hot encoded
CATEGORICAL_FEATURES = {
    "complexity_level": ["Low", "Medium", "High"],
    "tech_stack_difficulty": ["Basic", "Intermediate", "Advanced"],
    "requirement_volatility_score": ["Low", "Medium", "High"],
    "risk_level": ["Low", "Medium", "High"],
}

# Total numeric input dim = len(NUMERIC_FEATURES)
#                          + sum(len(v) for v in CATEGORICAL_FEATURES.values())
NUM_NUMERIC_FEATURES = len(NUMERIC_FEATURES) + sum(
    len(v) for v in CATEGORICAL_FEATURES.values()
)

# ──────────────────────────── training — H200 optimised ──────────────────
BATCH_SIZE = 128                           # 141 GB VRAM handles this easily
NUM_EPOCHS_PHASE1 = 5                      # BERT frozen
NUM_EPOCHS_PHASE2 = 20                     # BERT unfrozen
TOTAL_EPOCHS = NUM_EPOCHS_PHASE1 + NUM_EPOCHS_PHASE2

LR_HEAD = 2e-5                             # regression head learning rate
LR_BERT = 5e-6                             # BERT fine-tuning LR (gentle)
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

# Dropout kept enabled at inference for MC Dropout uncertainty
DROPOUT_RATE = 0.1
MC_DROPOUT_PASSES = 20                     # forward passes for epistemic uncertainty

# Mixed precision
USE_AMP = True

# DataLoader
NUM_WORKERS = 8
PIN_MEMORY = True

# ──────────────────────────── target ─────────────────────────────────────
TARGET_COLUMN = "actual_cost_usd"
LOG_TRANSFORM_TARGET = True                # log1p for better regression convergence

# ──────────────────────────── label column for cost ──────────────────────
LABEL_COLUMNS = [
    "project_id", "domain", "complexity_level", "team_size",
    "duration_months", "sprint_length_weeks", "num_sprints",
    "total_user_stories", "avg_story_points", "velocity_per_sprint",
    "tech_stack_difficulty", "requirement_volatility_score",
    "actual_effort_hours", "actual_cost_usd", "risk_level",
]
