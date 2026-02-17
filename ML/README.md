# Software Cost Estimation in Agile Methodology using Deep Learning

A production-grade deep learning system that predicts software project cost from Agile Scrum reports, featuring **BERT-Large NLP encoding**, **multi-input regression**, and **cost uncertainty modeling**.

---

## 🎯 Overview

This project trains a neural network to read natural-language Agile project reports and predict the development cost in USD — along with a confidence interval quantifying model uncertainty.

```
Project Report (.txt) → BERT-Large Encoder → ┐
                                              ├→ μ (Predicted Cost: $245,000)
Structured Features   → Feature MLP ────────→ ┘→ σ (Uncertainty:   ±$18,500)
```

### Key Features

- **10,000 synthetic Agile project reports** with deterministic cost labels
- **BERT-Large** (340M params) text encoder for rich report understanding
- **Dual-head output** — predicts both cost (μ) and uncertainty (σ)
- **MC Dropout** inference — epistemic + aleatoric uncertainty quantification
- **2-phase training** — freeze BERT, then fine-tune with discriminative LR
- **8 publication-quality plots** for evaluation
- **H200 GPU optimised** — BFloat16 AMP, batch size 128, pinned memory

---

## 📁 Project Structure

```
agile/
├── agile_cost_dataset/           # Generated training data
│   ├── reports/                  # 10,000 project report .txt files
│   ├── labels.csv                # Ground-truth cost labels + metadata
│   └── metadata.json             # Dataset generation info
│
├── generate_dataset.py           # Synthetic data generator
├── config.py                     # All hyperparameters & paths
├── dataset.py                    # PyTorch Dataset + feature scaling
├── model.py                      # BERT-Large + dual-head regression
├── train.py                      # 2-phase training loop
├── evaluate.py                   # Metrics + 8 plots + uncertainty
├── predict.py                    # Inference with confidence intervals
├── analysis.py                   # Dataset analysis & feature importance
├── requirements.txt              # Python dependencies
│
├── checkpoints/                  # Trained model (Git LFS tracked)
│   └── best_model.pt
└── results/                      # Evaluation outputs
    ├── metrics.json
    ├── training_history.json
    ├── plots/                    # 8 publication plots
    └── analysis/                 # Dataset analysis plots
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA (optimised for H200 141 GB VRAM)
- Git LFS installed

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd agile
git lfs pull                      # Download model checkpoints
pip install -r requirements.txt
```

### 2. Generate Dataset (if not present)

```bash
python generate_dataset.py
```

Produces 10,000 Agile project reports in `agile_cost_dataset/`.

### 3. Train the Model

```bash
python train.py
```

| Phase | Epochs | Strategy |
|-------|--------|----------|
| 1 — Warmup    | 1–5   | BERT frozen, train regression head |
| 2 — Fine-tune | 6–25  | Full model, discriminative LR (BERT: 5e-6, Head: 2e-5) |

### 4. Evaluate

```bash
python evaluate.py
```

Outputs: R², MAPE, MAE, RMSE, calibration error, PICP, and 8 plots.

### 5. Predict New Project Cost

```bash
# Single report
python predict.py agile_cost_dataset/reports/project_00001.txt

# Batch prediction
python predict.py agile_cost_dataset/reports/ --batch
```

**Example output:**
```
  Predicted Cost : $245,000.00
  Aleatoric unc. : ±$8,200.00
  Epistemic unc. : ±$5,100.00
  Total unc.     : ±$9,650.00
  90% CI         : $229,125.00 – $260,875.00
```

### 6. Dataset Analysis

```bash
python analysis.py
```

---

## 🏗️ Architecture

### Multi-Input BERT + Regression with Uncertainty

```
┌────────────────────────────────────────────────────────────┐
│  Input                                                     │
│  ├── Report text → BERT-Large → CLS token (1024-dim)       │
│  └── Numeric features → MLP (128 → 64-dim)                 │
│                                                            │
│  Fusion: Concatenate (1024 + 64 = 1088-dim)                │
│                                                            │
│  Output                                                    │
│  ├── μ Head: 1088 → 512 → 256 → 1  (predicted cost)        │
│  └── σ Head: 1088 → 256 → 1       (uncertainty, softplus)  │
└────────────────────────────────────────────────────────────┘
```

### Cost Uncertainty Modeling

| Type | Measures | Method |
|------|----------|--------|
| **Aleatoric** | Data-inherent noise | σ head trained with Gaussian NLL loss |
| **Epistemic** | Model confidence | MC Dropout (20 forward passes at inference) |
| **Total** | Combined | √(aleatoric² + epistemic²) |

---

## 📊 Cost Label Generation

Labels are computed deterministically — not random — ensuring strong, learnable signal:

```
final_effort = (team_size × duration × 160 + stories × avg_sp × 4)
             × complexity_mult × tech_mult × volatility_mult

final_cost = final_effort × cost_per_hour × (1 ± 5% noise)
```

| Multiplier | Low | Medium | High |
|------------|-----|--------|------|
| Complexity | 1.0 | 1.35   | 1.65 |
| Tech Stack | 1.0 (Basic) | 1.25 (Intermediate) | 1.5 (Advanced) |
| Volatility | 1.0 | 1.10   | 1.25 |

---

## 📈 Evaluation Plots

1. Training & validation loss curves
2. Predicted vs Actual scatter (with R²)
3. Residual distribution
4. Error by complexity level (box plot)
5. Error by domain (box plot)
6. Cumulative error distribution
7. Uncertainty calibration plot
8. Prediction intervals with 90% CI

---

## ⚡ H200 GPU Optimisations

| Setting | Value | Rationale |
|---------|-------|-----------|
| Mixed precision | BFloat16 | Native H200 support, 2× throughput |
| Batch size | 128 | 141 GB VRAM handles BERT-Large easily |
| Pin memory | Enabled | Faster host → device transfer |
| Data workers | 8 | Saturate GPU with parallel loading |
| Model | `bert-large-uncased` | 340M params — best quality encoder |

---

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **NLP**: BERT-Large (340M parameters)
- **Data**: pandas, NumPy, scikit-learn
- **Visualisation**: matplotlib, seaborn
- **Version Control**: Git + Git LFS (for model checkpoints)

---

## 📄 License

This project is developed for academic research purposes.

---

## 👥 Authors

Software Cost Estimation in Agile Methodology using Deep Learning — Research Project
