"""
model.py
========
Multi-input BERT-Large + Regression model with dual-head uncertainty output.

Architecture:
    Report text  → BERT-Large → CLS token → 1024-dim
    Numeric feats → MLP → 64-dim
    Concatenation (1088-dim) → μ head (predicted cost)
                              → σ head (predicted uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import ML.config


class FeatureMLP(nn.Module):
    """Small MLP to project structured numeric features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MuHead(nn.Module):
    """Regression head for predicted cost (μ)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


class SigmaHead(nn.Module):
    """Uncertainty head for predicted standard deviation (σ).
    Uses softplus to ensure σ > 0."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x).squeeze(-1))  # [B], always > 0


class AgileCostEstimator(nn.Module):
    """
    Multi-input model for Agile software cost estimation with
    uncertainty quantification.

    Inputs:
        input_ids       : [B, seq_len] — tokenised report text
        attention_mask  : [B, seq_len]
        numeric_features: [B, num_features] — structured features

    Outputs:
        mu    : [B] — predicted cost (normalised)
        sigma : [B] — predicted aleatoric uncertainty (normalised)
    """

    def __init__(self, num_numeric_features: int = config.NUM_NUMERIC_FEATURES):
        super().__init__()

        # BERT-Large encoder (1024-dim output)
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        bert_dim = self.bert.config.hidden_size  # 1024 for bert-large

        # Structured feature MLP
        self.feature_mlp = FeatureMLP(
            input_dim=num_numeric_features,
            hidden_dim=128,
            output_dim=64,
        )

        # Fusion dimension
        fusion_dim = bert_dim + 64  # 1024 + 64 = 1088

        # Dual heads
        self.mu_head = MuHead(fusion_dim)
        self.sigma_head = SigmaHead(fusion_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.
        Returns (mu, sigma) where both are [B] tensors.
        """
        # Text encoding — use CLS token embedding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [B, 1024]

        # Feature encoding
        feat_embedding = self.feature_mlp(numeric_features)     # [B, 64]

        # Fusion
        fused = torch.cat([cls_embedding, feat_embedding], dim=-1)  # [B, 1088]

        # Dual-head output
        mu = self.mu_head(fused)        # [B]
        sigma = self.sigma_head(fused)  # [B]

        return mu, sigma

    def freeze_bert(self):
        """Freeze all BERT parameters (Phase 1 training)."""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("  ❄️  BERT-Large frozen — training regression head only")

    def unfreeze_bert(self):
        """Unfreeze all BERT parameters (Phase 2 training)."""
        for param in self.bert.parameters():
            param.requires_grad = True
        print("  🔥 BERT-Large unfrozen — fine-tuning entire model")

    def get_parameter_groups(self):
        """
        Return parameter groups with discriminative learning rates.
        BERT gets a lower LR, regression heads get a higher LR.
        """
        bert_params = list(self.bert.parameters())
        head_params = (
            list(self.feature_mlp.parameters())
            + list(self.mu_head.parameters())
            + list(self.sigma_head.parameters())
        )
        return [
            {"params": bert_params, "lr": config.LR_BERT},
            {"params": head_params, "lr": config.LR_HEAD},
        ]


# ──────────────────────────── loss function ──────────────────────────────

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss.
    Trains both μ and σ simultaneously.

    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        variance = sigma.pow(2) + self.eps
        nll = 0.5 * (torch.log(variance) + (target - mu).pow(2) / variance)
        return nll.mean()
