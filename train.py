"""
train.py
========
Production-level training loop for the Agile Cost Estimator.

2-Phase strategy:
    Phase 1 (epochs 1-5)  : BERT frozen  → train regression head only
    Phase 2 (epochs 6-25) : BERT unfrozen → fine-tune entire model with
                            discriminative learning rates

H200 GPU optimisations:
    - BFloat16 mixed precision (AMP)
    - Batch size 128
    - Pinned memory + 8 DataLoader workers
    - Gradient clipping (max norm 1.0)
    - Cosine annealing LR with linear warmup
"""

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import config
from dataset import AgileCostDataset, get_dataloaders
from model import AgileCostEstimator, GaussianNLLLoss


# ──────────────────────────── seed everything ────────────────────────────

def seed_everything(seed: int = config.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ──────────────────────────── LR scheduler with warmup ───────────────────

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Cosine annealing with linear warmup.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ──────────────────────────── metrics ────────────────────────────────────

def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    target_scaler,
) -> Dict[str, float]:
    """
    Compute R², MAPE, MAE, RMSE on original dollar scale.
    """
    # inverse-transform to original scale
    preds_usd = target_scaler.inverse_transform(preds)
    targets_usd = target_scaler.inverse_transform(targets)

    # clip predictions to be non-negative
    preds_usd = np.maximum(preds_usd, 0)

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

    return {"r2": round(r2, 4), "mape": round(mape, 2), "mae": round(mae, 2), "rmse": round(rmse, 2)}


# ──────────────────────────── training engine ────────────────────────────

class Trainer:
    """Production training engine with 2-phase strategy."""

    def __init__(self):
        seed_everything()

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n{'='*60}")
        print(f"  Agile Cost Estimator — Training Pipeline")
        print(f"  Device : {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU    : {gpu_name} ({gpu_mem:.0f} GB)")
        print(f"{'='*60}\n")

        # data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()
        self.target_scaler = AgileCostDataset.get_target_scaler()
        self.feature_scaler = AgileCostDataset.get_feature_scaler()

        # model
        self.model = AgileCostEstimator().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,} total")

        # loss
        self.criterion = GaussianNLLLoss()

        # AMP scaler
        self.scaler = GradScaler(enabled=config.USE_AMP)

        # checkpointing
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_r2": [],
            "val_mape": [],
            "epoch_time": [],
        }

    # ── single epoch ─────────────────────────────────────────────────────

    def _train_one_epoch(self, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            features = batch["numeric_features"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.USE_AMP):
                mu, sigma = self.model(input_ids, attention_mask, features)
                loss = self.criterion(mu, sigma, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_GRAD_NORM)
            self.scaler.step(optimizer)
            self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── validation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            features = batch["numeric_features"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.USE_AMP):
                mu, sigma = self.model(input_ids, attention_mask, features)
                loss = self.criterion(mu, sigma, targets)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(mu.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        metrics = compute_metrics(preds, targets, self.target_scaler)

        return avg_loss, metrics

    # ── checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: dict):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "feature_scaler": self.feature_scaler.state_dict(),
            "target_scaler": self.target_scaler.state_dict(),
            "config": {
                "model_name": config.MODEL_NAME,
                "max_seq_length": config.MAX_SEQ_LENGTH,
                "num_numeric_features": config.NUM_NUMERIC_FEATURES,
                "dropout_rate": config.DROPOUT_RATE,
            },
        }
        path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
        torch.save(checkpoint, path)
        print(f"    💾 Best model saved → {path}")

    # ── main training loop ───────────────────────────────────────────────

    def train(self):
        """
        Full 2-phase training.
        """
        print("\n" + "─" * 60)
        print("  PHASE 1: BERT Frozen — Training Regression Head")
        print("─" * 60)

        # Phase 1: freeze BERT
        self.model.freeze_bert()
        optimizer_p1 = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.LR_HEAD,
            weight_decay=config.WEIGHT_DECAY,
        )
        steps_p1 = config.NUM_EPOCHS_PHASE1 * len(self.train_loader)
        scheduler_p1 = get_cosine_schedule_with_warmup(
            optimizer_p1, min(config.WARMUP_STEPS, steps_p1 // 2), steps_p1
        )

        for epoch in range(1, config.NUM_EPOCHS_PHASE1 + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(optimizer_p1, scheduler_p1)
            val_loss, metrics = self._validate()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(metrics["r2"])
            self.history["val_mape"].append(metrics["mape"])
            self.history["epoch_time"].append(elapsed)

            print(
                f"  Epoch {epoch:2d}/{config.TOTAL_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"R²: {metrics['r2']:.4f} | MAPE: {metrics['mape']:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, metrics)
            else:
                self.patience_counter += 1

        # Phase 2: unfreeze BERT
        print("\n" + "─" * 60)
        print("  PHASE 2: BERT Unfrozen — Fine-tuning Entire Model")
        print("─" * 60)

        self.model.unfreeze_bert()
        optimizer_p2 = AdamW(
            self.model.get_parameter_groups(),
            weight_decay=config.WEIGHT_DECAY,
        )
        steps_p2 = config.NUM_EPOCHS_PHASE2 * len(self.train_loader)
        scheduler_p2 = get_cosine_schedule_with_warmup(
            optimizer_p2, config.WARMUP_STEPS, steps_p2
        )
        self.patience_counter = 0  # reset for phase 2

        for epoch in range(
            config.NUM_EPOCHS_PHASE1 + 1, config.TOTAL_EPOCHS + 1
        ):
            t0 = time.time()
            train_loss = self._train_one_epoch(optimizer_p2, scheduler_p2)
            val_loss, metrics = self._validate()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_r2"].append(metrics["r2"])
            self.history["val_mape"].append(metrics["mape"])
            self.history["epoch_time"].append(elapsed)

            print(
                f"  Epoch {epoch:2d}/{config.TOTAL_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"R²: {metrics['r2']:.4f} | MAPE: {metrics['mape']:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n  ⏹  Early stopping at epoch {epoch} "
                          f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
                    break

        # save history
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n  📊 Training history saved → {history_path}")

        # final summary
        best_r2 = max(self.history["val_r2"])
        best_mape = min(self.history["val_mape"])
        total_time = sum(self.history["epoch_time"])
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Best R²   : {best_r2:.4f}")
        print(f"  Best MAPE : {best_mape:.2f}%")
        print(f"  Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")


# ──────────────────────────── entry point ────────────────────────────────

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
