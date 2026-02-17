"""
dataset.py
==========
PyTorch Dataset for loading Agile project reports + structured features.
Handles tokenisation, feature scaling, and train/val/test splitting.
"""

import os
import csv
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config


# ──────────────────────────── helpers ────────────────────────────────────

def _load_labels(csv_path: str) -> List[Dict[str, str]]:
    """Load labels.csv and return list of row dicts."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministically shuffle and split indices into train / val / test.
    """
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


# ──────────────────────────── feature scaler ─────────────────────────────

class FeatureScaler:
    """StandardScaler fitted on train split only, applied to all splits."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        """Compute mean and std from training features."""
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        # avoid division by zero for constant features
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardise features using pre-fitted mean/std."""
        assert self.mean is not None, "Scaler has not been fitted."
        return (features - self.mean) / self.std

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: Dict[str, np.ndarray]) -> None:
        self.mean = d["mean"]
        self.std = d["std"]


# ──────────────────────────── target scaler ──────────────────────────────

class TargetScaler:
    """Log + StandardScaler for the cost target."""

    def __init__(self, use_log: bool = True):
        self.use_log = use_log
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, values: np.ndarray) -> "TargetScaler":
        if self.use_log:
            values = np.log1p(values)
        self.mean = float(values.mean())
        self.std = float(values.std())
        if self.std == 0:
            self.std = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.use_log:
            values = np.log1p(values)
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Convert normalised predictions back to dollar values."""
        values = values * self.std + self.mean
        if self.use_log:
            values = np.expm1(values)
        return values

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "use_log": self.use_log}

    def load_state_dict(self, d: Dict) -> None:
        self.mean = d["mean"]
        self.std = d["std"]
        self.use_log = d["use_log"]


# ──────────────────────────── dataset class ──────────────────────────────

class AgileCostDataset(Dataset):
    """
    PyTorch Dataset for Agile cost estimation.

    Each sample contains:
        - input_ids, attention_mask  : tokenised report text
        - numeric_features           : scaled structured features
        - target                     : scaled cost label
        - raw_cost                   : original dollar cost (for evaluation)
    """

    # class-level shared state so all splits use the same scaler
    _all_rows: Optional[List[Dict]] = None
    _split_map: Optional[Dict[str, List[int]]] = None
    _feature_scaler: Optional[FeatureScaler] = None
    _target_scaler: Optional[TargetScaler] = None
    _tokenizer = None

    def __init__(self, split: str = "train"):
        """
        Args:
            split: one of 'train', 'val', 'test'
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split

        # lazily load everything once
        if AgileCostDataset._all_rows is None:
            self._init_shared_state()

        indices = AgileCostDataset._split_map[split]
        self.rows = [AgileCostDataset._all_rows[i] for i in indices]

        # extract numeric features
        raw_features = self._extract_features(self.rows)

        if split == "train":
            self.features = AgileCostDataset._feature_scaler.fit_transform(raw_features)
        else:
            self.features = AgileCostDataset._feature_scaler.transform(raw_features)

        # extract target
        raw_costs = np.array(
            [float(r[config.TARGET_COLUMN]) for r in self.rows], dtype=np.float32
        )
        self.raw_costs = raw_costs

        if split == "train":
            self.targets = AgileCostDataset._target_scaler.fit(raw_costs).transform(
                raw_costs
            )
        else:
            self.targets = AgileCostDataset._target_scaler.transform(raw_costs)

    # ── shared initialisation ────────────────────────────────────────────

    @classmethod
    def _init_shared_state(cls):
        cls._all_rows = _load_labels(config.LABELS_CSV)
        n = len(cls._all_rows)

        train_idx, val_idx, test_idx = _split_indices(
            n, config.TRAIN_RATIO, config.VAL_RATIO, config.SEED
        )
        cls._split_map = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        cls._feature_scaler = FeatureScaler()
        cls._target_scaler = TargetScaler(use_log=config.LOG_TRANSFORM_TARGET)
        cls._tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    @classmethod
    def reset(cls):
        """Reset shared state (useful if re-initialising with different config)."""
        cls._all_rows = None
        cls._split_map = None
        cls._feature_scaler = None
        cls._target_scaler = None
        cls._tokenizer = None

    @classmethod
    def get_feature_scaler(cls) -> FeatureScaler:
        return cls._feature_scaler

    @classmethod
    def get_target_scaler(cls) -> TargetScaler:
        return cls._target_scaler

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        return cls._tokenizer

    # ── feature extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_features(rows: List[Dict]) -> np.ndarray:
        """Extract numeric + one-hot-encoded categorical features."""
        records = []
        for row in rows:
            feat = []
            # continuous features
            for col in config.NUMERIC_FEATURES:
                feat.append(float(row[col]))
            # categorical features (one-hot)
            for col, categories in config.CATEGORICAL_FEATURES.items():
                val = row[col]
                for cat in categories:
                    feat.append(1.0 if val == cat else 0.0)
            records.append(feat)
        return np.array(records, dtype=np.float32)

    # ── text loading ─────────────────────────────────────────────────────

    @staticmethod
    def _load_report(project_id: str) -> str:
        """Read the .txt report file."""
        path = os.path.join(config.REPORTS_DIR, f"{project_id}.txt")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ── dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        text = self._load_report(row["project_id"])

        encoding = AgileCostDataset._tokenizer(
            text,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # [512]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [512]
            "numeric_features": torch.tensor(
                self.features[idx], dtype=torch.float32
            ),
            "target": torch.tensor(
                self.targets[idx], dtype=torch.float32
            ),
            "raw_cost": torch.tensor(
                self.raw_costs[idx], dtype=torch.float32
            ),
        }


# ──────────────────────────── dataloader factory ─────────────────────────

def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders.
    """
    train_ds = AgileCostDataset("train")
    val_ds = AgileCostDataset("val")
    test_ds = AgileCostDataset("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"  Dataset splits — Train: {len(train_ds):,}  "
          f"Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Numeric features dim: {config.NUM_NUMERIC_FEATURES}")

    return train_loader, val_loader, test_loader
