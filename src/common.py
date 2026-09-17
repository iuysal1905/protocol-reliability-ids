"""Shared utilities for both benchmark pipelines.

Metrics, the tabular network, seeding and preprocessing helpers used by the
CICIoT2023 and Edge-IIoTset pipelines. Keeping them here is deliberate: in the
original notebook the two benchmarks shared one namespace and later cells
rebound names such as ``y_train`` and ``train_one_epoch`` that earlier cells
still relied on. Importing from a module instead removes that class of error.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn as nn

from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------

def quantile_ece(y_true, y_prob, n_bins: int = 15):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.clip(
        np.asarray(y_prob, dtype=np.float64).ravel(),
        1e-7,
        1.0 - 1e-7,
    )

    order = np.argsort(y_prob)
    bins = np.array_split(order, n_bins)

    weighted_gaps = []
    absolute_gaps = []

    for idx in bins:
        if len(idx) == 0:
            continue

        observed = float(y_true[idx].mean())
        predicted = float(y_prob[idx].mean())
        gap = abs(observed - predicted)

        weighted_gaps.append(gap * len(idx) / len(y_true))
        absolute_gaps.append(gap)

    return float(np.sum(weighted_gaps)), float(np.max(absolute_gaps))


def evaluate_binary(y_true, y_prob, threshold: float = 0.50):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.clip(
        np.asarray(y_prob, dtype=np.float64).ravel(),
        1e-7,
        1.0 - 1e-7,
    )
    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    ece, mce = quantile_ece(y_true, y_prob, n_bins=15)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": (
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(
            y_true, y_prob, labels=[0, 1]
        ),
        "ece": ece,
        "mce": mce,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def binary_counts(y):
    counts = np.bincount(
        np.asarray(y, dtype=np.int64).ravel(),
        minlength=2,
    )
    return int(counts[0]), int(counts[1])


def family_js_divergence(
    family_train: pd.Series,
    family_test: pd.Series,
) -> float:
    labels = sorted(
        set(family_train.astype(str).unique())
        | set(family_test.astype(str).unique())
    )

    p = (
        family_train.astype(str)
        .value_counts(normalize=True)
        .reindex(labels, fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )

    q = (
        family_test.astype(str)
        .value_counts(normalize=True)
        .reindex(labels, fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )

    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a, b):
        mask = a > 0
        return float(
            np.sum(
                a[mask]
                * np.log2(
                    (a[mask] + eps)
                    / (b[mask] + eps)
                )
            )
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------
# 4. MLP
# ---------------------------------------------------------

class TabularMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(
    X,
    y,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(
            np.asarray(y).reshape(-1, 1),
            dtype=torch.float32,
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_mlp(model, loader):
    model.eval()

    y_true_parts = []
    y_prob_parts = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probabilities = torch.sigmoid(logits)

        y_true_parts.append(
            yb.cpu().numpy().ravel()
        )
        y_prob_parts.append(
            probabilities.cpu().numpy().ravel()
        )

    return (
        np.concatenate(y_true_parts).astype(np.int64),
        np.concatenate(y_prob_parts),
    )


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed: int,
):
    set_seed(seed)

    train_loader = make_loader(
        X_train,
        y_train,
        MLP_BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        MLP_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    test_loader = make_loader(
        X_test,
        y_test,
        MLP_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    model = TabularMLP(
        input_dim=X_train.shape[1]
    ).to(DEVICE)

    positive_count = float(
        (np.asarray(y_train) == 1).sum()
    )
    negative_count = float(
        (np.asarray(y_train) == 0).sum()
    )

    positive_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )

    best_state = None
    best_val_auc = -np.inf
    best_epoch = -1
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, MLP_EPOCHS + 1):
        train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_true, val_prob = predict_mlp(
            model,
            val_loader,
        )

        if np.unique(val_true).size < 2:
            raise RuntimeError(
                "Inner validation contains one class."
            )

        if not np.isfinite(val_prob).all():
            raise RuntimeError(
                "MLP produced non-finite validation probabilities."
            )

        val_auc = roc_auc_score(
            val_true,
            val_prob,
        )

        if (
            best_state is None
            or val_auc > best_val_auc
        ):
            best_val_auc = float(val_auc)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= MLP_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            "No valid MLP checkpoint was obtained."
        )

    training_time = time.time() - start_time

    model.load_state_dict(best_state)
    model.eval()

    test_true, test_prob = predict_mlp(
        model,
        test_loader,
    )

    return {
        "probability": test_prob,
        "best_epoch": best_epoch,
        "best_validation_roc_auc": best_val_auc,
        "training_time_sec": training_time,
    }

