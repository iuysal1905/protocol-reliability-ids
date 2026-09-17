"""Fit and score every model on one partition.

All Edge-IIoTset arms - the row-level reference, the ten endpoint-pair-disjoint
partitions, the two matched controls and the transport-identity ablation - go
through ``run_partition``. Sharing one function is what makes the arms
comparable: the preprocessing sequence, the estimator settings and the fixed
decision threshold cannot drift between them.

Preprocessing is fitted on inner-training rows only and applied to the
validation and test rows, separately within every partition.
"""

from __future__ import annotations

import gc
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .common import binary_counts, evaluate_binary, train_mlp

FIXED_THRESHOLD = 0.50
INNER_VALIDATION_SIZE = 0.15
MISSING_COLUMN_LIMIT = 0.95


def run_partition(
    predictors,
    labels,
    train_index,
    test_index,
    arm_label,
    split_id,
    seed_offset,
    feature_blocklist=None,
    val_index=None,
    mlp_seeds=None,
    tree_seed=None,
    threshold=FIXED_THRESHOLD,
):
    """Fit the always-attack reference, the MLP and three tree models.

    ``feature_blocklist`` drives the transport-identity ablation.
    ``val_index`` is supplied when the protocol carves validation from the same
    pool as the test partition rather than from the training rows; the
    row-level reference does this and the endpoint-pair protocol does not.
    """
    blocked = set(feature_blocklist or [])

    if val_index is None:
        inner_train_index, inner_val_index = train_test_split(
            train_index,
            test_size=INNER_VALIDATION_SIZE,
            random_state=20_000 + seed_offset,
            stratify=labels[train_index],
        )
    else:
        inner_train_index = np.asarray(train_index)
        inner_val_index = np.asarray(val_index)

    y_train = labels[inner_train_index]
    y_val = labels[inner_val_index]
    y_test = labels[test_index]

    if min(binary_counts(y_train)) == 0 or min(binary_counts(y_val)) == 0:
        raise RuntimeError(
            f"{arm_label} split {split_id}: an inner partition holds one class."
        )

    usable = [c for c in predictors.columns if c not in blocked]

    raw_train = predictors.iloc[inner_train_index][usable]
    raw_val = predictors.iloc[inner_val_index][usable]
    raw_test = predictors.iloc[test_index][usable]

    missing_ratio = raw_train.isna().mean()
    retained = missing_ratio[missing_ratio <= MISSING_COLUMN_LIMIT].index.tolist()

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(raw_train[retained])
    val_imputed = imputer.transform(raw_val[retained])
    test_imputed = imputer.transform(raw_test[retained])

    nonconstant = np.var(train_imputed, axis=0) > 1e-12
    feature_names = np.asarray(retained)[nonconstant].tolist()

    train_tree = np.asarray(train_imputed[:, nonconstant], dtype=np.float32)
    val_tree = np.asarray(val_imputed[:, nonconstant], dtype=np.float32)
    test_tree = np.asarray(test_imputed[:, nonconstant], dtype=np.float32)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_tree).astype(np.float32)
    val_scaled = scaler.transform(val_tree).astype(np.float32)
    test_scaled = scaler.transform(test_tree).astype(np.float32)

    rows = []

    def record(model_name, probability, training_time, seed_label=None):
        metrics = evaluate_binary(y_test, probability, threshold=threshold)
        metrics.update({
            "arm": arm_label,
            "split_id": split_id,
            "model": model_name,
            "seed": seed_label,
            "training_time_sec": training_time,
            "feature_count": len(feature_names),
            "test_rows": int(len(test_index)),
            "test_attack_ratio": float(y_test.mean()),
            "train_attack_ratio": float(y_train.mean()),
            "predicted_positive_rate": float((probability >= threshold).mean()),
            "mean_predicted_probability": float(np.mean(probability)),
        })
        rows.append(metrics)

    record(
        "Always attack baseline",
        np.ones(len(y_test), dtype=np.float64),
        0.0,
    )

    for seed in (mlp_seeds if mlp_seeds is not None else [100 + seed_offset]):
        output = train_mlp(
            train_scaled, y_train,
            val_scaled, y_val,
            test_scaled, y_test,
            seed=seed,
        )
        record(
            "Tabular MLP",
            output["probability"],
            output["training_time_sec"],
            seed_label=seed,
        )

    positive = float((y_train == 1).sum())
    negative = float((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive > 0 else 1.0

    model_seed = tree_seed if tree_seed is not None else 200 + seed_offset

    estimators = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=model_seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=model_seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.80,
            colsample_bytree=0.80,
            class_weight="balanced",
            random_state=model_seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    for model_name, estimator in estimators.items():
        started = time.time()
        estimator.fit(train_tree, y_train)
        elapsed = time.time() - started

        record(
            model_name,
            estimator.predict_proba(test_tree)[:, 1],
            elapsed,
            seed_label=model_seed,
        )

        del estimator
        gc.collect()

    return rows, feature_names
