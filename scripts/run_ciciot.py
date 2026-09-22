#!/usr/bin/env python3
"""Run the CICIoT2023 pipeline.

Stages:

  1. load the fixed splits, fit preprocessing on training rows only
  2. attribution ranking from training rows              -> Section III-E
  3. feature-budget selection on validation, five seeds  -> Table V
  4. seed stability for the selected profile, ten seeds  -> Table VI
  5. leave-one-family-out with refitted preprocessing    -> Table IX

Stage 2 and stage 3 are the corrected versions. In the first submission the
SHAP explanation sample was drawn from the test partition and the budgets were
compared on test metrics, so the profile that reached the manuscript had been
chosen with the evaluation partition in view. Here the ranking uses training
rows, the selection uses validation rows, and the test partition is read once,
for the selected budget. Test metrics for the other budgets are written to a
separate file and are reported as post-hoc.

Stage 5 refits the median fill and the standardization after the held-out
family is removed, so no statistic from that family enters the preprocessing.

Usage:
    python scripts/run_ciciot.py --train train.csv --val val.csv \\
                                 --test test.csv --out results/ciciot
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import TabularMLP, evaluate_binary, set_seed  # noqa: E402

GLOBAL_SEED = 42
LABEL_COLUMN = "label"
BUDGETS = [5, 10, 15, 20, 30, 40]
BUDGET_SEEDS = [11, 22, 33, 42, 55]
PROFILE_SEEDS = [11, 22, 33, 42, 55, 66, 77, 88, 99, 111]
BATCH_SIZE = 4096

# Training schedules, matching the reported analysis:
#   the full-feature reference network that is explained    20 epochs, patience 5
#   feature budgets and the selected-profile seed runs       12 epochs, patience 4
#   leave-one-family-out folds                               10 epochs, patience 3
REFERENCE_SCHEDULE = (20, 5)
BUDGET_SCHEDULE = (12, 4)
LOFO_SCHEDULE = (10, 3)


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def load_split(path, label_column=LABEL_COLUMN, max_rows=None):
    frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    frame["y_binary"] = (
        frame[label_column].astype(str).str.lower() != "benign"
    ).astype(np.int64)
    return frame


def build_matrices(train_df, val_df, test_df):
    """Fit the median fill and standardization on training rows only."""
    from sklearn.preprocessing import StandardScaler

    ignore = {LABEL_COLUMN, "y_binary"}
    features = [c for c in train_df.columns if c not in ignore]
    features = [c for c in features if train_df[c].nunique(dropna=False) > 1]

    medians = train_df[features].median()

    def clean(frame):
        block = frame[features].replace([np.inf, -np.inf], np.nan)
        return block.fillna(medians).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(clean(train_df)).astype(np.float32)
    X_val = scaler.transform(clean(val_df)).astype(np.float32)
    X_test = scaler.transform(clean(test_df)).astype(np.float32)

    return {
        "features": features,
        "medians": medians,
        "scaler": scaler,
        "X_train": X_train, "y_train": train_df["y_binary"].to_numpy(np.int64),
        "X_val": X_val, "y_val": val_df["y_binary"].to_numpy(np.int64),
        "X_test": X_test, "y_test": test_df["y_binary"].to_numpy(np.int64),
    }


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    from torch.utils.data import DataLoader, TensorDataset

    def dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32),
        )

    return (
        DataLoader(dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(dataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(dataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
    )


def predict(model, loader):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for xb, _ in loader:
            probabilities.append(torch.sigmoid(model(xb)).cpu().numpy().ravel())
    return np.concatenate(probabilities)


def fit_mlp(data, columns, seed, schedule=BUDGET_SCHEDULE):
    """Train one network on the given column subset and return probabilities."""
    set_seed(seed)

    index = [data["features"].index(c) for c in columns]

    X_train = data["X_train"][:, index]
    X_val = data["X_val"][:, index]
    X_test = data["X_test"][:, index]

    train_loader, val_loader, test_loader = make_loaders(
        X_train, data["y_train"], X_val, data["y_val"], X_test, data["y_test"]
    )

    model = TabularMLP(input_dim=len(columns))

    positive = float((data["y_train"] == 1).sum())
    negative = float((data["y_train"] == 0).sum())

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs, patience = schedule
    best_state, best_auc, wait = None, -np.inf, 0
    started = time.time()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        auc = evaluate_binary(
            data["y_val"], predict(model, val_loader)
        )["roc_auc"]

        if auc > best_auc:
            best_auc, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return {
        "model": model,
        "val_probability": predict(model, val_loader),
        "test_probability": predict(model, test_loader),
        "training_time_sec": time.time() - started,
    }


# ---------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------

def shap_ranking(data, out_dir):
    """Rank features by mean absolute attribution, computed on training rows.

    Background and explanation samples are drawn disjointly from the training
    partition so that the explained points are not also the reference points.
    """
    import shap

    rng = np.random.default_rng(GLOBAL_SEED)
    y_train = data["y_train"]

    background, explain = [], []
    for class_value in (0, 1):
        pool = np.flatnonzero(y_train == class_value)
        rng.shuffle(pool)
        background.append(pool[:128])
        explain.append(pool[128:328])

    background = np.concatenate(background)
    explain = np.concatenate(explain)
    assert len(np.intersect1d(background, explain)) == 0

    # Explain the trained full-feature reference network. It is fitted with
    # the reference schedule so that it matches the network in Table IV.
    fitted = fit_mlp(
        data, data["features"], GLOBAL_SEED, schedule=REFERENCE_SCHEDULE
    )
    model = fitted["model"]
    model.eval()

    wrapper = nn.Sequential(model, nn.Sigmoid())

    bg_tensor = torch.tensor(data["X_train"][background], dtype=torch.float32)
    ex_tensor = torch.tensor(data["X_train"][explain], dtype=torch.float32)

    try:
        explainer = shap.DeepExplainer(wrapper, bg_tensor)
        values = explainer.shap_values(ex_tensor)
        method = "DeepExplainer"
    except Exception as error:
        print(f"  DeepExplainer failed ({error.__class__.__name__}); "
              "using GradientExplainer")
        explainer = shap.GradientExplainer(wrapper, bg_tensor)
        values = explainer.shap_values(ex_tensor)
        method = "GradientExplainer"

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]

    ranking = (
        pd.DataFrame({
            "feature": data["features"],
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["method"] = method
    ranking.to_csv(out_dir / "shap_ranking_training.csv", index=False)

    print(f"  method: {method}")
    print(f"  top 10: {ranking['feature'].head(10).tolist()}")

    return ranking["feature"].tolist()


def select_budget(data, ranked, out_dir):
    """Choose the budget on validation, with the rule in units of seed noise."""
    validation_rows, test_rows = [], []

    for k in BUDGETS:
        if k > len(ranked):
            continue
        columns = ranked[:k]
        print(f"  budget top-{k}")

        for seed in BUDGET_SEEDS:
            fitted = fit_mlp(data, columns, seed)

            val = evaluate_binary(data["y_val"], fitted["val_probability"])
            test = evaluate_binary(data["y_test"], fitted["test_probability"])

            validation_rows.append({"n_features": k, "seed": seed, **val})
            test_rows.append({"n_features": k, "seed": seed, **test})

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)

    summary = (
        validation_df.groupby("n_features")["balanced_accuracy"]
        .agg(["mean", "std"]).reset_index()
    )

    pooled_sd = float(np.sqrt(np.mean(summary["std"].to_numpy() ** 2)))
    best = float(summary["mean"].max())
    inside = summary[summary["mean"] >= best - pooled_sd]
    selected = int(inside["n_features"].min())

    print(f"\n  pooled within-budget SD : {pooled_sd:.5f}")
    print(f"  spread across budgets   : "
          f"{summary['mean'].max() - summary['mean'].min():.5f}")
    print(f"  budgets inside the band : {sorted(inside['n_features'])}")
    print(f"  selected                : top-{selected}")

    validation_df.to_csv(out_dir / "budget_validation.csv", index=False)
    test_df.to_csv(out_dir / "budget_test_posthoc.csv", index=False)
    summary.to_csv(out_dir / "budget_summary.csv", index=False)

    with open(out_dir / "budget_selection.json", "w", encoding="utf-8") as handle:
        json.dump({
            "pooled_sd": pooled_sd,
            "best_mean": best,
            "band_lower": best - pooled_sd,
            "budgets_inside_band": sorted(int(v) for v in inside["n_features"]),
            "selected_budget": selected,
            "selected_features": ranked[:selected],
        }, handle, indent=2)

    return selected, ranked[:selected]


def seed_stability(data, columns, out_dir):
    """Ten-seed stability for the selected profile, with error counts."""
    from sklearn.metrics import confusion_matrix

    rows = []
    for seed in PROFILE_SEEDS:
        fitted = fit_mlp(data, columns, seed)
        probability = fitted["test_probability"]

        for threshold in (0.50, 0.05):
            prediction = (probability >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                data["y_test"], prediction, labels=[0, 1]
            ).ravel()

            metrics = evaluate_binary(data["y_test"], probability, threshold)
            rows.append({
                "seed": seed, "threshold": threshold, **metrics,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "predicted_positive_rate": float(prediction.mean()),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "profile_seed_stability.csv", index=False)

    for threshold in (0.50, 0.05):
        block = frame[frame["threshold"] == threshold]
        print(f"  threshold {threshold:.2f}: "
              f"BA {block['balanced_accuracy'].mean():.6f} "
              f"± {block['balanced_accuracy'].std():.6f}")

    return frame


def leave_one_family_out(train_df, val_df, test_df, out_dir):
    """Retrain per family, refitting preprocessing after the family is removed."""
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    def family_of(frame):
        return frame[LABEL_COLUMN].astype(str)

    families = sorted(
        f for f in family_of(train_df).unique()
        if f.lower() not in ("benign", "other")
    )
    print(f"  families: {families}")

    ignore = {LABEL_COLUMN, "y_binary"}
    columns = [c for c in train_df.columns if c not in ignore]
    columns = [c for c in columns if train_df[c].nunique(dropna=False) > 1]

    rows = []
    for family in families:
        train_mask = family_of(train_df) != family
        val_mask = family_of(val_df) != family

        fold_train = train_df.loc[train_mask]
        fold_val = val_df.loc[val_mask]

        if fold_train["y_binary"].nunique() < 2:
            print(f"  {family}: skipped, one class remains")
            continue

        # refit both preprocessing objects on the reduced training rows
        medians = (
            fold_train[columns].replace([np.inf, -np.inf], np.nan).median()
        )

        def prepare(frame):
            block = frame[columns].replace([np.inf, -np.inf], np.nan)
            return block.fillna(medians).astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(prepare(fold_train)).astype(np.float32)
        X_v = scaler.transform(prepare(fold_val)).astype(np.float32)
        X_te = scaler.transform(prepare(test_df)).astype(np.float32)

        fold_data = {
            "features": columns,
            "X_train": X_tr, "y_train": fold_train["y_binary"].to_numpy(np.int64),
            "X_val": X_v, "y_val": fold_val["y_binary"].to_numpy(np.int64),
            "X_test": X_te, "y_test": test_df["y_binary"].to_numpy(np.int64),
        }

        fitted = fit_mlp(
            fold_data, columns, GLOBAL_SEED + len(family), schedule=LOFO_SCHEDULE
        )
        probability = fitted["test_probability"]

        test_family = family_of(test_df)
        held = np.flatnonzero(test_family == family)
        benign = np.flatnonzero(test_family.str.lower() == "benign")

        for name, index in [
            ("heldout_family_only", held),
            ("benign_plus_heldout", np.concatenate([benign, held])),
        ]:
            if len(index) == 0:
                continue

            y_true = fold_data["y_test"][index]
            p = probability[index]
            prediction = (p >= 0.50).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true, prediction, labels=[0, 1]
            ).ravel()

            rows.append({
                "heldout_family": family,
                "eval_set": name,
                "n_train_after_holdout": int(len(fold_data["y_train"])),
                "n_eval": int(len(index)),
                "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "predicted_positive_rate": float(prediction.mean()),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })

        print(f"  {family}: trained on {len(fold_data['y_train'])} rows")

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "lofo_refitted.csv", index=False)
    return frame


# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/ciciot")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(GLOBAL_SEED)

    print("\n[1/5] Loading CICIoT2023")
    train_df = load_split(args.train, max_rows=args.max_rows)
    val_df = load_split(args.val, max_rows=args.max_rows)
    test_df = load_split(args.test, max_rows=args.max_rows)

    data = build_matrices(train_df, val_df, test_df)
    print(f"  train {data['X_train'].shape} | val {data['X_val'].shape} "
          f"| test {data['X_test'].shape}")
    print(f"  features: {len(data['features'])}")
    print(f"  attack ratio: train {data['y_train'].mean():.4f} "
          f"test {data['y_test'].mean():.4f}")

    print("\n[2/5] Attribution ranking from training rows")
    ranked = shap_ranking(data, out_dir)

    print("\n[3/5] Budget selection on validation")
    selected, columns = select_budget(data, ranked, out_dir)

    print(f"\n[4/5] Seed stability, top-{selected}")
    seed_stability(data, columns, out_dir)

    print("\n[5/5] Leave-one-family-out, preprocessing refitted per fold")
    leave_one_family_out(train_df, val_df, test_df, out_dir)

    print(f"\nDone. Results in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
