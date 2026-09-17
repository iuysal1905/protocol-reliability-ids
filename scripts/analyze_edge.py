#!/usr/bin/env python3
"""Post-process the Edge-IIoTset results.

Reads the CSV files written by ``run_edge.py`` and produces:

  * the prevalence regimes among the outer partitions
  * paired effect sizes for the model comparisons
  * the decomposition of the loss across the control and ablation arms

No model is fitted here. Everything is derived from the stored per-partition
results, so this script is cheap to rerun.

Usage:
    python scripts/analyze_edge.py --results results/edge
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CHANCE = 0.50
PREVALENCE_GAP = 0.10


def assign_regimes(values: np.ndarray, gap: float = PREVALENCE_GAP) -> np.ndarray:
    """Group partitions whose test prevalence differs by less than ``gap``."""
    order = np.argsort(values)
    labels = np.empty(len(values), dtype=int)

    current = 0
    labels[order[0]] = current

    for position in range(1, len(order)):
        if values[order[position]] - values[order[position - 1]] > gap:
            current += 1
        labels[order[position]] = current

    return labels


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the Wilcoxon effect size."""
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()

    return float((positive - negative) / ranks.sum())


def retention(value: float, reference: float) -> float:
    """Chance-adjusted retention of the row-level advantage, in percent."""
    return 100.0 * (value - CHANCE) / (reference - CHANCE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/edge")
    args = parser.parse_args()

    results = Path(args.results)

    stress = pd.read_csv(results / "endpoint_stress.csv")
    metadata = pd.read_csv(results / "split_metadata.csv")

    models = [
        m for m in stress["model"].unique()
        if "always" not in str(m).lower()
    ]

    # -- prevalence regimes ----------------------------------------------
    metadata["regime"] = assign_regimes(
        metadata["test_attack_ratio"].to_numpy()
    )

    regimes = (
        metadata.groupby("regime")
        .agg(
            partitions=("split_id", "count"),
            split_ids=("split_id", lambda s: sorted(s.tolist())),
            prevalence_min=("test_attack_ratio", "min"),
            prevalence_max=("test_attack_ratio", "max"),
        )
        .reset_index()
    )

    print("=" * 78)
    print("PREVALENCE REGIMES")
    print("=" * 78)
    print(regimes.to_string(index=False))
    print(
        f"\nThe {len(metadata)} partitions occupy {len(regimes)} regimes. "
        "Inference that treats\nthem as independent replicates overstates the "
        "available evidence; the paired\ncomparisons below are reported as "
        "descriptive for that reason."
    )

    paired = (
        stress.pivot_table(
            index="split_id", columns="model", values="balanced_accuracy"
        )
        .sort_index()
    )

    regime_of = dict(zip(metadata["split_id"], metadata["regime"]))
    paired_regimes = np.array([regime_of[s] for s in paired.index])

    print("\nMean balanced accuracy within each regime:")
    for regime in sorted(np.unique(paired_regimes)):
        block = paired.loc[paired_regimes == regime, models]
        print(f"\n  regime {regime} (n={len(block)})")
        for model, value in block.mean().sort_values(ascending=False).items():
            print(f"    {model:<24} {value:.4f}")

    # -- effect sizes -----------------------------------------------------
    rows = []
    for model_a, model_b in itertools.combinations(models, 2):
        differences = (paired[model_a] - paired[model_b]).to_numpy()
        if np.allclose(differences, 0):
            continue

        try:
            _, p_value = wilcoxon(differences)
        except ValueError:
            p_value = np.nan

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "mean_difference": float(differences.mean()),
            "rank_biserial": rank_biserial(differences),
            "wilcoxon_p": p_value,
        })

    effects = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("PAIRED EFFECT SIZES, BALANCED ACCURACY")
    print("=" * 78)
    print(effects.round(4).to_string(index=False))

    # -- decomposition ----------------------------------------------------
    row_level = pd.read_csv(results / "row_level.csv")
    row_reference = (
        row_level.groupby("model")["balanced_accuracy"].mean().to_dict()
    )

    controls_path = results / "matched_controls.csv"
    ablation_path = results / "transport_ablation.csv"

    controls = pd.read_csv(controls_path) if controls_path.exists() else None
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else None

    rows = []
    for model in models:
        if model not in row_reference:
            continue

        reference = row_reference[model]
        real = float(paired[model].mean())

        row = {
            "model": model,
            "row_level_BA": round(reference, 4),
            "endpoint_BA": round(real, 4),
            "endpoint_retention_pct": round(retention(real, reference), 1),
        }

        if controls is not None:
            for prefix, column in [
                ("P1", "placebo_retention_pct"),
                ("P2", "prevalence_retention_pct"),
            ]:
                block = controls[
                    (controls["model"] == model)
                    & (controls["arm"].str.startswith(prefix))
                ]
                if len(block):
                    row[column] = round(
                        retention(
                            float(block["balanced_accuracy"].mean()), reference
                        ),
                        1,
                    )

        if ablation is not None:
            noid_row = ablation[
                (ablation["model"] == model)
                & (ablation["protocol"] == "Row-level")
                & (ablation["feature_arm"] == "NO_ID")
            ]["balanced_accuracy"]

            noid_endpoint = ablation[
                (ablation["model"] == model)
                & (ablation["protocol"] == "Endpoint-pair-disjoint")
                & (ablation["feature_arm"] == "NO_ID")
            ]["balanced_accuracy"]

            if len(noid_row) and len(noid_endpoint):
                value = retention(
                    float(noid_endpoint.mean()), float(noid_row.mean())
                )
                row["noid_retention_pct"] = round(value, 1)
                row["recovered_by_ablation_pct"] = round(
                    value - row["endpoint_retention_pct"], 1
                )
                row["residual_unexplained_pct"] = round(100.0 - value, 1)

        rows.append(row)

    decomposition = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("DECOMPOSITION, CHANCE-ADJUSTED RETENTION (%)")
    print("=" * 78)
    print(decomposition.to_string(index=False))
    print(
        "\nRetention near 100 in a control column means the factor that control\n"
        "isolates does not account for the loss. recovered_by_ablation_pct is the\n"
        "retention regained by withdrawing transport-identity predictors, and\n"
        "residual_unexplained_pct is what remains once all three are accounted for."
    )

    out_dir = results / "analysis"
    out_dir.mkdir(exist_ok=True)

    regimes.to_csv(out_dir / "prevalence_regimes.csv", index=False)
    effects.to_csv(out_dir / "paired_effect_sizes.csv", index=False)
    decomposition.to_csv(out_dir / "loss_decomposition.csv", index=False)

    print(f"\nSaved to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
