#!/usr/bin/env python3
"""Run the complete Edge-IIoTset pipeline.

Stages, in order:

  1. load and deduplicate, build canonical endpoint pairs
  2. row-level reference split
  3. search for ten endpoint-pair-disjoint outer partitions
  4. fit every model on every partition                 -> Table X
  5. placebo-group and prevalence-matched controls      -> Table XI
  6. transport-identity ablation under both protocols   -> Table XII
  7. reproducibility artefacts and manifests            -> Appendix

Usage:
    python scripts/run_edge.py --data path/to/ML-EdgeIIoT-dataset.csv \\
                               --out results/edge

Every stage writes a CSV as it finishes, so an interrupted run can be resumed
by rerunning with the same output directory; completed stages are skipped
unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import family_js_divergence, set_seed  # noqa: E402
from src.edge_data import (  # noqa: E402
    TRANSPORT_IDENTITY_FIELDS,
    load_edge,
    make_placebo_groups,
    prevalence_matched_split,
    search_outer_splits,
)
from src.edge_runner import run_partition  # noqa: E402

ROW_LEVEL_SEED = 42
ROW_LEVEL_TEST_SIZE = 0.15
PLACEBO_SEED_BASE = 7000


def stage_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.csv"


def already_done(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  skipping, {path.name} exists (use --force to redo)")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="ML-EdgeIIoT-dataset.csv")
    parser.add_argument("--out", default="results/edge", help="output directory")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(ROW_LEVEL_SEED)

    # -- 1. data ---------------------------------------------------------
    print("\n[1/7] Loading Edge-IIoTset")
    bundle = load_edge(args.data)

    predictors = bundle["predictors"]
    labels = bundle["labels"]
    families = bundle["families"]
    groups = bundle["endpoint_groups"]

    for key, value in bundle["provenance"].items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")

    with open(out_dir / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(bundle["provenance"], handle, indent=2)

    # -- 2. outer splits -------------------------------------------------
    print(f"\n[2/7] Searching for {args.n_splits} endpoint-pair-disjoint splits")
    records, index_pairs = search_outer_splits(
        labels, groups, n_outer_splits=args.n_splits
    )
    split_df = pd.DataFrame(records)
    split_df.to_csv(out_dir / "outer_split_audit.csv", index=False)
    print(f"  accepted random states: "
          f"{[r['outer_random_state'] for r in records]}")

    # -- 3. row-level reference -----------------------------------------
    print("\n[3/7] Row-level reference")
    path = stage_path(out_dir, "row_level")
    if not already_done(path, args.force):
        from sklearn.model_selection import train_test_split

        all_idx = np.arange(len(labels))
        row_train, temp = train_test_split(
            all_idx,
            test_size=ROW_LEVEL_TEST_SIZE * 2,
            random_state=ROW_LEVEL_SEED,
            stratify=labels,
        )
        row_val, row_test = train_test_split(
            temp, test_size=0.5, random_state=ROW_LEVEL_SEED,
            stratify=labels[temp],
        )

        rows, _ = run_partition(
            predictors, labels, row_train, row_test,
            arm_label="row_level", split_id=0, seed_offset=900,
            val_index=row_val,
            mlp_seeds=[11, 22, 33, 42, 55],
            tree_seed=ROW_LEVEL_SEED,
        )
        pd.DataFrame(rows).to_csv(path, index=False)

    # -- 4. endpoint stress ----------------------------------------------
    print("\n[4/7] Endpoint-pair-disjoint partitions")
    path = stage_path(out_dir, "endpoint_stress")
    if not already_done(path, args.force):
        collected, meta = [], []
        for record, (train_idx, test_idx) in zip(records, index_pairs):
            sid = record["split_id"]
            print(f"  split {sid}/{len(records)}")

            rows, names = run_partition(
                predictors, labels, train_idx, test_idx,
                arm_label="endpoint_disjoint", split_id=sid,
                seed_offset=sid,
            )
            collected.extend(rows)

            meta.append({
                **record,
                "train_endpoint_pairs": int(len(np.unique(groups[train_idx]))),
                "test_endpoint_pairs": int(len(np.unique(groups[test_idx]))),
                "shared_endpoint_pairs": int(
                    len(np.intersect1d(
                        np.unique(groups[train_idx]), np.unique(groups[test_idx])
                    ))
                ),
                "family_js_divergence": family_js_divergence(
                    families.iloc[train_idx], families.iloc[test_idx]
                ),
                "retained_features": len(names),
            })

        pd.DataFrame(collected).to_csv(path, index=False)
        pd.DataFrame(meta).to_csv(out_dir / "split_metadata.csv", index=False)

    # -- 5. matched controls ---------------------------------------------
    print("\n[5/7] Matched controls")
    path = stage_path(out_dir, "matched_controls")
    if not already_done(path, args.force):
        from sklearn.model_selection import GroupShuffleSplit

        collected = []
        all_idx = np.arange(len(labels))

        for record in records:
            sid = record["split_id"]
            print(f"  P1 placebo, split {sid}")

            placebo = make_placebo_groups(groups, PLACEBO_SEED_BASE + sid)
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=0.15,
                random_state=record["outer_random_state"],
            )
            tr_pos, te_pos = next(
                splitter.split(all_idx, labels, groups=placebo)
            )
            rows, _ = run_partition(
                predictors, labels, all_idx[tr_pos], all_idx[te_pos],
                arm_label="P1_placebo_groups", split_id=sid,
                seed_offset=300 + sid,
            )
            collected.extend(rows)

        for record in records:
            sid = record["split_id"]
            print(f"  P2 prevalence-matched, split {sid}")

            tr, te, achieved, clipped = prevalence_matched_split(
                labels,
                n_test=record["test_rows"],
                target_ratio=record["test_attack_ratio"],
                seed=PLACEBO_SEED_BASE + 500 + sid,
            )
            if clipped:
                print(f"    target prevalence clipped "
                      f"(requested {record['test_attack_ratio']:.4f}, "
                      f"achieved {achieved:.4f})")

            rows, _ = run_partition(
                predictors, labels, tr, te,
                arm_label="P2_prevalence_matched", split_id=sid,
                seed_offset=600 + sid,
            )
            collected.extend(rows)

        pd.DataFrame(collected).to_csv(path, index=False)

    # -- 6. ablation ------------------------------------------------------
    print("\n[6/7] Transport-identity ablation")
    path = stage_path(out_dir, "transport_ablation")
    if not already_done(path, args.force):
        blocklist = [
            c for c in TRANSPORT_IDENTITY_FIELDS if c in predictors.columns
        ]
        print(f"  withdrawing {len(blocklist)} fields: {blocklist}")

        collected = []
        from sklearn.model_selection import train_test_split

        all_idx = np.arange(len(labels))
        row_train, temp = train_test_split(
            all_idx, test_size=ROW_LEVEL_TEST_SIZE * 2,
            random_state=ROW_LEVEL_SEED, stratify=labels,
        )
        row_val, row_test = train_test_split(
            temp, test_size=0.5, random_state=ROW_LEVEL_SEED,
            stratify=labels[temp],
        )

        for arm, blocked in [("FULL", []), ("NO_ID", blocklist)]:
            rows, _ = run_partition(
                predictors, labels, row_train, row_test,
                arm_label=f"row_level/{arm}", split_id=0, seed_offset=900,
                feature_blocklist=blocked, val_index=row_val,
                mlp_seeds=[11, 22, 33, 42, 55], tree_seed=ROW_LEVEL_SEED,
            )
            for r in rows:
                r["protocol"], r["feature_arm"] = "Row-level", arm
            collected.extend(rows)

        for record, (train_idx, test_idx) in zip(records, index_pairs):
            sid = record["split_id"]
            print(f"  endpoint NO_ID, split {sid}")
            rows, _ = run_partition(
                predictors, labels, train_idx, test_idx,
                arm_label="endpoint_disjoint/NO_ID", split_id=sid,
                seed_offset=1000 + sid, feature_blocklist=blocklist,
            )
            for r in rows:
                r["protocol"] = "Endpoint-pair-disjoint"
                r["feature_arm"] = "NO_ID"
            collected.extend(rows)

        pd.DataFrame(collected).to_csv(path, index=False)

    # -- 7. manifests -----------------------------------------------------
    print("\n[7/7] Manifests and group statistics")
    sizes = pd.Series(groups).value_counts()
    purity = (
        pd.DataFrame({"group": groups, "label": labels})
        .groupby("group")["label"].agg(["size", "mean"])
    )

    stats = {
        "n_groups": int(sizes.size),
        "singleton_groups": int((sizes == 1).sum()),
        "median_group_size": float(sizes.median()),
        "largest_group_records": int(sizes.max()),
        "largest_group_share_pct": float(100 * sizes.max() / len(groups)),
        "top10_share_pct": float(100 * sizes.head(10).sum() / len(groups)),
        "attack_only_groups": int((purity["mean"] == 1.0).sum()),
        "benign_only_groups": int((purity["mean"] == 0.0).sum()),
        "mixed_groups": int(
            len(purity)
            - (purity["mean"] == 1.0).sum()
            - (purity["mean"] == 0.0).sum()
        ),
    }
    for key, value in stats.items():
        print(f"  {key}: {value}")

    with open(out_dir / "group_statistics.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    manifest = pd.concat([
        pd.DataFrame({
            "split_id": record["split_id"],
            "endpoint_group": np.unique(groups[test_idx]),
            "assignment": "outer_test",
        })
        for record, (_, test_idx) in zip(records, index_pairs)
    ], ignore_index=True)

    manifest.to_csv(
        out_dir / "outer_split_manifest.csv.gz", index=False, compression="gzip"
    )
    print(f"  manifest rows: {len(manifest)} (test side; training side is the "
          "complement)")

    print(f"\nDone. Results in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
