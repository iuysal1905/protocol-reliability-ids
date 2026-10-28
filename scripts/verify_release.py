#!/usr/bin/env python3
"""Validate the supplied original manifests and summary tables, no raw datasets needed."""
from pathlib import Path
from collections import Counter,defaultdict
import csv, gzip, json, math

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "reproducibility"
LEGACY = ROOT / "results" / "published_endpoint_stress_v2"


def csv_rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, mode="rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def verify():
    sizes = {}
    for row in csv_rows(REPRO / "endpoint_group_sizes.csv"):
        group = row.get("", row.get("group"))
        sizes[group] = int(row["n_records"])
    assert len(sizes) == 25723 and sum(sizes.values()) == 156986
    assert sum(v==1 for v in sizes.values()) == 25664
    assert max(sizes.values()) == 89976
    composition = list(csv_rows(REPRO / "endpoint_group_class_composition.csv"))
    assert len(composition) == len(sizes)
    assert all(int(r["n_records"]) == sizes[r["group"]] for r in composition)
    assert sum(float(r["attack_ratio"])==1.0 for r in composition) == 25711
    assert sum(float(r["attack_ratio"])==0.0 for r in composition) == 8

    audits = {int(r["split_id"]):r for r in csv_rows(LEGACY / "outer_split_audit.csv")}
    paper = {int(r["split_id"]):r for r in csv_rows(REPRO / "published_table_C_VII_partition_composition.csv")}
    assert len(audits) == len(paper) == 10
    assignments = defaultdict(Counter)
    for r in csv_rows(REPRO / "outer_split_manifest.csv.gz"):
        sid = int(r["split_id"])
        assert r["assignment"] in ("outer_train", "outer_test")
        assignments[sid][r["assignment"]] += 1
    assert len(assignments)==10
    for sid,counts in assignments.items():
        assert counts == Counter({"outer_train":21864, "outer_test":3859})
        a,p = audits[sid],paper[sid]
        assert int(a["outer_random_state"])==int(p["outer_random_state"])
        assert int(a["train_rows"])==int(p["train_records"])
        assert int(a["test_rows"])==int(p["test_records"])
        assert int(a["test_benign"])==int(p["test_benign"])
        assert int(a["test_attack"])==int(p["test_attack"])
        assert int(p["shared_endpoint_pairs"])==0

    row_assignments = defaultdict(Counter)
    for r in csv_rows(REPRO / "outer_split_row_manifest.csv.gz"):
        row_assignments[int(r["split_id"])][r["assignment"]] += 1
    for sid,c in row_assignments.items():
        assert c["outer_train"]==int(audits[sid]["train_rows"])
        assert c["outer_test"]==int(audits[sid]["test_rows"])

    models = defaultdict(list)
    for r in csv_rows(LEGACY / "repeated_endpoint_stress_detailed.csv"):
        models[r["model"]].append(r)
        assert int(r["tn"])+int(r["fp"])+int(r["fn"])+int(r["tp"]) == int(audits[int(r["split_id"])]["test_rows"])
    assert len(models)==5 and all(len(rs)==10 for rs in models.values())
    expected = {"Tabular MLP":.5357,"Random Forest":.4945,
                "XGBoost":.4942,"LightGBM":.4622}
    for model, exp in expected.items():
        mean = sum(float(r["balanced_accuracy"]) for r in models[model])/10
        assert abs(mean-exp)<.00005, (model,mean,exp)
    assert len(list(csv_rows(REPRO / "outer_split_manifest.csv")))==257230
    assert sum(1 for _ in csv_rows(REPRO / "outer_split_row_manifest.csv.gz")) == 1569860
    print("PASS: 25,723 pairs; 156,986 records; 10 disjoint partitions; ")
    print("      both full manifests; published Table C.VII; 50 model results; Table X BA values.")
    print("NOTE: Original source datasets are not included; model retraining was not tested.")

if __name__ == "__main__":
    verify()
