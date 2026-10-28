# Protocol-dependent reliability of IoT intrusion detection

Code accompanying the manuscript on evaluation-protocol dependence in IoT
intrusion detection, covering CICIoT2023 and Edge-IIoTset.

The study does not propose a detection model. It measures how much of the
performance reported under conventional row-level evaluation survives when the
test boundary is drawn differently, and examines plausible contributors using two
matched control arms and a feature ablation.

## What this repository reproduces

| Manuscript item | Produced by |
|---|---|
| Table V, feature budgets | `scripts/run_ciciot.py`, stage 3 |
| Table VI, seed stability | `scripts/run_ciciot.py`, stage 4 |
| Table IX, leave-one-family-out | `scripts/run_ciciot.py`, stage 5 |
| Table X, cross-protocol comparison | `scripts/run_edge.py`, stages 3 and 4 |
| Table XI, matched controls | `scripts/run_edge.py`, stage 5 |
| Table XII, transport-identity ablation | `scripts/run_edge.py`, stage 6 |
| Section III-F group statistics | `scripts/run_edge.py`, stage 7 |
| Split manifests | `scripts/run_edge.py`, stage 7 |
| Effect sizes and loss decomposition | `scripts/analyze_edge.py` |

## Data

Neither dataset is redistributed here.

**CICIoT2023** — obtain from the Canadian Institute for Cybersecurity. The
scripts expect three CSV files already split into training, validation and
test partitions, each with a `label` column whose value is `BenignTraffic` for
benign records and the attack name otherwise.

**Edge-IIoTset** — obtain the `ML-EdgeIIoT-dataset.csv` release. The script
reads it directly and performs its own deduplication.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The reported results were produced with Python 3.13.9 and the principal library versions listed in `requirements.txt` (SciPy is not version-pinned). Everything runs on CPU; no GPU is required. A full Edge
run takes a few hours on a laptop, dominated by the ten endpoint partitions
and the two control arms.

## Running

```bash
python scripts/run_ciciot.py \
    --train data/ciciot_train.csv \
    --val   data/ciciot_val.csv \
    --test  data/ciciot_test.csv \
    --out   results/ciciot

python scripts/run_edge.py \
    --data data/ML-EdgeIIoT-dataset.csv \
    --out  results/edge

python scripts/analyze_edge.py --results results/edge
```

Each Edge stage writes its CSV as it finishes and is skipped on a rerun unless
`--force` is passed, so an interrupted run can be resumed.

## How the two benchmarks are separated

The benchmarks live in separate scripts and share only `src/common.py`. This is
deliberate. The original analysis was carried out in one notebook where both
pipelines used the same names: later cells rebound `y_train`, `y_val`, `y_test`
and `train_one_epoch`, so a cell written against the CICIoT2023 arrays could
silently receive Edge-IIoTset labels of a different length. Importing shared
code from a module and keeping the two pipelines in separate processes removes
that class of error by construction.

## Method notes that matter for reproduction

**Endpoint pairs.** A group is the canonical undirected pair formed from
`ip.src_host` and `ip.dst_host`: the two identifiers are sorted
lexicographically and joined, so A-to-B and B-to-A are one group. Missing
identifiers are replaced by the tokens `missing_source` and
`missing_destination` before the pair is formed; no row is dropped for a
missing identifier.

These are address strings recorded in the capture. The dataset carries no
device inventory, so a group is a communication relationship between two
addresses, not a verified physical device.

**Outer-split search.** Random states are tried in ascending order from zero
and the first ten that put at least 500 benign and 500 attack records in both
the outer training and the outer test partition, with a test-group set not
already accepted, are kept. The search is deterministic, so the accepted
states are a property of the dataset and the thresholds. On the released file
they are 3, 6, 10, 16, 18, 19, 21, 22, 25 and 27.

**Preprocessing.** Numeric coercion, replacement of infinities, removal of
columns more than 95% missing, median imputation, removal of zero-variance
columns, then standardization. Every step is fitted on inner-training rows only
and applied to the validation and test rows, separately within each partition.

**Attribution ranking.** Computed on training rows, with background and
explanation samples drawn disjointly. Feature budgets are compared on the
validation partition across five seeds; the selected budget is the smallest
whose mean lies within one pooled within-budget standard deviation of the best
mean. The test partition is excluded from ranking and budget selection. After the
budget is chosen on validation rows, the test metrics of all budgets are
written to `budget_test_posthoc.csv` as a post-selection diagnostic.

**Training schedules.** The explained full-feature reference network uses at
most 20 epochs with patience 5, as in the baseline comparison. The feature
budgets and the ten-seed runs of the selected profile use at most 12 epochs
with patience 4. Leave-one-family-out folds use at most 10 epochs with
patience 3. Early stopping selects the checkpoint by validation ROC-AUC in all
three.

**Leave-one-family-out.** The median fill and the standardization are refitted
after the held-out family is removed, so no statistic from that family enters
the fitted preprocessing objects.

**Class weighting.** Applied to every model. Because attack records are the
majority class in Edge-IIoTset, the negative-to-positive ratio used as the
positive-class weight is below one and therefore reduces the attack
contribution rather than boosting it.

## Known limitations of this release

The scripts were derived from the notebook used for the reported analysis
rather than the reverse. They reproduce the same procedure and, on the released
datasets, the same numbers, but a rerun on a different platform may differ in
the last decimal because of library-level floating-point behaviour. Reported
standard deviations across seeds give the scale within which such differences
are expected.

Latency and model-size figures are specific to the machine that produced them
and are not hardware-independent estimates.

## Layout

```
src/common.py        metrics, the tabular network, seeding
src/edge_data.py     loading, endpoint pairs, split search, control arms
src/edge_runner.py   one function that fits and scores one partition
scripts/run_ciciot.py
scripts/run_edge.py
scripts/analyze_edge.py
scripts/verify_release.py
results/published_endpoint_stress_v2/   original saved stress outputs
results/reproducibility/                group/row manifests and configuration
```

## Zenodo release contents and provenance

The ZIP contains **actual stored results** from the original endpoint-pair-
disjoint stress experiment and the later reproducibility notebook, in addition
to scripts. See `results/README.md` for precise provenance, distinctions
between legacy result metadata and the final Table C.VII, and what was **not**
provided in the source uploads. The raw datasets are not redistributed.

The full pair-group manifest (`outer_split_manifest.csv`) contains both train
and test assignments for all ten partitions; the row manifest contains the
post-deduplication source-row indices. Neither is a standalone training dataset.

For a quick check of the published files without datasets or training, run:

```bash
python scripts/verify_release.py
python tests/test_smoke.py
```

A full model rerun still requires the original datasets. Historical endpoint
stress results were not substituted for independent matched-control or ablation
outputs. Those arms must be rerun from source data if their detailed CSVs are
needed.

## Changelog

**v1.1.0 (Zenodo revised archive, 2026-10-28)** — included published
endpoint-group and row manifests, source-produced group-composition tables,
original stress metrics, and the final Table C.VII partition composition;
restored verified Edge MLP hyperparameters (20 epochs, patience 5), fixed
missing pandas import, corrected CICIoT2023 `BenignTraffic` encoding and
seven-family grouping, and kept test budget diagnostics post-selection. The
transport-identity list is now the eight fields in the manuscript. Existing
experimental results have not been regenerated or overwritten.

**v1.0.1** — `run_ciciot.py` now explains the trained reference network; v1.0.0
explained a freshly initialized network in the attribution stage. Training
schedules for the reference network and the leave-one-family-out folds now
match the reported analysis. The outer-split search now also requires 500
benign and 500 attack records on the training side and skips repeated
test-group sets, as the reported analysis did. `requirements.txt` now pins the
versions listed in the manuscript. Results in the manuscript were produced by the
original notebook and are unaffected.

## License

MIT, see `LICENSE`. The datasets carry their own terms; consult the original
providers.

## Archived release

Zenodo v1.1.0:
https://doi.org/10.5281/zenodo.23741237