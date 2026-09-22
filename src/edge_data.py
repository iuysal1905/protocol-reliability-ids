"""Edge-IIoTset loading, endpoint-pair construction and outer-split search.

The canonical undirected endpoint pair is the unit of separation used by the
strict protocol. A pair is formed by sorting the two host identifiers
lexicographically and joining them, so that traffic from A to B and traffic
from B to A belong to one group.

Nothing in this module fits a model. The outer-split search is deterministic:
random states are tried in ascending order from zero and the first
``n_outer_splits`` that satisfy the minimum class counts in the outer test
partition are kept. Running this module twice on the same file yields the same
partitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .common import binary_counts

SOURCE_FIELD = "ip.src_host"
DESTINATION_FIELD = "ip.dst_host"

#: Columns withheld from the predictor matrix. Identifiers, timestamps,
#: endpoint addresses and raw payload or application-content fields are
#: excluded because they name the conversation rather than describe it.
DROP_COLUMNS = [
    "Attack_label",
    "Attack_type",
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.uri.query",
    "http.referer",
    "http.request.full_uri",
    "tcp.payload",
    "dns.qry.name",
    "mqtt.msg_decoded_as",
    "mqtt.msg",
    "mqtt.protoname",
    "mqtt.topic",
]

#: Transport addressing and session-state fields withdrawn in the NO_ID arm of
#: the ablation. The selection is semantic: these variables record which
#: conversation a packet belongs to rather than how the traffic behaves.
TRANSPORT_IDENTITY_FIELDS = [
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "udp.stream",
    "tcp.seq",
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.stream",
]


def load_edge(data_path: str | Path) -> dict:
    """Read the dataset, deduplicate, and build labels and endpoint groups."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path, low_memory=False)

    original_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())

    frame = frame.drop_duplicates().reset_index(drop=True)

    frame["Attack_label"] = pd.to_numeric(frame["Attack_label"], errors="coerce")
    frame = frame.dropna(subset=["Attack_label"]).copy()
    frame["Attack_label"] = frame["Attack_label"].astype(np.int8)
    frame = frame[frame["Attack_label"].isin([0, 1])].reset_index(drop=True)

    labels = frame["Attack_label"].to_numpy(dtype=np.int64)
    families = frame["Attack_type"].fillna("Unknown").astype(str)

    source = frame[SOURCE_FIELD].fillna("missing_source").astype(str).to_numpy()
    destination = (
        frame[DESTINATION_FIELD].fillna("missing_destination").astype(str).to_numpy()
    )

    endpoint_groups = np.where(
        source <= destination,
        source + " <-> " + destination,
        destination + " <-> " + source,
    )

    present_drops = [c for c in DROP_COLUMNS if c in frame.columns]
    predictors = frame.drop(columns=present_drops).copy()

    for column in predictors.columns:
        predictors[column] = pd.to_numeric(predictors[column], errors="coerce")

    predictors = predictors.replace([np.inf, -np.inf], np.nan)

    empty_columns = [
        c for c in predictors.columns if predictors[c].notna().sum() == 0
    ]
    predictors = predictors.drop(columns=empty_columns)

    return {
        "frame": frame,
        "predictors": predictors,
        "labels": labels,
        "families": families,
        "endpoint_groups": endpoint_groups,
        "provenance": {
            "rows_before_deduplication": original_rows,
            "exact_duplicates_removed": duplicate_rows,
            "rows_after_deduplication": len(frame),
            "benign_records": int((labels == 0).sum()),
            "attack_records": int((labels == 1).sum()),
            "n_endpoint_groups": int(pd.Series(endpoint_groups).nunique()),
            "candidate_features": int(predictors.shape[1]),
            "columns_dropped_as_requested": present_drops,
            "columns_dropped_as_empty": empty_columns,
        },
    }


def search_outer_splits(
    labels: np.ndarray,
    endpoint_groups: np.ndarray,
    n_outer_splits: int = 10,
    outer_test_size: float = 0.15,
    min_test_benign: int = 500,
    min_test_attack: int = 500,
    max_states: int = 5000,
) -> tuple[list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    """Find endpoint-pair-disjoint outer partitions with usable class counts.

    A random state is accepted when both the outer training and the outer test
    partition hold at least the minimum number of benign and attack records,
    and when its test-group set differs from every set already accepted.

    Returns one record and one index pair per accepted partition. The search is
    deterministic, so the accepted random states are a property of the dataset
    and the thresholds rather than of the run.
    """
    indices = np.arange(len(labels))

    records: list[dict] = []
    index_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    seen_test_sets: set = set()

    for state in range(max_states):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=outer_test_size, random_state=state
        )
        train_pos, test_pos = next(
            splitter.split(indices, labels, groups=endpoint_groups)
        )

        train_index = indices[train_pos]
        test_index = indices[test_pos]

        train_benign, train_attack = binary_counts(labels[train_index])
        test_benign, test_attack = binary_counts(labels[test_index])

        if test_benign < min_test_benign or test_attack < min_test_attack:
            continue
        if train_benign < min_test_benign or train_attack < min_test_attack:
            continue

        # A random state that reproduces an already accepted test-group set
        # adds no new partition and is skipped, as in the reported analysis.
        test_key = tuple(sorted(pd.unique(endpoint_groups[test_index])))
        if test_key in seen_test_sets:
            continue
        seen_test_sets.add(test_key)

        records.append({
            "split_id": len(records) + 1,
            "outer_random_state": state,
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "train_benign": int(train_benign),
            "train_attack": int(train_attack),
            "test_benign": int(test_benign),
            "test_attack": int(test_attack),
            "train_attack_ratio": float(labels[train_index].mean()),
            "test_attack_ratio": float(labels[test_index].mean()),
        })
        index_pairs.append((train_index.copy(), test_index.copy()))

        if len(records) >= n_outer_splits:
            break

    if len(records) < n_outer_splits:
        raise RuntimeError(
            f"Only {len(records)} feasible outer splits were found within "
            f"{max_states} random states; {n_outer_splits} were required."
        )

    return records, index_pairs


def make_placebo_groups(real_groups: np.ndarray, seed: int) -> np.ndarray:
    """Randomize group membership while preserving the group-size multiset.

    The splitter then faces an identically skewed grouping problem in which the
    groups no longer correspond to communication relationships, which is what
    separates the partitioning mechanism from endpoint structure.
    """
    rng = np.random.default_rng(seed)

    sizes = pd.Series(real_groups).value_counts().to_numpy(dtype=np.int64)
    n_rows = len(real_groups)
    permutation = rng.permutation(n_rows)

    placebo = np.empty(n_rows, dtype=np.int64)
    cursor = 0
    for group_id, size in enumerate(sizes):
        placebo[permutation[cursor:cursor + size]] = group_id
        cursor += size

    assert cursor == n_rows
    return placebo


def prevalence_matched_split(
    labels: np.ndarray,
    n_test: int,
    target_ratio: float,
    seed: int,
    min_class_count: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Draw a row-level test partition with a prescribed size and prevalence.

    No grouping is applied. When the dataset cannot supply the requested class
    counts the shortfall is reported rather than silently absorbed.
    """
    rng = np.random.default_rng(seed)

    attack_pool = np.flatnonzero(labels == 1)
    benign_pool = np.flatnonzero(labels == 0)

    requested_attack = int(round(n_test * target_ratio))
    requested_benign = n_test - requested_attack

    take_attack = max(
        min(requested_attack, len(attack_pool) - min_class_count), min_class_count
    )
    take_benign = max(
        min(requested_benign, len(benign_pool) - min_class_count), min_class_count
    )

    test_index = np.concatenate([
        rng.choice(attack_pool, size=take_attack, replace=False),
        rng.choice(benign_pool, size=take_benign, replace=False),
    ])

    mask = np.ones(len(labels), dtype=bool)
    mask[test_index] = False

    clipped = (
        requested_attack != take_attack or requested_benign != take_benign
    )

    return (
        np.sort(np.flatnonzero(mask)),
        np.sort(test_index),
        float(labels[test_index].mean()),
        clipped,
    )
