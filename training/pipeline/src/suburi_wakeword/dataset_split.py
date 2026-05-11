from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable


def _score(record: dict, seed: str) -> int:
    key = f"{seed}\0{record.get('label')}\0{record.get('text')}\0{record.get('sample_id')}"
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)


def assign_splits(
    records: Iterable[dict],
    *,
    holdout_labels: set[str] | None = None,
    validation_fraction: float = 0.25,
    seed: str = "hai_stackchan_ja",
) -> list[dict]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    holdout_labels = holdout_labels or {"holdout"}
    records_list = [dict(record) for record in records]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records_list:
        if record.get("label") in holdout_labels or record.get("split") == "holdout":
            record["split"] = "holdout"
        else:
            grouped[record["label"]].append(record)

    for label, group in grouped.items():
        ordered = sorted(group, key=lambda record: _score(record, seed))
        validation_count = 1 if len(ordered) > 1 else 0
        validation_count = max(validation_count, int(round(len(ordered) * validation_fraction))) if len(ordered) > 1 else 0
        validation_ids = {id(record) for record in ordered[:validation_count]}
        for record in group:
            record["split"] = "validation" if id(record) in validation_ids else "train"
    return records_list
