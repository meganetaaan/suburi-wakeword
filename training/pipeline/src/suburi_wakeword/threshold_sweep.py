from __future__ import annotations

from typing import Iterable, Sequence


def sweep_thresholds(records: Iterable[dict], *, thresholds: Sequence[float]) -> list[dict]:
    rows = []
    materialized = list(records)
    positives = [record for record in materialized if record["label"] == "positive"]
    negatives = [record for record in materialized if record["label"] != "positive"]
    for threshold in thresholds:
        false_rejects = sum(1 for record in positives if float(record["score"]) < threshold)
        false_accepts = sum(1 for record in negatives if float(record["score"]) >= threshold)
        rows.append({
            "threshold": float(threshold),
            "positive_samples": len(positives),
            "negative_samples": len(negatives),
            "false_rejects": false_rejects,
            "false_accepts": false_accepts,
            "frr": false_rejects / max(1, len(positives)),
            "far_per_sample": false_accepts / max(1, len(negatives)),
            "note": "Offline sample-level sweep; convert to FAR/hour only with timestamped continuous-audio evaluation.",
        })
    return rows


def default_thresholds(start: float = 0.05, stop: float = 0.95, step: float = 0.05) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values
