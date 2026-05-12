from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Sequence


def _augmentation_name(record: dict) -> str:
    augmentation = record.get("augmentation")
    if isinstance(augmentation, dict):
        return str(augmentation.get("name") or augmentation.get("type") or "augmented")
    if augmentation:
        return str(augmentation)
    return "raw"


def _is_non_positive(record: dict) -> bool:
    return record.get("label") != "positive"


def summarize_false_accepts_by_text(records: Sequence[dict], sample_scores: Sequence[dict], *, threshold: float) -> list[dict]:
    """Summarize real microWakeWord false accepts by original text.

    `sample_scores` must come from real `.tflite` streaming inference. Positive
    samples are intentionally excluded; every non-positive label, including
    `holdout`, contributes to false-accept analysis.
    """

    records_by_id = {str(record["sample_id"]): record for record in records}
    buckets: dict[str, dict] = defaultdict(
        lambda: {
            "sample_count": 0,
            "false_accepts": 0,
            "scores": [],
            "labels": Counter(),
            "augmentation_counts": Counter(),
        }
    )

    for sample_score in sample_scores:
        sample_id = str(sample_score["sample_id"])
        record = records_by_id.get(sample_id)
        if record is None or not _is_non_positive(record):
            continue
        text = str(record.get("text") or sample_id)
        score = float(sample_score["score"])
        bucket = buckets[text]
        bucket["sample_count"] += 1
        bucket["scores"].append(score)
        bucket["labels"][str(record.get("label", "unknown"))] += 1
        bucket["augmentation_counts"][_augmentation_name(record)] += 1
        if score >= threshold:
            bucket["false_accepts"] += 1

    rows = []
    for text, bucket in buckets.items():
        scores = bucket["scores"]
        sample_count = int(bucket["sample_count"])
        false_accepts = int(bucket["false_accepts"])
        rows.append(
            {
                "text": text,
                "sample_count": sample_count,
                "false_accepts": false_accepts,
                "false_accept_rate": false_accepts / max(1, sample_count),
                "mean_score": mean(scores) if scores else 0.0,
                "median_score": median(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "labels": dict(bucket["labels"]),
                "augmentation_counts": dict(bucket["augmentation_counts"]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -float(row["false_accept_rate"]),
            -int(row["false_accepts"]),
            -float(row["max_score"]),
            str(row["text"]),
        ),
    )
