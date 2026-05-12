from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .audio import load_wav_mono
from .features import extract_features


def _read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _record_audio_path(record: dict, manifest_path: Path) -> Path:
    raw = record.get("normalized_audio_path") or record.get("audio_path") or record.get("raw_audio_path")
    if not raw:
        raise ValueError(f"Record is missing an audio path: {record.get('sample_id', '<unknown>')}")
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def _label_value(record: dict) -> int:
    return 1 if record.get("label") == "positive" else 0


def _load_feature_matrix(manifest_path: Path, records: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for record in records:
        audio, sample_rate = load_wav_mono(_record_audio_path(record, manifest_path))
        features.append(extract_features(audio, sample_rate))
        labels.append(_label_value(record))
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.int8)


def _stratified_folds(labels: np.ndarray, folds: int) -> list[np.ndarray]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    if len(positive_indices) < folds or len(negative_indices) < folds:
        raise ValueError(
            f"Not enough samples for {folds}-fold validation: "
            f"positive={len(positive_indices)}, negative={len(negative_indices)}"
        )
    fold_indices: list[list[int]] = [[] for _ in range(folds)]
    for source in (positive_indices, negative_indices):
        for offset, index in enumerate(source):
            fold_indices[offset % folds].append(int(index))
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in fold_indices]


def _predict_nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    normalized_train = (train_x - mean) / std
    normalized_test = (test_x - mean) / std
    positive_centroid = normalized_train[train_y == 1].mean(axis=0)
    negative_centroid = normalized_train[train_y == 0].mean(axis=0)
    positive_distance = np.linalg.norm(normalized_test - positive_centroid, axis=1)
    negative_distance = np.linalg.norm(normalized_test - negative_centroid, axis=1)
    return (positive_distance <= negative_distance).astype(np.int8)


def _metric_row(*, labels: np.ndarray, predictions: np.ndarray, fold: int | None = None) -> dict:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    false_rejects = int(np.sum((labels == 1) & (predictions == 0)))
    false_accepts = int(np.sum((labels == 0) & (predictions == 1)))
    true_positives = int(np.sum((labels == 1) & (predictions == 1)))
    true_negatives = int(np.sum((labels == 0) & (predictions == 0)))
    row = {
        "sample_count": int(len(labels)),
        "positive_samples": positives,
        "negative_samples": negatives,
        "accuracy": (true_positives + true_negatives) / max(1, int(len(labels))),
        "precision": true_positives / max(1, true_positives + false_accepts),
        "recall": true_positives / max(1, positives),
        "frr": false_rejects / max(1, positives),
        "far_per_sample": false_accepts / max(1, negatives),
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
    }
    if fold is not None:
        row["fold"] = fold
    return row


def cross_validate_manifest(manifest_path: Path, *, folds: int = 3) -> dict:
    """Run a lightweight stratified k-fold acoustic proxy evaluation.

    This intentionally evaluates the dataset separability with the repository's
    small handcrafted audio features and a nearest-centroid classifier. It is a
    quick smoke metric for comparing data scales, not a production microWakeWord
    FAR/hour estimate.
    """
    records = _read_manifest(manifest_path)
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    features, labels = _load_feature_matrix(manifest_path, records)
    folds_indices = _stratified_folds(labels, folds)
    all_predictions = np.zeros_like(labels)
    fold_metrics = []
    all_indices = np.arange(len(labels))
    for fold_number, test_indices in enumerate(folds_indices, start=1):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
        predictions = _predict_nearest_centroid(features[train_indices], labels[train_indices], features[test_indices])
        all_predictions[test_indices] = predictions
        fold_metrics.append(_metric_row(labels=labels[test_indices], predictions=predictions, fold=fold_number))
    aggregate = _metric_row(labels=labels, predictions=all_predictions)
    accuracies = [row["accuracy"] for row in fold_metrics]
    return {
        "evaluation": "nearest_centroid_audio_feature_kfold",
        "note": "Lightweight offline proxy; use streaming microWakeWord evaluation for production FAR/hour.",
        "manifest": str(manifest_path),
        "folds": folds,
        "sample_count": aggregate["sample_count"],
        "positive_samples": aggregate["positive_samples"],
        "negative_samples": aggregate["negative_samples"],
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision": aggregate["precision"],
        "recall": aggregate["recall"],
        "frr": aggregate["frr"],
        "far_per_sample": aggregate["far_per_sample"],
        "false_rejects_total": aggregate["false_rejects"],
        "false_accepts_total": aggregate["false_accepts"],
        "fold_metrics": fold_metrics,
    }


def compare_training_scales(manifests: Iterable[tuple[str, Path]], *, folds: int = 3) -> list[dict]:
    rows = []
    for scale, manifest_path in manifests:
        report = cross_validate_manifest(manifest_path, folds=folds)
        rows.append({
            "scale": scale,
            "manifest": report["manifest"],
            "sample_count": report["sample_count"],
            "positive_samples": report["positive_samples"],
            "negative_samples": report["negative_samples"],
            "accuracy_mean": report["accuracy_mean"],
            "accuracy_std": report["accuracy_std"],
            "precision": report["precision"],
            "recall": report["recall"],
            "frr": report["frr"],
            "far_per_sample": report["far_per_sample"],
            "false_rejects_total": report["false_rejects_total"],
            "false_accepts_total": report["false_accepts_total"],
        })
    return rows
