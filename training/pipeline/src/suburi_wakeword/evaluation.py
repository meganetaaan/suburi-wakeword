from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .audio import load_wav_mono
from .threshold_sweep import sweep_thresholds


ModelFactory = Callable[[str], object]


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


def _metric_row(*, labels: np.ndarray, predictions: np.ndarray) -> dict:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    false_rejects = int(np.sum((labels == 1) & (predictions == 0)))
    false_accepts = int(np.sum((labels == 0) & (predictions == 1)))
    true_positives = int(np.sum((labels == 1) & (predictions == 1)))
    true_negatives = int(np.sum((labels == 0) & (predictions == 0)))
    return {
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
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


def _is_placeholder_tflite(path: Path) -> bool:
    return b"suburi-microwakeword-smoke-tflite-placeholder" in path.read_bytes()[:4096]


def _load_upstream_model_factory(source_dir: Path | None = None) -> type:
    """Load upstream microWakeWord's real TFLite inference Model class."""
    if source_dir is not None:
        resolved = str(source_dir.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    module = importlib.import_module("microwakeword.inference")
    return module.Model


def _score_record(model: object, record: dict, manifest_path: Path, *, step_ms: int) -> tuple[float, list[float]]:
    audio, sample_rate = load_wav_mono(_record_audio_path(record, manifest_path))
    if sample_rate != 16000:
        raise ValueError(f"microWakeWord evaluation expects 16 kHz WAV: {record.get('sample_id')} ({sample_rate} Hz)")
    scores = [float(score) for score in model.predict_clip(audio, step_ms=step_ms)]
    return (max(scores) if scores else 0.0), scores


def evaluate_microwakeword_manifest(
    manifest_path: Path,
    tflite_model_path: Path,
    *,
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
    step_ms: int = 10,
    source_dir: Path | None = None,
    model_factory: ModelFactory | None = None,
    include_sample_scores: bool = True,
    splits: Sequence[str] | None = None,
) -> dict:
    """Score a manifest with a real microWakeWord TFLite streaming model.

    This evaluator intentionally does not compute handcrafted features, nearest
    centroids, or any other proxy metric. Each WAV is passed through upstream
    `microwakeword.inference.Model.predict_clip`, and the clip score is the max
    streaming score emitted by that real `.tflite` model.
    """
    manifest_path = Path(manifest_path)
    tflite_model_path = Path(tflite_model_path)
    if not tflite_model_path.exists():
        raise FileNotFoundError(f"microWakeWord TFLite model is missing: {tflite_model_path}")
    if _is_placeholder_tflite(tflite_model_path):
        raise ValueError(f"Refusing to evaluate placeholder TFLite model: {tflite_model_path}")
    records = _read_manifest(manifest_path)
    if splits is not None:
        split_set = set(splits)
        records = [record for record in records if record.get("split") in split_set]
    if not records:
        raise ValueError(f"No records found in {manifest_path} for splits={splits}")

    factory = model_factory or _load_upstream_model_factory(source_dir)
    model = factory(str(tflite_model_path))

    labels: list[int] = []
    predictions: list[int] = []
    score_rows: list[dict] = []
    sample_rows: list[dict] = []
    for record in records:
        score, streaming_scores = _score_record(model, record, manifest_path, step_ms=step_ms)
        prediction = 1 if score >= threshold else 0
        label = _label_value(record)
        labels.append(label)
        predictions.append(prediction)
        score_row = {
            "sample_id": record.get("sample_id"),
            "label": record.get("label"),
            "split": record.get("split"),
            "score": score,
        }
        score_rows.append(score_row)
        if include_sample_scores:
            sample_rows.append({
                **score_row,
                "expected_positive": bool(label),
                "prediction": "positive" if prediction else "negative",
                "streaming_score_count": len(streaming_scores),
            })

    label_array = np.asarray(labels, dtype=np.int8)
    prediction_array = np.asarray(predictions, dtype=np.int8)
    metrics = _metric_row(labels=label_array, predictions=prediction_array)
    report = {
        "evaluation": "real_microwakeword_tflite_streaming",
        "manifest": str(manifest_path),
        "model": str(tflite_model_path),
        "threshold": float(threshold),
        "evaluated_splits": list(splits) if splits is not None else None,
        "step_ms": int(step_ms),
        "sample_count": metrics["sample_count"],
        "positive_samples": metrics["positive_samples"],
        "negative_samples": metrics["negative_samples"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "frr": metrics["frr"],
        "far_per_sample": metrics["far_per_sample"],
        "false_rejects_total": metrics["false_rejects"],
        "false_accepts_total": metrics["false_accepts"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
    }
    if thresholds is not None:
        report["threshold_sweep"] = sweep_thresholds(score_rows, thresholds=thresholds)
    if include_sample_scores:
        report["sample_scores"] = sample_rows
    return report


def compare_microwakeword_manifests(
    manifests: Iterable[tuple[str, Path]],
    tflite_model_path: Path,
    *,
    threshold: float = 0.5,
    thresholds: Sequence[float] | None = None,
    step_ms: int = 10,
    source_dir: Path | None = None,
    model_factory: ModelFactory | None = None,
    splits: Sequence[str] | None = None,
) -> list[dict]:
    rows = []
    for scale, manifest_path in manifests:
        report = evaluate_microwakeword_manifest(
            manifest_path,
            tflite_model_path,
            threshold=threshold,
            thresholds=thresholds,
            step_ms=step_ms,
            source_dir=source_dir,
            model_factory=model_factory,
            include_sample_scores=False,
            splits=splits,
        )
        rows.append({
            "scale": scale,
            "evaluation": report["evaluation"],
            "manifest": report["manifest"],
            "model": report["model"],
            "threshold": report["threshold"],
            "sample_count": report["sample_count"],
            "positive_samples": report["positive_samples"],
            "negative_samples": report["negative_samples"],
            "accuracy": report["accuracy"],
            "precision": report["precision"],
            "recall": report["recall"],
            "frr": report["frr"],
            "far_per_sample": report["far_per_sample"],
            "false_rejects_total": report["false_rejects_total"],
            "false_accepts_total": report["false_accepts_total"],
            "true_positives": report["true_positives"],
            "true_negatives": report["true_negatives"],
        })
    return rows
