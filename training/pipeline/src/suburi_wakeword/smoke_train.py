from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio import load_wav_mono
from .features import extract_features


@dataclass(frozen=True)
class CentroidModel:
    labels: tuple[str, ...]
    centroids: np.ndarray


def train_centroid_model(features: np.ndarray, labels: Sequence[str]) -> CentroidModel:
    unique = tuple(sorted(set(labels)))
    centroids = []
    for label in unique:
        rows = features[np.array(labels) == label]
        if len(rows) == 0:
            raise ValueError(f"No rows for label {label}")
        centroids.append(rows.mean(axis=0))
    return CentroidModel(unique, np.vstack(centroids).astype(np.float32))


def predict_label(model: CentroidModel, feature: np.ndarray) -> str:
    distances = np.linalg.norm(model.centroids - feature.astype(np.float32), axis=1)
    return model.labels[int(np.argmin(distances))]


def load_features_from_manifest(manifest_path: Path, base_dir: Path) -> tuple[np.ndarray, list[str], list[dict]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        audio_path = base_dir / record["normalized_audio_path"]
        audio, sample_rate = load_wav_mono(audio_path)
        rows.append(extract_features(audio, sample_rate))
        labels.append(record["label"])
        records.append(record)
    return np.vstack(rows).astype(np.float32), labels, records


def evaluate_self(model: CentroidModel, features: np.ndarray, labels: Sequence[str]) -> dict:
    predictions = [predict_label(model, row) for row in features]
    correct = sum(int(p == y) for p, y in zip(predictions, labels))
    return {
        "samples": len(labels),
        "accuracy_self_eval": correct / max(1, len(labels)),
        "predictions": predictions,
        "labels": list(labels),
        "note": "Smoke self-evaluation only; not a production FAR/FRR estimate.",
    }


def save_model(model: CentroidModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(model.labels), centroids=model.centroids)
