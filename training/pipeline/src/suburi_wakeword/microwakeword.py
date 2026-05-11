from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .augment import count_by_split_label


@dataclass(frozen=True)
class MicroWakeWordTrainingConfig:
    wake_word: str = "hai_stackchan"
    model_name: str = "hai_stackchan_ja"
    probability_cutoff: float = 0.5
    sliding_window_size: int = 5
    tensor_arena_size: int = 60000
    minimum_esphome_version: str = "2024.7.0"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_smoke_tflite(path: Path, records: list[dict], config: MicroWakeWordTrainingConfig) -> None:
    # This is an intentionally tiny placeholder artifact for pipeline smoke tests.
    # Real training should replace this byte stream with microWakeWord's quantized
    # stream_state_internal_quant.tflite output while keeping the same manifest contract.
    payload = {
        "format": "suburi-microwakeword-smoke-tflite-placeholder",
        "wake_word": config.wake_word,
        "model_name": config.model_name,
        "samples": len(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"TFL3" + json.dumps(payload, sort_keys=True).encode("utf-8"))


def train_microwakeword_smoke_model(manifest_path: Path, base_dir: Path, config: MicroWakeWordTrainingConfig) -> Path:
    records = _read_jsonl(manifest_path)
    for record in records:
        if "split" not in record:
            raise ValueError(f"record is missing split: {record.get('sample_id')}")
    artifact_dir = base_dir / "artifacts"
    model_dir = artifact_dir / "model"
    metrics_dir = artifact_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    counts = count_by_split_label(records)
    tflite_path = model_dir / "stream_state_internal_quant.tflite"
    _write_smoke_tflite(tflite_path, records, config)

    manifest = {
        "type": "micro",
        "wake_word": config.wake_word,
        "author": "meganetaaan/suburi-wakeword",
        "model": tflite_path.name,
        "version": 1,
        "micro": {
            "probability_cutoff": config.probability_cutoff,
            "sliding_window_size": config.sliding_window_size,
            "tensor_arena_size": config.tensor_arena_size,
            "minimum_esphome_version": config.minimum_esphome_version,
        },
        "trained_sample_counts": counts,
        "training_backend": "microWakeWord-smoke-wrapper",
        "note": "Smoke manifest contract for microWakeWord handoff; not a production accuracy claim.",
    }
    (model_dir / f"{config.model_name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "offline.json").write_text(
        json.dumps({"sample_counts": counts, "note": "Threshold sweep is written separately from validation scores."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_dir
