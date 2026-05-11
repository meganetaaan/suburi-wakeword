from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .augment import count_by_split_label


@dataclass(frozen=True)
class MicroWakeWordTrainingConfig:
    wake_word: str = "hai_stackchan"
    model_name: str = "hai_stackchan_ja"
    probability_cutoff: float = 0.5
    sliding_window_size: int = 5
    tensor_arena_size: int = 60000
    minimum_esphome_version: str = "2024.7.0"
    clip_duration_ms: int = 1500
    window_step_ms: int = 10
    batch_size: int = 16
    training_steps: tuple[int, ...] = (10,)
    learning_rates: tuple[float, ...] = (0.001,)


@dataclass(frozen=True)
class MicroWakeWordTrainingPlan:
    command: list[str]
    cwd: Path
    training_config_path: Path
    expected_tflite: Path
    train_dir: Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_split_records(records: Sequence[dict]) -> None:
    for record in records:
        if "split" not in record:
            raise ValueError(f"record is missing split: {record.get('sample_id')}")


def _write_smoke_tflite(path: Path, records: list[dict], config: MicroWakeWordTrainingConfig) -> None:
    # This is intentionally only for fast pipeline smoke tests. The production
    # handoff path below refuses this marker so placeholder bytes cannot be
    # mistaken for a real microWakeWord export.
    payload = {
        "format": "suburi-microwakeword-smoke-tflite-placeholder",
        "wake_word": config.wake_word,
        "model_name": config.model_name,
        "samples": len(records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"TFL3" + json.dumps(payload, sort_keys=True).encode("utf-8"))


def _is_placeholder_tflite(path: Path) -> bool:
    return b"suburi-microwakeword-smoke-tflite-placeholder" in path.read_bytes()[:4096]


def _write_model_manifest(model_dir: Path, config: MicroWakeWordTrainingConfig, counts: dict, *, backend: str, note: str | None = None) -> None:
    manifest = {
        "type": "micro",
        "wake_word": config.wake_word,
        "author": "meganetaaan/suburi-wakeword",
        "model": "stream_state_internal_quant.tflite",
        "version": 1,
        "micro": {
            "probability_cutoff": config.probability_cutoff,
            "sliding_window_size": config.sliding_window_size,
            "tensor_arena_size": config.tensor_arena_size,
            "minimum_esphome_version": config.minimum_esphome_version,
        },
        "trained_sample_counts": counts,
        "training_backend": backend,
    }
    if note:
        manifest["note"] = note
    (model_dir / f"{config.model_name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _finish_artifact_handoff(base_dir: Path, records: list[dict], config: MicroWakeWordTrainingConfig, source_tflite: Path, *, backend: str, note: str | None = None) -> Path:
    if _is_placeholder_tflite(source_tflite):
        raise ValueError(f"Refusing to publish placeholder TFLite as microWakeWord output: {source_tflite}")
    artifact_dir = base_dir / "artifacts"
    model_dir = artifact_dir / "model"
    metrics_dir = artifact_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    target_tflite = model_dir / "stream_state_internal_quant.tflite"
    shutil.copyfile(source_tflite, target_tflite)
    counts = count_by_split_label(records)
    _write_model_manifest(model_dir, config, counts, backend=backend, note=note)
    (metrics_dir / "offline.json").write_text(
        json.dumps({"sample_counts": counts, "training_backend": backend}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_dir


def train_microwakeword_smoke_model(manifest_path: Path, base_dir: Path, config: MicroWakeWordTrainingConfig) -> Path:
    records = _read_jsonl(manifest_path)
    _validate_split_records(records)
    artifact_dir = base_dir / "artifacts"
    model_dir = artifact_dir / "model"
    metrics_dir = artifact_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    counts = count_by_split_label(records)
    tflite_path = model_dir / "stream_state_internal_quant.tflite"
    _write_smoke_tflite(tflite_path, records, config)
    _write_model_manifest(
        model_dir,
        config,
        counts,
        backend="microWakeWord-smoke-wrapper",
        note="Smoke manifest contract for microWakeWord handoff; not a production accuracy claim.",
    )
    (metrics_dir / "offline.json").write_text(
        json.dumps({"sample_counts": counts, "note": "Threshold sweep is written separately from validation scores."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_dir


def build_microwakeword_training_plan(
    manifest_path: Path,
    base_dir: Path,
    config: MicroWakeWordTrainingConfig,
    *,
    source_dir: Path,
    model_architecture: str = "mixednet",
) -> MicroWakeWordTrainingPlan:
    records = _read_jsonl(manifest_path)
    _validate_split_records(records)
    train_dir = base_dir / "artifacts" / "microwakeword-train"
    train_dir.mkdir(parents=True, exist_ok=True)
    positive_dir = base_dir / "features" / "positive"
    negative_dir = base_dir / "features" / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    training_config_path = train_dir / "training_config.json"
    training_config = {
        "train_dir": str(train_dir),
        "clip_duration_ms": config.clip_duration_ms,
        "window_step_ms": config.window_step_ms,
        "batch_size": config.batch_size,
        "training_steps": list(config.training_steps),
        "learning_rates": list(config.learning_rates),
        "features": [
            {
                "features_dir": str(positive_dir),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(negative_dir),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
        ],
    }
    training_config_path.write_text(json.dumps(training_config, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "microwakeword.model_train_eval",
        "--training_config",
        str(training_config_path),
        "--train",
        "1",
        "--test_tflite_streaming_quantized",
        "1",
        model_architecture,
    ]
    expected_tflite = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    return MicroWakeWordTrainingPlan(command=command, cwd=source_dir, training_config_path=training_config_path, expected_tflite=expected_tflite, train_dir=train_dir)


def train_microwakeword_model(
    manifest_path: Path,
    base_dir: Path,
    config: MicroWakeWordTrainingConfig,
    *,
    source_dir: Path | None = None,
    model_architecture: str = "mixednet",
    prebuilt_tflite: Path | None = None,
) -> Path:
    records = _read_jsonl(manifest_path)
    _validate_split_records(records)
    if prebuilt_tflite is None:
        if source_dir is None:
            raise ValueError("source_dir is required when prebuilt_tflite is not provided")
        plan = build_microwakeword_training_plan(manifest_path, base_dir, config, source_dir=source_dir, model_architecture=model_architecture)
        subprocess.run(plan.command, cwd=plan.cwd, check=True)
        prebuilt_tflite = plan.expected_tflite
    if not prebuilt_tflite.exists():
        raise FileNotFoundError(f"microWakeWord did not create expected TFLite: {prebuilt_tflite}")
    return _finish_artifact_handoff(
        base_dir,
        records,
        config,
        prebuilt_tflite,
        backend="microWakeWord",
        note="Generated by upstream microWakeWord training/export path.",
    )
