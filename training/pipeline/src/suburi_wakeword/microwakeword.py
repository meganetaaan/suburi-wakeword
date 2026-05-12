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
    probability_cutoff: float = 0.9
    sliding_window_size: int = 5
    tensor_arena_size: int = 60000
    minimum_esphome_version: str = "2024.7.0"
    clip_duration_ms: int = 1500
    window_step_ms: int = 10
    batch_size: int = 16
    training_steps: tuple[int, ...] = (10,)
    learning_rates: tuple[float, ...] = (0.001,)
    positive_class_weight: float = 1.0
    negative_class_weight: float = 1.0


@dataclass(frozen=True)
class MicroWakeWordTrainingPlan:
    command: list[str]
    cwd: Path
    training_config_path: Path
    expected_tflite: Path
    train_dir: Path
    feature_plan: MicroWakeWordFeaturePlan


@dataclass(frozen=True)
class MicroWakeWordFeaturePlan:
    command: list[str]
    cwd: Path
    script_path: Path
    positive_features_dir: Path
    negative_features_dir: Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_split_records(records: Sequence[dict]) -> None:
    for record in records:
        if "split" not in record:
            raise ValueError(f"record is missing split: {record.get('sample_id')}")


def _record_audio_path(record: dict, base_dir: Path) -> Path:
    path = Path(record["normalized_audio_path"])
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _mww_split_name(record: dict) -> str:
    split = record["split"]
    if split == "train":
        return "training"
    if split == "validation":
        return "validation"
    if split == "holdout":
        return "testing"
    if split in {"testing", "testing_ambient", "validation_ambient"}:
        return split
    raise ValueError(f"unsupported split for microWakeWord features: {split}")


def _feature_class_dir(record: dict, positive_features_dir: Path, negative_features_dir: Path) -> Path:
    return positive_features_dir if record["label"] == "positive" else negative_features_dir


def _stage_feature_audio(records: Sequence[dict], base_dir: Path, positive_features_dir: Path, negative_features_dir: Path) -> None:
    for record in records:
        source = _record_audio_path(record, base_dir)
        if not source.exists():
            raise FileNotFoundError(f"normalized audio is missing for {record.get('sample_id')}: {source}")
        class_dir = _feature_class_dir(record, positive_features_dir, negative_features_dir)
        split_dir = class_dir / _mww_split_name(record) / "wav"
        split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, split_dir / f"{record['sample_id']}.wav")
        if record["label"] == "positive" and record["split"] == "validation":
            # Upstream ROC evaluation expects positive samples in `testing`.
            # Until a dedicated positive test split exists, mirror validation positives
            # so export/evaluation can complete without mixing holdout into positives.
            testing_dir = class_dir / "testing" / "wav"
            testing_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, testing_dir / f"{record['sample_id']}.wav")


def _write_feature_generation_script(path: Path, positive_features_dir: Path, negative_features_dir: Path, *, step_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''from pathlib import Path

import wave

import numpy as np
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.audio.spectrograms import SpectrogramGeneration  # documents upstream equivalent


FEATURE_DIRS = [{str(positive_features_dir)!r}, {str(negative_features_dir)!r}]
SPLITS = ["training", "validation", "testing"]
STEP_MS = {int(step_ms)}


def load_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2 or sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz 16-bit PCM WAV: {{path}}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def spectrogram_generator(wav_dir: Path, split: str):
    repeat = 2 if split == "training" else 1
    for _ in range(repeat):
        for wav_path in sorted(wav_dir.glob("*.wav")):
            yield generate_features_for_clip(load_wav_mono(wav_path), STEP_MS)


def generate_split(features_dir: Path, split: str) -> None:
    wav_dir = features_dir / split / "wav"
    if not wav_dir.exists() or not list(wav_dir.glob("*.wav")):
        return
    out_dir = features_dir / split / "wakeword_mmap"
    if out_dir.exists():
        return
    RaggedMmap.from_generator(
        out_dir=str(out_dir),
        sample_generator=spectrogram_generator(wav_dir, split),
        batch_size=100,
        verbose=True,
    )


def main() -> None:
    for feature_dir in FEATURE_DIRS:
        for split in SPLITS:
            generate_split(Path(feature_dir), split)


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def build_microwakeword_feature_plan(
    manifest_path: Path,
    base_dir: Path,
    *,
    source_dir: Path,
    config: MicroWakeWordTrainingConfig = MicroWakeWordTrainingConfig(),
) -> MicroWakeWordFeaturePlan:
    records = _read_jsonl(manifest_path)
    _validate_split_records(records)
    positive_features_dir = base_dir / "features" / "positive"
    negative_features_dir = base_dir / "features" / "negative"
    positive_features_dir.mkdir(parents=True, exist_ok=True)
    negative_features_dir.mkdir(parents=True, exist_ok=True)
    _stage_feature_audio(records, base_dir, positive_features_dir, negative_features_dir)
    script_path = base_dir / "artifacts" / "microwakeword-train" / "generate_features.py"
    _write_feature_generation_script(script_path, positive_features_dir.resolve(), negative_features_dir.resolve(), step_ms=config.window_step_ms)
    return MicroWakeWordFeaturePlan(
        command=["uv", "run", "--no-sync", "python", str(script_path.resolve())],
        cwd=source_dir,
        script_path=script_path,
        positive_features_dir=positive_features_dir,
        negative_features_dir=negative_features_dir,
    )


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
    train_dir = base_dir / "artifacts" / "microwakeword-model"
    feature_plan = build_microwakeword_feature_plan(manifest_path, base_dir, source_dir=source_dir, config=config)
    training_config_dir = base_dir / "artifacts" / "microwakeword-train"
    training_config_dir.mkdir(parents=True, exist_ok=True)
    training_config_path = training_config_dir / "training_config.json"
    training_config = {
        "train_dir": str(train_dir.resolve()),
        "clip_duration_ms": config.clip_duration_ms,
        "window_step_ms": config.window_step_ms,
        "batch_size": config.batch_size,
        "training_steps": list(config.training_steps),
        "learning_rates": list(config.learning_rates),
        "positive_class_weight": [float(config.positive_class_weight)],
        "negative_class_weight": [float(config.negative_class_weight)],
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 1,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "accuracy",
        "features": [
            {
                "features_dir": str(feature_plan.positive_features_dir.resolve()),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(feature_plan.negative_features_dir.resolve()),
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
        "--no-sync",
        "python",
        "-m",
        "microwakeword.model_train_eval",
        "--training_config",
        str(training_config_path.resolve()),
        "--train",
        "1",
        "--test_tflite_streaming_quantized",
        "1",
        model_architecture,
        "--residual_connection",
        "0,0,0,0",
    ]
    expected_tflite = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    return MicroWakeWordTrainingPlan(
        command=command,
        cwd=source_dir,
        training_config_path=training_config_path,
        expected_tflite=expected_tflite,
        train_dir=train_dir,
        feature_plan=feature_plan,
    )


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
        subprocess.run(plan.feature_plan.command, cwd=plan.feature_plan.cwd, check=True)
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
