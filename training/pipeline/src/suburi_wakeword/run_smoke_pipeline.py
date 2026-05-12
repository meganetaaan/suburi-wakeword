from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .audio import normalize_to_wav16k, write_wav_mono16
from .augment import AugmentationPlan, augment_manifest_records
from .dataset_split import assign_splits
from .microwakeword import MicroWakeWordTrainingConfig, train_microwakeword_smoke_model
from .threshold_sweep import default_thresholds, sweep_thresholds
from .tts import build_large_synthetic_jobs, build_smoke_jobs, ensure_tsukuyomi_model, synthesize_job, write_job_manifest


def ensure_piper_plus_source(path: Path) -> Path:
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", "--branch", "dev", "https://github.com/ayutaz/piper-plus.git", str(path)], check=True)
    python_dir = path / "src" / "python"
    subprocess.run(["uv", "add", "onnxruntime", "soundfile", "pyopenjtalk-plus", "g2p-en", "--dev"], cwd=python_dir, check=True)
    subprocess.run([
        "uv", "run", "python", "-c",
        "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('cmudict', quiet=True)",
    ], cwd=python_dir, check=True)
    return python_dir


def _ensure_smoke_background_noise(output_root: Path) -> Path:
    noise_path = output_root / "background" / "synthetic-room-tone.wav"
    if not noise_path.exists():
        rng = np.random.default_rng(20260512)
        write_wav_mono16(noise_path, rng.normal(0.0, 0.02, 16000).astype(np.float32), 16000)
    return noise_path


def _smoke_validation_scores(records: list[dict]) -> list[dict]:
    return [
        {
            "sample_id": record["sample_id"],
            "label": record["label"],
            "split": record["split"],
            "score": 0.82 if record["label"] == "positive" else 0.18 if record["label"] == "negative" else 0.35,
        }
        for record in records
        if record["split"] in {"validation", "holdout"}
    ]


def _length_scales_for_count(count: int) -> tuple[float, ...]:
    count = max(1, int(count))
    if count == 1:
        return (1.3,)
    return tuple(round(float(value), 3) for value in np.linspace(1.1, 1.7, count))


def build_jobs_for_profile(profile: str, *, samples_per_variant: int = 1):
    if profile == "smoke":
        return build_smoke_jobs(length_scales=_length_scales_for_count(samples_per_variant))
    if profile == "expanded":
        return build_large_synthetic_jobs()
    raise ValueError(f"Unknown dataset profile: {profile}")


def run(output_root: Path, samples_per_variant: int = 1, dataset_profile: str = "smoke") -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    local_tools = Path(".local-data/tools")
    piper_python_dir = ensure_piper_plus_source(local_tools / "piper-plus")
    model_path, config_path = ensure_tsukuyomi_model(Path(".local-data/piper-models/tsukuyomi"))

    raw_audio_root = output_root / "raw"
    normalized_root = output_root / "normalized"
    jobs = build_jobs_for_profile(dataset_profile, samples_per_variant=samples_per_variant)
    write_job_manifest(jobs, raw_audio_root, output_root / "tts-jobs.jsonl")

    manifest_records = []
    for job in jobs:
        raw_path = raw_audio_root / f"{job.sample_id}.wav"
        norm_path = normalized_root / f"{job.sample_id}.wav"
        if not raw_path.exists():
            synthesize_job(job, piper_plus_python_dir=piper_python_dir, model_path=model_path, config_path=config_path, output_wav=raw_path)
        normalize_to_wav16k(raw_path, norm_path)
        manifest_records.append({
            "sample_id": job.sample_id,
            "phrase_id": job.phrase_id,
            "label": job.label,
            "text": job.text,
            "source": "tts_local",
            "source_engine": job.source_engine,
            "voice": job.voice,
            "language": job.language,
            "length_scale": job.length_scale,
            "noise_scale": job.noise_scale,
            "noise_scale_w": job.noise_scale_w,
            "voice_profile_id": job.voice_profile_id,
            "model_id": job.model_id,
            "speaker_id": job.speaker_id,
            "prosody_id": job.prosody_id,
            "raw_audio_path": str(raw_path),
            "normalized_audio_path": str(norm_path),
        })

    noise_path = _ensure_smoke_background_noise(output_root)
    augmented_records = augment_manifest_records(
        manifest_records,
        output_dir=output_root / "augmented",
        plan=AugmentationPlan(
            speed_factors=(0.9, 1.1),
            gain_db=(-3.0, 3.0),
            reverb_decays=(0.25,),
            background_noise_paths=(noise_path,),
            background_noise_snr_db=(12.0,),
        ),
    )
    split_records = assign_splits([*manifest_records, *augmented_records], holdout_labels={"holdout"})

    dataset_manifest = output_root / "dataset.jsonl"
    dataset_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split_records), encoding="utf-8")

    artifact_dir = train_microwakeword_smoke_model(
        dataset_manifest,
        output_root,
        MicroWakeWordTrainingConfig(wake_word="hai_stackchan", model_name="hai_stackchan_ja", probability_cutoff=0.5),
    )
    metrics_dir = artifact_dir / "metrics"
    validation_scores = _smoke_validation_scores(split_records)
    threshold_rows = sweep_thresholds(validation_scores, thresholds=default_thresholds())
    (metrics_dir / "threshold-sweep.json").write_text(json.dumps(threshold_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "validation-scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in validation_scores), encoding="utf-8")
    (artifact_dir / "README.md").write_text(
        "# microWakeWord smoke artifacts\n\n"
        "This run validates the Piper TTS, augmentation, split, threshold-sweep, and "
        "microWakeWord manifest handoff contracts. The `.tflite` is a smoke placeholder until "
        "the pinned microWakeWord trainer is wired in. Do not treat these metrics as production FAR/FRR.\n",
        encoding="utf-8",
    )
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs/smoke/hai_stackchan_ja"))
    parser.add_argument("--samples-per-variant", type=int, default=1)
    parser.add_argument("--dataset-profile", choices=("smoke", "expanded"), default="smoke")
    args = parser.parse_args()
    artifact_dir = run(args.output_root, args.samples_per_variant, args.dataset_profile)
    print(artifact_dir)


if __name__ == "__main__":
    main()
