from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .audio import normalize_to_wav16k
from .smoke_train import evaluate_self, load_features_from_manifest, save_model, train_centroid_model
from .tts import build_smoke_jobs, ensure_tsukuyomi_model, synthesize_job, write_job_manifest


def ensure_piper_plus_source(path: Path) -> Path:
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", "--branch", "dev", "https://github.com/ayutaz/piper-plus.git", str(path)], check=True)
    python_dir = path / "src" / "python"
    subprocess.run(["uv", "add", "onnxruntime", "soundfile", "pyopenjtalk-plus", "--dev"], cwd=python_dir, check=True)
    return python_dir


def run(output_root: Path, samples_per_variant: int = 1) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    local_tools = Path(".local-data/tools")
    piper_python_dir = ensure_piper_plus_source(local_tools / "piper-plus")
    model_path, config_path = ensure_tsukuyomi_model(Path(".local-data/piper-models/tsukuyomi"))

    raw_audio_root = output_root / "raw"
    normalized_root = output_root / "normalized"
    jobs = build_smoke_jobs(length_scales=(1.3, 1.5)[: max(1, samples_per_variant)])
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
            "raw_audio_path": str(raw_path),
            "normalized_audio_path": str(norm_path),
        })

    dataset_manifest = output_root / "dataset.jsonl"
    dataset_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_records), encoding="utf-8")

    features, labels, _ = load_features_from_manifest(dataset_manifest, Path("."))
    model = train_centroid_model(features, labels)
    metrics = evaluate_self(model, features, labels)
    artifact_dir = output_root / "artifacts"
    save_model(model, artifact_dir / "centroid-smoke-model.npz")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "README.md").write_text(
        "# Smoke wake-word model\n\n"
        "This is a tiny centroid classifier used only to validate the local Piper TTS, "
        "dataset, feature extraction, and training/evaluation plumbing. It is not the final microWakeWord model.\n",
        encoding="utf-8",
    )
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("runs/smoke/hai_stackchan_ja"))
    parser.add_argument("--samples-per-variant", type=int, default=1)
    args = parser.parse_args()
    artifact_dir = run(args.output_root, args.samples_per_variant)
    print(artifact_dir)


if __name__ == "__main__":
    main()
