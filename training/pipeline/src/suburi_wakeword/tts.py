from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MODEL_REPO = "https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan/resolve/main"
DEFAULT_MODEL_NAME = "tsukuyomi-chan-6lang-fp16.onnx"
DEFAULT_CONFIG_NAME = "config.json"


@dataclass(frozen=True)
class TtsJob:
    sample_id: str
    phrase_id: str
    label: str
    text: str
    length_scale: float
    source_engine: str = "piper-plus"
    voice: str = "ja_JP-tsukuyomi-chan-medium"
    language: str = "ja"


def _sample_id(label: str, text: str, length_scale: float) -> str:
    digest = hashlib.sha1(f"{label}\0{text}\0{length_scale}".encode("utf-8")).hexdigest()[:12]
    return f"{label}-{digest}"


def build_smoke_jobs(length_scales: Sequence[float] = (1.3, 1.5)) -> list[TtsJob]:
    """Build a tiny Japanese-only wake-word smoke dataset plan."""
    positive_texts = [
        "ハイ、スタックチャン",
        "はい、スタックチャン",
        "ハイスタックチャン",
        "ハイ、ｽﾀｯｸﾁｬﾝ",
    ]
    negative_texts = [
        "スタック",
        "ちゃん",
        "ハイ、ロボットちゃん",
        "こんにちは、今日はいい天気です",
        "スタックは机の上にあります",
        "ねえ、ロボットちゃん",
    ]
    holdout_texts = [
        "Hi, Stack-chan",
    ]
    jobs: list[TtsJob] = []
    for text in positive_texts:
        for scale in length_scales:
            jobs.append(TtsJob(_sample_id("positive", text, scale), "hai_stackchan_ja", "positive", text, float(scale)))
    for text in negative_texts:
        for scale in length_scales[:1]:
            jobs.append(TtsJob(_sample_id("negative", text, scale), "hai_stackchan_ja", "negative", text, float(scale)))
    for text in holdout_texts:
        for scale in length_scales[:1]:
            jobs.append(TtsJob(_sample_id("holdout", text, scale), "hai_stackchan_ja", "holdout", text, float(scale), language="en"))
    return jobs


def ensure_tsukuyomi_model(model_dir: Path) -> tuple[Path, Path]:
    """Download the Piper Plus Tsukuyomi-chan model into an ignored local directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / DEFAULT_MODEL_NAME
    config_path = model_dir / DEFAULT_CONFIG_NAME
    for name, path in [(DEFAULT_MODEL_NAME, model_path), (DEFAULT_CONFIG_NAME, config_path)]:
        if not path.exists():
            urllib.request.urlretrieve(f"{DEFAULT_MODEL_REPO}/{name}?download=true", path)
    return model_path, config_path


def build_infer_command(job: TtsJob, *, model_path: Path, config_path: Path, output_dir: Path) -> list[str]:
    return [
        "uv", "run", "python", "-m", "piper_train.infer_onnx",
        "--model", str(model_path.resolve()),
        "--config", str(config_path.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--text", job.text,
        "--language", job.language,
        "--speaker-id", "0",
        "--length-scale", str(job.length_scale),
        "--device", "cpu",
    ]


def synthesize_job(
    job: TtsJob,
    *,
    piper_plus_python_dir: Path,
    model_path: Path,
    config_path: Path,
    output_wav: Path,
) -> None:
    """Synthesize one job using piper-plus source runtime.

    The current PyPI CLI does not yet handle the speaker_embedding inputs required by
    the 2026 MB-iSTFT Japanese models, so the smoke pipeline uses the upstream
    piper_train.infer_onnx module from a local piper-plus checkout.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_wav.parent
    tmp_output = output_dir / "output.wav"
    if tmp_output.exists():
        tmp_output.unlink()
    cmd = build_infer_command(job, model_path=model_path, config_path=config_path, output_dir=output_dir)
    subprocess.run(cmd, cwd=piper_plus_python_dir, check=True)
    if not tmp_output.exists():
        raise FileNotFoundError(f"piper-plus did not create {tmp_output}")
    tmp_output.replace(output_wav)


def write_job_manifest(jobs: Iterable[TtsJob], audio_root: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            record = asdict(job)
            record["audio_path"] = str((audio_root / f"{job.sample_id}.wav").as_posix())
            record["sample_rate"] = 22050
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
