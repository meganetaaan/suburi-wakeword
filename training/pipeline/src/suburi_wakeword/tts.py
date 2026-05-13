from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MODEL_REPO = "https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan/resolve/main"
DEFAULT_MODEL_NAME = "tsukuyomi-chan-6lang-fp16.onnx"
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_VOICE = "ja_JP-tsukuyomi-chan-medium"


DEFAULT_POSITIVE_TEXTS = (
    "ハイ、スタックチャン",
    "はい、スタックチャン",
    "ハイスタックチャン",
    "ハイ、ｽﾀｯｸﾁｬﾝ",
)
DEFAULT_EXPANDED_POSITIVE_TEXTS = (
    *DEFAULT_POSITIVE_TEXTS,
    "ハイ スタックチャン",
    "はいスタックチャン",
    "ハイ、スタックちゃん",
    "はい、スタックちゃん",
    "ハイ、すたっくちゃん",
    "はい、すたっくちゃん",
)
DEFAULT_NEGATIVE_TEXTS = (
    "スタック",
    "ちゃん",
    "ハイ、ロボットちゃん",
    "こんにちは、今日はいい天気です",
    "スタックは机の上にあります",
    "ねえ、ロボットちゃん",
)
DEFAULT_EXPANDED_NEGATIVE_TEXTS = (
    *DEFAULT_NEGATIVE_TEXTS,
    "ハイ、スタッフさん",
    "はい、スタッフさん",
    "ハイ、スタートちゃん",
    "はい、ストックちゃん",
    "スタックチャンネル",
    "スタックちゃんと呼びました",
    "ねえ、スタックチャンネルを開いて",
    "ハイ、スタックはどこですか",
)
DEFAULT_HOLDOUT_TEXTS = ("Hi, Stack-chan",)
DEFAULT_EXPANDED_5K_POSITIVE_TEXTS = (
    *DEFAULT_EXPANDED_POSITIVE_TEXTS,
    "ハイー、スタックチャン",
    "はいー、スタックチャン",
)
DEFAULT_EXPANDED_5K_NEGATIVE_TEXTS = (
    *DEFAULT_EXPANDED_NEGATIVE_TEXTS,
    "ハイ、スタッフちゃん",
    "はい、スタッフちゃん",
    "ハイ、ストックさん",
    "はい、ストックさん",
    "ハイ、スタックさん",
    "はい、スタックさん",
    "ハイ、スタックチャンネル",
    "はい、スタックチャンネル",
    "スタックちゃん、こんにちは",
    "スタックチャンネルです",
)
DEFAULT_EXPANDED_5K_HARD_NEGATIVE_TEXTS = (
    *DEFAULT_EXPANDED_5K_NEGATIVE_TEXTS,
    "ねえ、スタックチャンネル",
    "スタックチャンネルを開いて",
    "ハイ、スタックチャンネルを開いて",
    "はい、スタックチャンネルを開いて",
    "スタックちゃんです",
    "ハイ、スタックちゃんと呼びました",
    "ハイ、スタックちゃん、こんにちは",
    "はい、スタックちゃん、こんにちは",
)
DEFAULT_EXPANDED_5K_LEXICAL_NEGATIVE_TEXTS = (
    *DEFAULT_EXPANDED_5K_NEGATIVE_TEXTS,
    "ハイ、スタッキーちゃん",
    "はい、スタッキーちゃん",
    "ハイ、スタックあんちゃん",
    "はい、スタックあんちゃん",
    "スタックちゃんのとなり",
    "スタックちゃんではありません",
)
DEFAULT_EXPANDED_5K_HOLDOUT_TEXTS = (
    "Hi, Stack-chan",
    "Hey, Stack-chan",
)


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    voice: str = DEFAULT_VOICE
    speaker_id: int = 0
    source_engine: str = "piper-plus"
    model_id: str = "tsukuyomi-chan-6lang-fp16"


@dataclass(frozen=True)
class ProsodyVariant:
    id: str
    length_scale: float
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8


DEFAULT_EXPANDED_5K_PROSODY_VARIANTS = (
    ProsodyVariant(id="fast-flat", length_scale=0.85, noise_scale=0.55, noise_scale_w=0.6),
    ProsodyVariant(id="fast-default", length_scale=0.95, noise_scale=0.667, noise_scale_w=0.8),
    ProsodyVariant(id="normal-flat", length_scale=1.1, noise_scale=0.55, noise_scale_w=0.7),
    ProsodyVariant(id="normal-default", length_scale=1.2, noise_scale=0.667, noise_scale_w=0.8),
    ProsodyVariant(id="slow-lively", length_scale=1.35, noise_scale=0.8, noise_scale_w=1.05),
    ProsodyVariant(id="very-slow-varied", length_scale=1.55, noise_scale=0.9, noise_scale_w=1.25),
)


@dataclass(frozen=True)
class TtsJob:
    sample_id: str
    phrase_id: str
    label: str
    text: str
    length_scale: float
    source_engine: str = "piper-plus"
    voice: str = DEFAULT_VOICE
    language: str = "ja"
    voice_profile_id: str = "tsukuyomi"
    model_id: str = "tsukuyomi-chan-6lang-fp16"
    speaker_id: int = 0
    prosody_id: str = "default"
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8


def _sample_id(
    label: str,
    text: str,
    *,
    voice_profile_id: str,
    prosody_id: str,
    length_scale: float,
    noise_scale: float,
    noise_scale_w: float,
    speaker_id: int,
    language: str,
) -> str:
    digest = hashlib.sha1(
        "\0".join(
            [
                label,
                text,
                voice_profile_id,
                prosody_id,
                str(length_scale),
                str(noise_scale),
                str(noise_scale_w),
                str(speaker_id),
                language,
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{label}-{digest}"


def _job(
    *,
    phrase_id: str,
    label: str,
    text: str,
    language: str,
    voice: VoiceProfile,
    prosody: ProsodyVariant,
) -> TtsJob:
    return TtsJob(
        sample_id=_sample_id(
            label,
            text,
            voice_profile_id=voice.id,
            prosody_id=prosody.id,
            length_scale=prosody.length_scale,
            noise_scale=prosody.noise_scale,
            noise_scale_w=prosody.noise_scale_w,
            speaker_id=voice.speaker_id,
            language=language,
        ),
        phrase_id=phrase_id,
        label=label,
        text=text,
        length_scale=float(prosody.length_scale),
        source_engine=voice.source_engine,
        voice=voice.voice,
        language=language,
        voice_profile_id=voice.id,
        model_id=voice.model_id,
        speaker_id=int(voice.speaker_id),
        prosody_id=prosody.id,
        noise_scale=float(prosody.noise_scale),
        noise_scale_w=float(prosody.noise_scale_w),
    )


def build_expanded_jobs(
    *,
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    holdout_texts: Sequence[str],
    voices: Sequence[VoiceProfile],
    prosody_variants: Sequence[ProsodyVariant],
    phrase_id: str = "hai_stackchan_ja",
) -> list[TtsJob]:
    """Build a multiplicative TTS plan across phrase, voice, and prosody axes.

    English near-miss holdouts stay outside positives even when all other axes are
    multiplied. This function only plans synthesis; generated audio remains ignored.
    """
    jobs: list[TtsJob] = []
    for label, texts, language in (
        ("positive", positive_texts, "ja"),
        ("negative", negative_texts, "ja"),
        ("holdout", holdout_texts, "en"),
    ):
        for text in texts:
            for voice in voices:
                for prosody in prosody_variants:
                    jobs.append(_job(phrase_id=phrase_id, label=label, text=text, language=language, voice=voice, prosody=prosody))
    return jobs


def build_smoke_jobs(length_scales: Sequence[float] = (1.3, 1.5)) -> list[TtsJob]:
    """Build a tiny Japanese-only wake-word smoke dataset plan."""
    prosody_variants = tuple(
        ProsodyVariant(id=f"length-{scale:g}", length_scale=float(scale)) for scale in length_scales
    )
    return build_expanded_jobs(
        positive_texts=DEFAULT_POSITIVE_TEXTS,
        negative_texts=DEFAULT_NEGATIVE_TEXTS,
        holdout_texts=DEFAULT_HOLDOUT_TEXTS,
        voices=(VoiceProfile(id="tsukuyomi"),),
        prosody_variants=prosody_variants,
    )


def build_large_synthetic_jobs(
    *,
    voices: Sequence[VoiceProfile] = (VoiceProfile(id="tsukuyomi"),),
    prosody_variants: Sequence[ProsodyVariant] = (
        ProsodyVariant(id="fast-flat", length_scale=0.9, noise_scale=0.55, noise_scale_w=0.6),
        ProsodyVariant(id="fast-default", length_scale=1.0, noise_scale=0.667, noise_scale_w=0.8),
        ProsodyVariant(id="default", length_scale=1.15, noise_scale=0.667, noise_scale_w=0.8),
        ProsodyVariant(id="slow-lively", length_scale=1.35, noise_scale=0.8, noise_scale_w=1.05),
        ProsodyVariant(id="very-slow-varied", length_scale=1.6, noise_scale=0.9, noise_scale_w=1.25),
    ),
) -> list[TtsJob]:
    """Build a larger multiplicative synthetic dataset plan.

    The default keeps one currently verified Tsukuyomi voice/model profile, while
    the API accepts additional `VoiceProfile`s so different models/speakers can be
    multiplied in without changing downstream manifests.
    """
    return build_expanded_jobs(
        positive_texts=DEFAULT_EXPANDED_POSITIVE_TEXTS,
        negative_texts=DEFAULT_EXPANDED_NEGATIVE_TEXTS,
        holdout_texts=DEFAULT_HOLDOUT_TEXTS,
        voices=voices,
        prosody_variants=prosody_variants,
    )


def build_expanded_5k_jobs(*, voices: Sequence[VoiceProfile]) -> list[TtsJob]:
    """Build the recommended ~5k utterance plan before real microWakeWord training.

    Counts before augmentation with four voice profiles:
    12 positive + 24 negative + 2 holdout phrases, times 6 prosody variants,
    times 4 voices = 912 base TTS jobs. The standard smoke augmentation expands
    this to 6,384 utterances including raw originals.
    """
    return build_expanded_jobs(
        positive_texts=DEFAULT_EXPANDED_5K_POSITIVE_TEXTS,
        negative_texts=DEFAULT_EXPANDED_5K_NEGATIVE_TEXTS,
        holdout_texts=DEFAULT_EXPANDED_5K_HOLDOUT_TEXTS,
        voices=voices,
        prosody_variants=DEFAULT_EXPANDED_5K_PROSODY_VARIANTS,
    )


def build_expanded_5k_hard_negative_jobs(*, voices: Sequence[VoiceProfile]) -> list[TtsJob]:
    """Build expanded-5k plus targeted hard negatives from real false accepts."""
    return build_expanded_jobs(
        positive_texts=DEFAULT_EXPANDED_5K_POSITIVE_TEXTS,
        negative_texts=DEFAULT_EXPANDED_5K_HARD_NEGATIVE_TEXTS,
        holdout_texts=DEFAULT_EXPANDED_5K_HOLDOUT_TEXTS,
        voices=voices,
        prosody_variants=DEFAULT_EXPANDED_5K_PROSODY_VARIANTS,
    )


def build_expanded_5k_lexical_negative_jobs(*, voices: Sequence[VoiceProfile]) -> list[TtsJob]:
    """Build expanded-5k plus lexical impostors that should stay non-wake."""
    return build_expanded_jobs(
        positive_texts=DEFAULT_EXPANDED_5K_POSITIVE_TEXTS,
        negative_texts=DEFAULT_EXPANDED_5K_LEXICAL_NEGATIVE_TEXTS,
        holdout_texts=DEFAULT_EXPANDED_5K_HOLDOUT_TEXTS,
        voices=voices,
        prosody_variants=DEFAULT_EXPANDED_5K_PROSODY_VARIANTS,
    )


def load_voice_profiles(path: Path) -> tuple[VoiceProfile, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("voices", data if isinstance(data, list) else None)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Voice profile config must contain a non-empty voices list: {path}")
    return tuple(
        VoiceProfile(
            id=str(row["id"]),
            voice=str(row.get("voice", DEFAULT_VOICE)),
            speaker_id=int(row.get("speaker_id", 0)),
            source_engine=str(row.get("source_engine", "piper-plus")),
            model_id=str(row.get("model_id", "tsukuyomi-chan-6lang-fp16")),
        )
        for row in rows
    )


def summarize_jobs(jobs: Sequence[TtsJob], *, augmentation_multiplier: int = 1) -> dict:
    labels = Counter(job.label for job in jobs)
    return {
        "base_jobs": len(jobs),
        "estimated_after_augmentation": len(jobs) * int(augmentation_multiplier),
        "labels": dict(labels),
        "positive_base_jobs": labels.get("positive", 0),
        "negative_base_jobs": labels.get("negative", 0),
        "holdout_base_jobs": labels.get("holdout", 0),
        "voice_profiles": len({job.voice_profile_id for job in jobs}),
        "prosody_variants": len({job.prosody_id for job in jobs}),
        "texts": len({job.text for job in jobs}),
    }


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
        "--speaker-id", str(job.speaker_id),
        "--length-scale", str(job.length_scale),
        "--noise-scale", str(job.noise_scale),
        "--noise-scale-w", str(job.noise_scale_w),
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
