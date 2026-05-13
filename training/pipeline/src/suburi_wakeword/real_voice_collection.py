from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


DecodeAudio = Callable[[Path, Path], None]


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_audio_path(raw: str, *, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    pipeline_candidate = manifest_path.parents[2] / path if len(manifest_path.parents) > 2 else candidate
    if pipeline_candidate.exists():
        return pipeline_candidate
    return candidate


def build_ffmpeg_decode_command(source_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def decode_with_ffmpeg(source_path: Path, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode browser WebM/Opus recordings into 16 kHz PCM WAV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_decode_command(source_path, output_path), check=True)


def summarize_collection_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    rows = _read_jsonl(manifest_path)
    missing_audio = []
    total_bytes = 0
    durations = []
    for row in rows:
        raw = row.get("audio_path")
        if raw:
            audio_path = _resolve_audio_path(str(raw), manifest_path=manifest_path)
            if audio_path.exists():
                total_bytes += audio_path.stat().st_size
            else:
                missing_audio.append(str(audio_path))
        duration = row.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
    return {
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "participants": sorted({str(row.get("participant_id")) for row in rows if row.get("participant_id")}),
        "labels": dict(Counter(str(row.get("label")) for row in rows)),
        "texts": dict(Counter(str(row.get("text")) for row in rows)),
        "formats": dict(Counter(str(row.get("audio_format")) for row in rows)),
        "missing_audio_count": len(missing_audio),
        "missing_audio": missing_audio,
        "total_audio_bytes": total_bytes,
        "duration_ms_min": min(durations) if durations else None,
        "duration_ms_max": max(durations) if durations else None,
        "duration_ms_mean": (sum(durations) / len(durations)) if durations else None,
    }


def _safe_id(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text).strip("-") or "unknown"


def build_collection_eval_manifest(
    collection_manifest_path: Path,
    output_manifest_path: Path,
    *,
    decode_audio: DecodeAudio = decode_with_ffmpeg,
) -> dict:
    collection_manifest_path = Path(collection_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    wav_root = output_manifest_path.parent / "wav16k"
    rows = _read_jsonl(collection_manifest_path)
    output_rows = []
    for row in rows:
        source_path = _resolve_audio_path(str(row.get("audio_path") or ""), manifest_path=collection_manifest_path)
        if not source_path.exists():
            raise FileNotFoundError(f"recording audio is missing: {source_path}")
        participant_id = _safe_id(row.get("participant_id"))
        take_id = _safe_id(row.get("take_id") or row.get("prompt_id"))
        label = str(row.get("label"))
        prompt_id = _safe_id(row.get("prompt_id"))
        output_wav = wav_root / label / prompt_id / f"{participant_id}_{take_id}.wav"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        decode_audio(source_path, output_wav)
        output_rows.append(
            {
                "sample_id": f"real_{participant_id}_{take_id}",
                "label": label,
                "split": "real_holdout",
                "audio_path": str(output_wav),
                "normalized_audio_path": str(output_wav),
                "source_text": row.get("text"),
                "text": row.get("text"),
                "participant_id": participant_id,
                "prompt_id": row.get("prompt_id"),
                "take_id": row.get("take_id"),
                "recorded_at": row.get("recorded_at"),
                "collection_audio_path": str(source_path),
            }
        )
    with output_manifest_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "collection_manifest": str(collection_manifest_path),
        "output_manifest": str(output_manifest_path),
        "sample_count": len(output_rows),
        "labels": dict(Counter(row["label"] for row in output_rows)),
        "wav_root": str(wav_root),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare browser-collected real voice recordings for evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Summarize a collection manifest without decoding audio.")
    summarize.add_argument("manifest", type=Path)

    prepare = subparsers.add_parser("prepare-eval", help="Decode recordings to 16 kHz mono WAV and write an eval manifest.")
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output_manifest", type=Path)

    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize_collection_manifest(args.manifest)
    else:
        result = build_collection_eval_manifest(args.manifest, args.output_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
