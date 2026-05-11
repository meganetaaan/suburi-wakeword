from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .audio import _resample_linear, load_wav_mono, write_wav_mono16


@dataclass(frozen=True)
class AugmentationPlan:
    speed_factors: Sequence[float] = (0.9, 1.1)
    gain_db: Sequence[float] = (-3.0, 3.0)
    reverb_decays: Sequence[float] = (0.25,)
    background_noise_paths: Sequence[Path] = ()
    background_noise_snr_db: Sequence[float] = (12.0,)


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("\0".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


def _speed(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    # factor > 1 shortens audio, factor < 1 lengthens it.
    virtual_rate = int(round(16000 * factor))
    return _resample_linear(audio, virtual_rate, 16000)


def _gain(audio: np.ndarray, db: float) -> np.ndarray:
    return np.clip(audio * (10.0 ** (db / 20.0)), -1.0, 1.0).astype(np.float32)


def _reverb(audio: np.ndarray, decay: float) -> np.ndarray:
    if decay <= 0:
        raise ValueError("reverb decay must be positive")
    delays = [int(16000 * 0.035), int(16000 * 0.071)]
    wet = audio.astype(np.float32, copy=True)
    for idx, delay in enumerate(delays, start=1):
        if delay >= len(wet):
            continue
        wet[delay:] += audio[:-delay] * (decay ** idx)
    peak = float(np.max(np.abs(wet))) if len(wet) else 0.0
    if peak > 1.0:
        wet = wet / peak
    return wet.astype(np.float32)


def _background_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) == 0:
        return audio.astype(np.float32, copy=True)
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(audio)].astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(audio * audio))) + 1e-8
    noise_rms = float(np.sqrt(np.mean(noise * noise))) + 1e-8
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    mixed = audio + noise * (target_noise_rms / noise_rms)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def augment_manifest_records(records: Iterable[dict], *, output_dir: Path, plan: AugmentationPlan = AugmentationPlan()) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented: list[dict] = []
    noise_cache: dict[Path, tuple[np.ndarray, int]] = {}

    for record in records:
        source_path = Path(record["normalized_audio_path"])
        audio, sample_rate = load_wav_mono(source_path)
        if sample_rate != 16000:
            audio = _resample_linear(audio, sample_rate, 16000)
            sample_rate = 16000

        variants: list[tuple[str, dict, np.ndarray]] = []
        for factor in plan.speed_factors:
            variants.append(("speed", {"factor": float(factor)}, _speed(audio, float(factor))))
        for db in plan.gain_db:
            variants.append(("gain", {"db": float(db)}, _gain(audio, float(db))))
        for decay in plan.reverb_decays:
            variants.append(("reverb", {"decay": float(decay)}, _reverb(audio, float(decay))))
        for noise_path in plan.background_noise_paths:
            noise_path = Path(noise_path)
            if noise_path not in noise_cache:
                noise_cache[noise_path] = load_wav_mono(noise_path)
            noise_audio, noise_rate = noise_cache[noise_path]
            if noise_rate != 16000:
                noise_audio = _resample_linear(noise_audio, noise_rate, 16000)
            for snr in plan.background_noise_snr_db:
                variants.append((
                    "background_noise",
                    {"noise_path": str(noise_path), "snr_db": float(snr)},
                    _background_noise(audio, noise_audio, float(snr)),
                ))

        for kind, params, variant_audio in variants:
            aug_id = f"{record['sample_id']}-{kind}-{_stable_id(record['sample_id'], kind, params)}"
            output_path = output_dir / f"{aug_id}.wav"
            write_wav_mono16(output_path, variant_audio, 16000)
            augmented_record = dict(record)
            augmented_record.update({
                "sample_id": aug_id,
                "source_sample_id": record["sample_id"],
                "normalized_audio_path": str(output_path),
                "augmentation": {"kind": kind, **params},
            })
            augmented.append(augmented_record)
    return augmented


def count_by_split_label(records: Iterable[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[record.get("split", "unsplit")][record["label"]] += 1
    return {split: dict(labels) for split, labels in counts.items()}
