from __future__ import annotations

import numpy as np


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return np.zeros(8, dtype=np.float32)
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.int8))))) if len(audio) > 1 else 0.0
    window = audio * np.hanning(len(audio)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    energy = float(np.sum(spectrum) + 1e-8)
    centroid = float(np.sum(freqs * spectrum) / energy)
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, cumulative[-1] * 0.85)) if cumulative[-1] > 0 else 0
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    low = float(np.sum(spectrum[freqs < 500]) / energy)
    mid = float(np.sum(spectrum[(freqs >= 500) & (freqs < 2500)]) / energy)
    high = float(np.sum(spectrum[freqs >= 2500]) / energy)
    return np.array([duration, rms, peak, zcr, centroid / sample_rate, rolloff / sample_rate, low, mid + high], dtype=np.float32)
