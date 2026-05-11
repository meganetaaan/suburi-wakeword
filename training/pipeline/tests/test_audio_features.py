import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from suburi_wakeword.audio import load_wav_mono, normalize_to_wav16k
from suburi_wakeword.features import extract_features


class AudioFeatureTests(unittest.TestCase):
    def test_normalize_to_wav16k_outputs_mono_16khz(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.wav"
            dst = Path(tmp) / "dst.wav"
            samples = (0.2 * np.sin(2 * math.pi * 440 * np.arange(2205) / 22050) * 32767).astype("<i2")
            with wave.open(str(src), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(samples.tobytes())

            normalize_to_wav16k(src, dst)
            audio, sample_rate = load_wav_mono(dst)

            self.assertEqual(sample_rate, 16000)
            self.assertEqual(audio.ndim, 1)
            self.assertGreater(len(audio), 1000)
            self.assertLessEqual(float(np.max(np.abs(audio))), 1.0)

    def test_extract_features_returns_finite_vector(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[1000:4000] = 0.25

        features = extract_features(audio, sample_rate=16000)

        self.assertEqual(features.shape, (8,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[1], 0.0)  # RMS


if __name__ == "__main__":
    unittest.main()
