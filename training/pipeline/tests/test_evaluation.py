import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from suburi_wakeword.audio import write_wav_mono16
from suburi_wakeword.evaluation import evaluate_microwakeword_manifest, compare_microwakeword_manifests


class FakeMicroWakeWordModel:
    def __init__(self, model_path: str, stride: int | None = None):
        self.model_path = Path(model_path)
        self.stride = stride

    def predict_clip(self, data: np.ndarray, step_ms: int = 20):
        # Deliberately model-shaped: tests exercise microWakeWord's Model API
        # contract, not handcrafted feature/proxy classifiers.
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        return [peak]


class EvaluationTests(unittest.TestCase):
    def _write_dataset(self, root: Path, *, positives: int = 6, negatives: int = 6) -> Path:
        records = []
        sample_rate = 16000
        t = np.linspace(0.0, 0.5, int(sample_rate * 0.5), endpoint=False)
        for i in range(positives):
            audio = 0.8 * np.sin(2 * np.pi * (440 + i * 4) * t).astype(np.float32)
            path = root / "wav" / f"positive-{i}.wav"
            write_wav_mono16(path, audio, sample_rate)
            records.append({
                "sample_id": f"positive-{i}",
                "label": "positive",
                "split": "training" if i % 2 == 0 else "validation",
                "normalized_audio_path": str(path),
            })
        for i in range(negatives):
            audio = 0.2 * np.sin(2 * np.pi * (1600 + i * 6) * t).astype(np.float32)
            path = root / "wav" / f"negative-{i}.wav"
            write_wav_mono16(path, audio, sample_rate)
            records.append({
                "sample_id": f"negative-{i}",
                "label": "negative" if i % 2 == 0 else "holdout",
                "split": "training" if i % 2 == 0 else "holdout",
                "normalized_audio_path": str(path),
            })
        manifest = root / "dataset.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return manifest

    def test_evaluate_microwakeword_manifest_uses_model_scores_not_proxy_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_dataset(root)
            model_path = root / "stream_state_internal_quant.tflite"
            model_path.write_bytes(b"TFL3-real-test-model")

            report = evaluate_microwakeword_manifest(
                manifest,
                model_path,
                threshold=0.5,
                thresholds=(0.25, 0.5, 0.75),
                splits=("validation", "holdout"),
                model_factory=FakeMicroWakeWordModel,
            )

        self.assertEqual(report["evaluation"], "real_microwakeword_tflite_streaming")
        self.assertEqual(report["model"], str(model_path))
        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["positive_samples"], 3)
        self.assertEqual(report["negative_samples"], 3)
        self.assertEqual(report["evaluated_splits"], ["validation", "holdout"])
        self.assertEqual(report["threshold"], 0.5)
        self.assertEqual(report["false_accepts_total"], 0)
        self.assertEqual(report["false_rejects_total"], 0)
        self.assertEqual([row["threshold"] for row in report["threshold_sweep"]], [0.25, 0.5, 0.75])
        self.assertNotIn("proxy", json.dumps(report).lower())
        self.assertNotIn("centroid", json.dumps(report).lower())

    def test_compare_microwakeword_manifests_keeps_results_in_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = self._write_dataset(root / "small", positives=4, negatives=4)
            large = self._write_dataset(root / "large", positives=8, negatives=8)
            model_path = root / "stream_state_internal_quant.tflite"
            model_path.write_bytes(b"TFL3-real-test-model")

            rows = compare_microwakeword_manifests([
                ("small", small),
                ("large", large),
            ], model_path, threshold=0.5, model_factory=FakeMicroWakeWordModel)

        self.assertEqual([row["scale"] for row in rows], ["small", "large"])
        self.assertEqual([row["sample_count"] for row in rows], [8, 16])
        self.assertTrue(all(row["evaluation"] == "real_microwakeword_tflite_streaming" for row in rows))


if __name__ == "__main__":
    unittest.main()
