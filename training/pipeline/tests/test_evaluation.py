import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from suburi_wakeword.audio import write_wav_mono16
from suburi_wakeword.evaluation import cross_validate_manifest, compare_training_scales


class EvaluationTests(unittest.TestCase):
    def _write_dataset(self, root: Path, *, positives: int = 6, negatives: int = 6) -> Path:
        records = []
        sample_rate = 16000
        t = np.linspace(0.0, 0.5, int(sample_rate * 0.5), endpoint=False)
        for i in range(positives):
            audio = 0.4 * np.sin(2 * np.pi * (440 + i * 4) * t).astype(np.float32)
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
                "label": "negative",
                "split": "training" if i % 2 == 0 else "holdout",
                "normalized_audio_path": str(path),
            })
        manifest = root / "dataset.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return manifest

    def test_cross_validate_manifest_reports_fold_and_aggregate_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._write_dataset(Path(tmp))

            report = cross_validate_manifest(manifest, folds=3)

        self.assertEqual(report["folds"], 3)
        self.assertEqual(report["sample_count"], 12)
        self.assertEqual(report["positive_samples"], 6)
        self.assertEqual(report["negative_samples"], 6)
        self.assertEqual(len(report["fold_metrics"]), 3)
        self.assertGreaterEqual(report["accuracy_mean"], 0.9)
        self.assertEqual(report["false_accepts_total"], 0)
        self.assertEqual(report["false_rejects_total"], 0)

    def test_compare_training_scales_keeps_results_in_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = self._write_dataset(root / "small", positives=4, negatives=4)
            large = self._write_dataset(root / "large", positives=8, negatives=8)

            rows = compare_training_scales([
                ("small", small),
                ("large", large),
            ], folds=2)

        self.assertEqual([row["scale"] for row in rows], ["small", "large"])
        self.assertEqual([row["sample_count"] for row in rows], [8, 16])
        self.assertTrue(all("accuracy_mean" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
