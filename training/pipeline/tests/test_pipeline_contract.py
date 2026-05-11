import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from suburi_wakeword.audio import load_wav_mono, write_wav_mono16
from suburi_wakeword.augment import AugmentationPlan, augment_manifest_records
from suburi_wakeword.dataset_split import assign_splits
from suburi_wakeword.microwakeword import MicroWakeWordTrainingConfig, train_microwakeword_smoke_model
from suburi_wakeword.threshold_sweep import sweep_thresholds


class PipelineContractTests(unittest.TestCase):
    def test_augmentation_writes_speed_gain_reverb_and_background_noise_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "normalized" / "positive-a.wav"
            noise = root / "noise.wav"
            sample_rate = 16000
            t = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
            write_wav_mono16(src, 0.2 * np.sin(2 * math.pi * 440 * t), sample_rate)
            write_wav_mono16(noise, 0.05 * np.sin(2 * math.pi * 110 * t), sample_rate)
            records = [{"sample_id": "positive-a", "label": "positive", "normalized_audio_path": str(src)}]

            augmented = augment_manifest_records(
                records,
                output_dir=root / "augmented",
                plan=AugmentationPlan(speed_factors=(0.9,), gain_db=(3.0,), reverb_decays=(0.2,), background_noise_paths=(noise,)),
            )

            kinds = {record["augmentation"]["kind"] for record in augmented}
            self.assertEqual(kinds, {"speed", "gain", "reverb", "background_noise"})
            for record in augmented:
                audio_path = Path(record["normalized_audio_path"])
                self.assertTrue(audio_path.exists())
                _, sr = load_wav_mono(audio_path)
                self.assertEqual(sr, 16000)
                self.assertEqual(record["source_sample_id"], "positive-a")

    def test_assign_splits_keeps_positive_negative_and_holdout_separate(self):
        records = [
            {"sample_id": "p1", "label": "positive", "text": "ハイ、スタックチャン"},
            {"sample_id": "p2", "label": "positive", "text": "はい、スタックチャン"},
            {"sample_id": "n1", "label": "negative", "text": "スタック"},
            {"sample_id": "n2", "label": "negative", "text": "ちゃん"},
            {"sample_id": "h1", "label": "holdout", "text": "Hi, Stack-chan"},
        ]

        split_records = assign_splits(records, holdout_labels={"holdout"}, seed="fixed")

        by_id = {record["sample_id"]: record for record in split_records}
        self.assertEqual(by_id["h1"]["split"], "holdout")
        self.assertEqual({record["split"] for record in split_records if record["label"] == "positive"}, {"train", "validation"})
        self.assertEqual({record["split"] for record in split_records if record["label"] == "negative"}, {"train", "validation"})

    def test_threshold_sweep_reports_frr_and_far_per_threshold(self):
        rows = sweep_thresholds(
            [{"label": "positive", "score": 0.9}, {"label": "positive", "score": 0.4}, {"label": "negative", "score": 0.6}],
            thresholds=(0.5, 0.8),
        )

        self.assertEqual(rows[0]["threshold"], 0.5)
        self.assertEqual(rows[0]["false_rejects"], 1)
        self.assertEqual(rows[0]["false_accepts"], 1)
        self.assertAlmostEqual(rows[1]["frr"], 0.5)
        self.assertAlmostEqual(rows[1]["far_per_sample"], 0.0)

    def test_microwakeword_smoke_training_exports_tflite_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"sample_id": "p1", "label": "positive", "split": "train", "normalized_audio_path": "p1.wav"},
                {"sample_id": "n1", "label": "negative", "split": "train", "normalized_audio_path": "n1.wav"},
                {"sample_id": "p2", "label": "positive", "split": "validation", "normalized_audio_path": "p2.wav"},
                {"sample_id": "n2", "label": "negative", "split": "validation", "normalized_audio_path": "n2.wav"},
            ]
            manifest = root / "dataset.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            artifact_dir = train_microwakeword_smoke_model(
                manifest,
                root,
                MicroWakeWordTrainingConfig(wake_word="hai_stackchan", model_name="hai_stackchan_ja", probability_cutoff=0.55),
            )

            tflite = artifact_dir / "model" / "stream_state_internal_quant.tflite"
            mww_manifest = artifact_dir / "model" / "hai_stackchan_ja.json"
            self.assertTrue(tflite.exists())
            manifest_json = json.loads(mww_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_json["type"], "micro")
            self.assertEqual(manifest_json["wake_word"], "hai_stackchan")
            self.assertEqual(manifest_json["model"], "stream_state_internal_quant.tflite")
            self.assertEqual(manifest_json["trained_sample_counts"]["train"]["positive"], 1)


if __name__ == "__main__":
    unittest.main()
