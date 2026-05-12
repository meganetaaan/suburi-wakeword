import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from suburi_wakeword.audio import load_wav_mono, write_wav_mono16
from suburi_wakeword.augment import AugmentationPlan, augment_manifest_records
from suburi_wakeword.dataset_split import assign_splits
from suburi_wakeword.microwakeword import (
    MicroWakeWordFeaturePlan,
    MicroWakeWordTrainingConfig,
    MicroWakeWordTrainingPlan,
    build_microwakeword_feature_plan,
    build_microwakeword_training_plan,
    train_microwakeword_model,
    train_microwakeword_smoke_model,
)
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

    def _write_split_manifest(self, root: Path) -> Path:
        audio_dir = root / "audio"
        audio_dir.mkdir()
        for name in ["p1", "n1", "p2", "n2", "h1"]:
            write_wav_mono16(audio_dir / f"{name}.wav", np.zeros(1600, dtype=np.float32), 16000)
        records = [
            {"sample_id": "p1", "label": "positive", "split": "train", "normalized_audio_path": str(audio_dir / "p1.wav")},
            {"sample_id": "n1", "label": "negative", "split": "train", "normalized_audio_path": str(audio_dir / "n1.wav")},
            {"sample_id": "p2", "label": "positive", "split": "validation", "normalized_audio_path": str(audio_dir / "p2.wav")},
            {"sample_id": "n2", "label": "negative", "split": "validation", "normalized_audio_path": str(audio_dir / "n2.wav")},
            {"sample_id": "h1", "label": "holdout", "split": "holdout", "normalized_audio_path": str(audio_dir / "h1.wav")},
        ]
        manifest = root / "dataset.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return manifest

    def test_microwakeword_feature_plan_stages_audio_and_writes_upstream_ragged_mmap_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_split_manifest(root)
            source_dir = root / "micro-wake-word"

            plan = build_microwakeword_feature_plan(manifest, root, source_dir=source_dir)

            self.assertIsInstance(plan, MicroWakeWordFeaturePlan)
            self.assertEqual(plan.command[:4], ["uv", "run", "--no-sync", "python"])
            self.assertEqual(plan.command[4], str(plan.script_path.resolve()))
            self.assertEqual(plan.cwd, source_dir)
            self.assertEqual(plan.positive_features_dir, root / "features" / "positive")
            self.assertEqual(plan.negative_features_dir, root / "features" / "negative")
            self.assertTrue((root / "features" / "positive" / "training" / "wav" / "p1.wav").exists())
            self.assertTrue((root / "features" / "positive" / "validation" / "wav" / "p2.wav").exists())
            self.assertTrue((root / "features" / "positive" / "testing" / "wav" / "p2.wav").exists())
            self.assertTrue((root / "features" / "negative" / "training" / "wav" / "n1.wav").exists())
            self.assertTrue((root / "features" / "negative" / "validation" / "wav" / "n2.wav").exists())
            self.assertTrue((root / "features" / "negative" / "testing" / "wav" / "h1.wav").exists())
            script = plan.script_path.read_text(encoding="utf-8")
            self.assertIn(str((root / "features" / "positive").resolve()), script)
            self.assertIn("from mmap_ninja.ragged import RaggedMmap", script)
            self.assertIn("SpectrogramGeneration", script)
            self.assertIn("wakeword_mmap", script)

    def test_microwakeword_feature_plan_accepts_manifest_paths_relative_to_current_working_directory(self):
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            audio = root / "nested" / "sample.wav"
            write_wav_mono16(audio, np.zeros(1600, dtype=np.float32), 16000)
            manifest = root / "dataset.jsonl"
            manifest.write_text(json.dumps({
                "sample_id": "p1",
                "label": "positive",
                "split": "train",
                "normalized_audio_path": str(audio),
            }) + "\n", encoding="utf-8")

            plan = build_microwakeword_feature_plan(manifest, root, source_dir=root / "micro-wake-word")

            self.assertTrue((plan.positive_features_dir / "training" / "wav" / "p1.wav").exists())
            self.assertEqual(plan.command[4], str(plan.script_path.resolve()))

    def test_microwakeword_training_plan_uses_generated_feature_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_split_manifest(root)
            source_dir = root / "micro-wake-word"
            config = MicroWakeWordTrainingConfig(wake_word="hai_stackchan", model_name="hai_stackchan_ja", probability_cutoff=0.55)

            plan = build_microwakeword_training_plan(manifest, root, config, source_dir=source_dir, model_architecture="mixednet")

            self.assertIsInstance(plan, MicroWakeWordTrainingPlan)
            self.assertEqual(plan.expected_tflite, root / "artifacts" / "microwakeword-model" / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite")
            self.assertEqual(plan.command[:5], ["uv", "run", "--no-sync", "python", "-m"])
            self.assertIn("microwakeword.model_train_eval", plan.command)
            self.assertIn("--test_tflite_streaming_quantized", plan.command)
            self.assertIn("mixednet", plan.command)
            self.assertIn(str(plan.training_config_path.resolve()), plan.command)
            training_config = json.loads(plan.training_config_path.read_text(encoding="utf-8"))
            self.assertEqual(training_config["train_dir"], str((root / "artifacts" / "microwakeword-model").resolve()))
            self.assertEqual(training_config["features"][0]["features_dir"], str((root / "features" / "positive").resolve()))
            self.assertEqual(training_config["features"][0]["truth"], True)
            self.assertEqual(training_config["features"][1]["features_dir"], str((root / "features" / "negative").resolve()))
            self.assertEqual(training_config["features"][1]["truth"], False)

    def test_microwakeword_commands_use_absolute_generated_paths_for_upstream_cwd(self):
        root = Path("tmp-mww-command-contract")
        if root.exists():
            shutil.rmtree(root)
        try:
            root.mkdir()
            manifest = self._write_split_manifest(root)
            source_dir = root / "micro-wake-word"

            feature_plan = build_microwakeword_feature_plan(manifest, root, source_dir=source_dir)
            training_plan = build_microwakeword_training_plan(manifest, root, MicroWakeWordTrainingConfig(), source_dir=source_dir)

            self.assertEqual(feature_plan.command[4], str(feature_plan.script_path.resolve()))
            self.assertIn(str(training_plan.training_config_path.resolve()), training_plan.command)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_microwakeword_model_handoff_rejects_placeholder_tflite_and_copies_real_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_split_manifest(root)
            upstream_output = root / "upstream" / "stream_state_internal_quant.tflite"
            upstream_output.parent.mkdir(parents=True)
            upstream_output.write_bytes(b"TFL3-real-microwakeword-output")

            artifact_dir = train_microwakeword_model(
                manifest,
                root,
                MicroWakeWordTrainingConfig(wake_word="hai_stackchan", model_name="hai_stackchan_ja", probability_cutoff=0.55),
                prebuilt_tflite=upstream_output,
            )

            tflite = artifact_dir / "model" / "stream_state_internal_quant.tflite"
            mww_manifest = artifact_dir / "model" / "hai_stackchan_ja.json"
            self.assertEqual(tflite.read_bytes(), b"TFL3-real-microwakeword-output")
            manifest_json = json.loads(mww_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_json["training_backend"], "microWakeWord")
            self.assertEqual(manifest_json["trained_sample_counts"]["train"]["positive"], 1)

            placeholder = root / "placeholder.tflite"
            placeholder.write_bytes(b'TFL3{"format":"suburi-microwakeword-smoke-tflite-placeholder"}')
            with self.assertRaisesRegex(ValueError, "placeholder"):
                train_microwakeword_model(manifest, root, MicroWakeWordTrainingConfig(), prebuilt_tflite=placeholder)


if __name__ == "__main__":
    unittest.main()
