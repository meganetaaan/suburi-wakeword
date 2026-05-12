import json
import tempfile
import unittest
from pathlib import Path

from suburi_wakeword.run_smoke_pipeline import build_jobs_for_profile
from suburi_wakeword.tts import ProsodyVariant, VoiceProfile, build_infer_command, build_expanded_jobs, build_large_synthetic_jobs, build_smoke_jobs, load_voice_profiles, summarize_jobs


class TtsJobTests(unittest.TestCase):
    def test_build_smoke_jobs_starts_with_japanese_positive_variants(self):
        jobs = build_smoke_jobs(length_scales=(1.3,))
        positives = [job for job in jobs if job.label == "positive"]

        self.assertGreaterEqual(len(positives), 4)
        self.assertTrue(all(job.phrase_id == "hai_stackchan_ja" for job in positives))
        self.assertIn("ハイ、スタックチャン", {job.text for job in positives})
        self.assertNotIn("Hi, Stack-chan", {job.text for job in positives})

    def test_build_smoke_jobs_includes_hard_negatives(self):
        jobs = build_smoke_jobs(length_scales=(1.5,))
        negatives = [job.text for job in jobs if job.label == "negative"]

        self.assertIn("スタック", negatives)
        self.assertIn("ハイ、ロボットちゃん", negatives)

    def test_build_smoke_jobs_scales_positive_negative_and_holdout_samples(self):
        jobs = build_smoke_jobs(length_scales=(1.1, 1.2, 1.3, 1.4))

        self.assertEqual(len([job for job in jobs if job.label == "positive"]), 16)
        self.assertEqual(len([job for job in jobs if job.label == "negative"]), 24)
        self.assertEqual(len([job for job in jobs if job.label == "holdout"]), 4)

    def test_build_expanded_jobs_multiplies_phrases_voices_and_prosody(self):
        jobs = build_expanded_jobs(
            positive_texts=("ハイ、スタックチャン", "はい、スタックちゃん"),
            negative_texts=("スタック",),
            holdout_texts=("Hi, Stack-chan",),
            voices=(
                VoiceProfile(id="tsukuyomi", voice="ja_JP-tsukuyomi-chan-medium", speaker_id=0),
                VoiceProfile(id="alt", voice="ja_JP-alt-medium", speaker_id=1),
            ),
            prosody_variants=(
                ProsodyVariant(id="fast-flat", length_scale=0.9, noise_scale=0.55, noise_scale_w=0.6),
                ProsodyVariant(id="slow-lively", length_scale=1.4, noise_scale=0.8, noise_scale_w=1.1),
            ),
        )

        positives = [job for job in jobs if job.label == "positive"]
        negatives = [job for job in jobs if job.label == "negative"]
        holdouts = [job for job in jobs if job.label == "holdout"]
        self.assertEqual(len(positives), 2 * 2 * 2)
        self.assertEqual(len(negatives), 1 * 2 * 2)
        self.assertEqual(len(holdouts), 1 * 2 * 2)
        self.assertEqual(len({job.sample_id for job in jobs}), len(jobs))
        self.assertEqual({job.voice_profile_id for job in positives}, {"tsukuyomi", "alt"})
        self.assertEqual({job.prosody_id for job in positives}, {"fast-flat", "slow-lively"})
        self.assertNotIn("Hi, Stack-chan", {job.text for job in positives})
        self.assertTrue(all(job.language == "en" for job in holdouts))

    def test_build_infer_command_passes_prosody_and_speaker_parameters(self):
        job = build_expanded_jobs(
            positive_texts=("ハイ、スタックチャン",),
            negative_texts=(),
            holdout_texts=(),
            voices=(VoiceProfile(id="speaker1", voice="ja_JP-test", speaker_id=3),),
            prosody_variants=(ProsodyVariant(id="fast", length_scale=0.85, noise_scale=0.5, noise_scale_w=0.7),),
        )[0]

        command = build_infer_command(
            job,
            model_path=Path("/tmp/model.onnx"),
            config_path=Path("/tmp/config.json"),
            output_dir=Path("/tmp/out"),
        )

        self.assertIn("--noise-scale", command)
        self.assertEqual(command[command.index("--noise-scale") + 1], "0.5")
        self.assertIn("--noise-scale-w", command)
        self.assertEqual(command[command.index("--noise-scale-w") + 1], "0.7")
        self.assertEqual(command[command.index("--speaker-id") + 1], "3")

    def test_large_synthetic_jobs_are_much_larger_than_smoke_and_keep_metadata_axes(self):
        jobs = build_large_synthetic_jobs()
        smoke_jobs = build_smoke_jobs(length_scales=(1.3,))
        positives = [job for job in jobs if job.label == "positive"]

        self.assertGreaterEqual(len(jobs), len(smoke_jobs) * 5)
        self.assertGreaterEqual(len({job.text for job in positives}), 10)
        self.assertGreaterEqual(len({job.prosody_id for job in positives}), 5)
        self.assertEqual({job.voice_profile_id for job in jobs}, {"tsukuyomi"})
        self.assertNotIn("Hi, Stack-chan", {job.text for job in positives})

    def test_pipeline_profile_selects_expanded_dataset_plan(self):
        smoke_jobs = build_jobs_for_profile("smoke", samples_per_variant=1)
        expanded_jobs = build_jobs_for_profile("expanded", samples_per_variant=1)

        self.assertGreater(len(expanded_jobs), len(smoke_jobs) * 5)
        self.assertIn("prosody_id", expanded_jobs[0].__dataclass_fields__)
        with self.assertRaises(ValueError):
            build_jobs_for_profile("unknown", samples_per_variant=1)

    def test_expanded_5k_profile_reaches_target_after_augmentation(self):
        jobs = build_jobs_for_profile(
            "expanded-5k",
            samples_per_variant=1,
            voices=(
                VoiceProfile(id="voice-a", speaker_id=0),
                VoiceProfile(id="voice-b", speaker_id=1),
                VoiceProfile(id="voice-c", speaker_id=2),
                VoiceProfile(id="voice-d", speaker_id=3),
            ),
        )
        summary = summarize_jobs(jobs, augmentation_multiplier=6)

        self.assertEqual(summary["base_jobs"], 912)
        self.assertEqual(summary["estimated_after_augmentation"], 5472)
        self.assertEqual(summary["labels"], {"positive": 288, "negative": 576, "holdout": 48})
        self.assertEqual(summary["voice_profiles"], 4)
        self.assertEqual(summary["prosody_variants"], 6)
        self.assertGreater(summary["negative_base_jobs"], summary["positive_base_jobs"])

    def test_voice_profiles_can_be_loaded_from_local_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voices.json"
            path.write_text(json.dumps({
                "voices": [
                    {"id": "tsukuyomi", "speaker_id": 0, "model_id": "tsukuyomi-chan-6lang-fp16"},
                    {"id": "alt", "voice": "ja_JP-alt-medium", "speaker_id": 1, "source_engine": "piper-plus", "model_id": "alt-model"},
                ]
            }), encoding="utf-8")

            voices = load_voice_profiles(path)

        self.assertEqual([voice.id for voice in voices], ["tsukuyomi", "alt"])
        self.assertEqual(voices[0].voice, "ja_JP-tsukuyomi-chan-medium")
        self.assertEqual(voices[1].speaker_id, 1)
        self.assertEqual(voices[1].model_id, "alt-model")


if __name__ == "__main__":
    unittest.main()
