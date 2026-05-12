import unittest
from suburi_wakeword.tts import build_smoke_jobs


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


if __name__ == "__main__":
    unittest.main()
