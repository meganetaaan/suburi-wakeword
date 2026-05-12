import unittest

from suburi_wakeword.false_accept_analysis import summarize_false_accepts_by_text


class FalseAcceptAnalysisTests(unittest.TestCase):
    def test_summarizes_false_accepts_by_source_text(self):
        records = [
            {"sample_id": "neg-a", "label": "negative", "text": "スタックチャンネル"},
            {"sample_id": "neg-a-reverb", "source_sample_id": "neg-a", "label": "negative", "text": "スタックチャンネル", "augmentation": {"name": "reverb"}},
            {"sample_id": "neg-b", "label": "negative", "text": "スタック"},
            {"sample_id": "pos-a", "label": "positive", "text": "ハイ、スタックチャン"},
        ]
        sample_scores = [
            {"sample_id": "neg-a", "score": 0.99},
            {"sample_id": "neg-a-reverb", "score": 0.97},
            {"sample_id": "neg-b", "score": 0.40},
            {"sample_id": "pos-a", "score": 0.30},
        ]

        rows = summarize_false_accepts_by_text(records, sample_scores, threshold=0.965)

        self.assertEqual(rows[0]["text"], "スタックチャンネル")
        self.assertEqual(rows[0]["false_accepts"], 2)
        self.assertEqual(rows[0]["sample_count"], 2)
        self.assertEqual(rows[0]["false_accept_rate"], 1.0)
        self.assertEqual(rows[0]["augmentation_counts"], {"raw": 1, "reverb": 1})
        self.assertEqual(rows[1]["text"], "スタック")
        self.assertEqual(rows[1]["false_accepts"], 0)

    def test_treats_holdout_as_negative_for_far_analysis(self):
        records = [
            {"sample_id": "holdout-a", "label": "holdout", "text": "Hi, Stack-chan"},
            {"sample_id": "pos-a", "label": "positive", "text": "ハイ、スタックチャン"},
        ]
        sample_scores = [
            {"sample_id": "holdout-a", "score": 0.99},
            {"sample_id": "pos-a", "score": 0.99},
        ]

        rows = summarize_false_accepts_by_text(records, sample_scores, threshold=0.965)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "Hi, Stack-chan")
        self.assertEqual(rows[0]["labels"], {"holdout": 1})


if __name__ == "__main__":
    unittest.main()
