import unittest

import numpy as np

from suburi_wakeword.smoke_train import predict_label, train_centroid_model


class SmokeTrainTests(unittest.TestCase):
    def test_centroid_model_predicts_nearest_label(self):
        features = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 9.9],
        ], dtype=np.float32)
        labels = ["negative", "negative", "positive", "positive"]

        model = train_centroid_model(features, labels)

        self.assertEqual(predict_label(model, np.array([9.8, 10.2], dtype=np.float32)), "positive")
        self.assertEqual(predict_label(model, np.array([0.2, 0.0], dtype=np.float32)), "negative")


if __name__ == "__main__":
    unittest.main()
