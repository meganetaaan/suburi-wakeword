import tempfile
import unittest
from pathlib import Path

from suburi_wakeword.real_eval_runner import build_training_step_experiments


class RealEvalRunnerTests(unittest.TestCase):
    def test_build_training_step_experiments_names_output_roots_by_training_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "samples3"
            root.mkdir()
            experiments = build_training_step_experiments(
                [("samples3", root)],
                training_steps=(2, 20),
            )

        self.assertEqual([experiment.name for experiment in experiments], ["samples3_steps2", "samples3_steps20"])
        self.assertEqual([experiment.training_steps for experiment in experiments], [(2,), (20,)])
        self.assertEqual(experiments[0].source_root, root)
        self.assertEqual(experiments[0].run_root, root / "artifacts" / "real-eval" / "steps2")
        self.assertEqual(experiments[1].run_root, root / "artifacts" / "real-eval" / "steps20")


if __name__ == "__main__":
    unittest.main()
