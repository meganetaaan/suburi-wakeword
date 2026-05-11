import unittest
from pathlib import Path

from suburi_wakeword.tts import TtsJob, build_infer_command


class PiperCommandTests(unittest.TestCase):
    def test_build_infer_command_uses_absolute_model_config_and_output_paths(self):
        job = TtsJob("sample", "hai_stackchan_ja", "positive", "ハイ、スタックチャン", 1.3)
        command = build_infer_command(
            job,
            model_path=Path("relative/model.onnx"),
            config_path=Path("relative/config.json"),
            output_dir=Path("relative/out"),
        )

        self.assertIn(str(Path("relative/model.onnx").resolve()), command)
        self.assertIn(str(Path("relative/config.json").resolve()), command)
        self.assertIn(str(Path("relative/out").resolve()), command)


if __name__ == "__main__":
    unittest.main()
