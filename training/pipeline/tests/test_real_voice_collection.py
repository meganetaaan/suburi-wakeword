from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from suburi_wakeword.real_voice_collection import (
    build_collection_eval_manifest,
    build_ffmpeg_decode_command,
    summarize_collection_manifest,
)


class RealVoiceCollectionTests(unittest.TestCase):
    def test_build_ffmpeg_decode_command_writes_16k_mono_pcm_wav(self):
        command = build_ffmpeg_decode_command(Path("input.webm"), Path("out.wav"))

        self.assertEqual(command[:2], ["ffmpeg", "-y"])
        self.assertIn("input.webm", command)
        self.assertIn("out.wav", command)
        self.assertIn("-ac", command)
        self.assertIn("1", command)
        self.assertIn("-ar", command)
        self.assertIn("16000", command)
        self.assertIn("pcm_s16le", command)

    def test_summarize_collection_manifest_counts_labels_texts_and_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "collection/session/positive/p001/take.webm"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"webm")
            manifest = root / "collection/session/manifest.jsonl"
            rows = [
                {"participant_id": "s1", "label": "positive", "text": "ハイ、スタックちゃん", "audio_format": "webm", "audio_path": str(audio)},
                {"participant_id": "s1", "label": "negative", "text": "スタッキーちゃん", "audio_format": "webm", "audio_path": "missing.webm"},
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

            summary = summarize_collection_manifest(manifest)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["participants"], ["s1"])
        self.assertEqual(summary["labels"], {"positive": 1, "negative": 1})
        self.assertEqual(summary["texts"]["ハイ、スタックちゃん"], 1)
        self.assertEqual(summary["formats"], {"webm": 2})
        self.assertEqual(summary["missing_audio_count"], 1)

    def test_build_collection_eval_manifest_converts_records_to_normalized_eval_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "collection/session/positive/p001/take.webm"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"webm")
            manifest = root / "collection/session/manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "recorded_at": "20260513T120001Z",
                        "participant_id": "s1",
                        "prompt_id": "p001",
                        "take_id": "p001-t01",
                        "label": "positive",
                        "text": "ハイ、スタックちゃん",
                        "audio_path": str(source),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_manifest = root / "eval/dataset.jsonl"
            calls: list[tuple[Path, Path]] = []

            result = build_collection_eval_manifest(
                manifest,
                output_manifest,
                decode_audio=lambda src, dst: calls.append((src, dst)) or dst.write_bytes(b"wav"),
            )

            rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows[0]["sample_id"], "real_s1_p001-t01")
        self.assertEqual(rows[0]["split"], "real_holdout")
        self.assertEqual(rows[0]["source_text"], "ハイ、スタックちゃん")
        self.assertEqual(rows[0]["normalized_audio_path"], str(calls[0][1]))
        self.assertEqual(rows[0]["audio_path"], str(calls[0][1]))


if __name__ == "__main__":
    unittest.main()
