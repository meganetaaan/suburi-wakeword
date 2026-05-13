from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from suburi_wakeword.voice_collection_server import (
    build_server,
    default_prompt_plan,
    load_prompt_plan,
    render_index_html,
    save_recording_upload,
)


class VoiceCollectionServerTests(unittest.TestCase):
    def test_default_prompt_plan_includes_positive_and_near_miss_negatives(self):
        plan = default_prompt_plan()

        labels = {prompt["label"] for prompt in plan["prompts"]}
        texts = {prompt["text"] for prompt in plan["prompts"]}

        self.assertEqual(plan["wake_phrase"], "ハイ、スタックちゃん")
        self.assertIn("positive", labels)
        self.assertIn("negative", labels)
        self.assertIn("ハイ、スタックちゃん", texts)
        self.assertIn("スタックちゃんと呼びました", texts)
        self.assertIn("スタッキーちゃん", texts)

    def test_load_prompt_plan_expands_repeats_with_stable_take_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            path.write_text(
                json.dumps(
                    {
                        "session_name": "demo",
                        "prompts": [
                            {"id": "p001", "label": "positive", "text": "ハイ、スタックちゃん", "repeat": 2},
                            {"id": "n001", "label": "negative", "text": "スタッキーちゃん"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            plan = load_prompt_plan(path)

        self.assertEqual([p["take_id"] for p in plan["prompts"]], ["p001-t01", "p001-t02", "n001-t01"])
        self.assertEqual([p["repeat_index"] for p in plan["prompts"]], [1, 2, 1])
        self.assertEqual(plan["session_name"], "demo")

    def test_build_server_serves_prompt_plan_with_configured_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = build_server(
                host="127.0.0.1",
                port=0,
                output_root=Path(tmp),
                prompt_plan=default_prompt_plan(),
            )
            try:
                self.assertEqual(server.RequestHandlerClass.output_root, Path(tmp))
                self.assertEqual(server.RequestHandlerClass.plan["wake_phrase"], "ハイ、スタックちゃん")
            finally:
                server.server_close()

    def test_default_prompt_plan_includes_consent_terms(self):
        plan = default_prompt_plan()

        consent = plan["consent"]

        self.assertIn("評価用音声", consent["title"])
        self.assertTrue(consent["required"])
        self.assertIn("学習には使わず", "\n".join(consent["items"]))
        self.assertIn("同意します", consent["checkbox_label"])

    def test_render_index_html_requires_terms_dialog_before_recording(self):
        html = render_index_html(default_prompt_plan())

        self.assertIn("consent-dialog", html)
        self.assertIn("評価用音声の収集について", html)
        self.assertIn("agree-consent", html)
        self.assertIn("同意してはじめる", html)
        self.assertIn("localStorage.getItem('voiceCollectionConsentAccepted')", html)

    def test_save_recording_upload_rejects_missing_consent_acceptance(self):
        payload = {
            "participant_id": "User 01 / demo",
            "prompt": {"id": "n001", "take_id": "n001-t01", "label": "negative", "text": "スタッキーちゃん"},
            "mime_type": "audio/webm;codecs=opus",
            "audio_base64": base64.b64encode(b"audio").decode("ascii"),
            "duration_ms": 1234,
        }

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "consent"):
                save_recording_upload(payload, output_root=Path(tmp), now="20260513T120001Z")

    def test_save_recording_upload_writes_audio_and_manifest_by_label_and_prompt(self):
        fake_audio = b"not really a wav but upload bytes"
        payload = {
            "participant_id": "User 01 / demo",
            "prompt": {"id": "n001", "take_id": "n001-t01", "label": "negative", "text": "スタッキーちゃん"},
            "mime_type": "audio/webm;codecs=opus",
            "audio_base64": base64.b64encode(fake_audio).decode("ascii"),
            "duration_ms": 1234,
            "client_started_at": "2026-05-13T12:00:00.000Z",
            "consent": {"accepted": True, "version": "real-voice-eval-v1", "accepted_at": "2026-05-13T12:00:00.000Z"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = save_recording_upload(payload, output_root=Path(tmp), now="20260513T120001Z")
            audio_path = Path(result["audio_path"])
            manifest_path = Path(result["manifest_path"])

            self.assertTrue(audio_path.exists())
            self.assertEqual(audio_path.read_bytes(), fake_audio)
            self.assertIn("negative/n001", audio_path.as_posix())
            self.assertEqual(audio_path.suffix, ".webm")

            row = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["participant_id"], "user-01-demo")
        self.assertEqual(row["label"], "negative")
        self.assertEqual(row["text"], "スタッキーちゃん")
        self.assertEqual(row["duration_ms"], 1234)
        self.assertEqual(row["audio_format"], "webm")
        self.assertEqual(row["consent"]["version"], "real-voice-eval-v1")
        self.assertTrue(row["consent"]["accepted"])


if __name__ == "__main__":
    unittest.main()
