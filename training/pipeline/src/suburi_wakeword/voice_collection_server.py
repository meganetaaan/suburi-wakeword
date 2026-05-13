from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_OUTPUT_ROOT = Path("data/real-voice/collection")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_LABELS = {"positive", "negative", "holdout", "ambient"}
_MIME_EXTENSIONS = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
}


def _slug(value: str, *, fallback: str = "unknown") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip("-._")
    return value or fallback


def default_consent_terms() -> dict[str, Any]:
    return {
        "version": "real-voice-eval-v1",
        "required": True,
        "title": "評価用音声の収集について",
        "items": [
            "このページでは、画面に表示される短い文を読み上げた音声を録音します。",
            "録音データは、日本語ウェイクワード「ハイ、スタックちゃん」の評価用データとして保存します。",
            "まずは評価用として扱い、明示的な追加確認なしに学習には使わず、公開リポジトリにもコミットしません。",
            "参加者IDには本名ではなく speaker-01 のような仮名を使ってください。",
            "録音をやめたい場合は、保存せずにページを閉じれば送信されません。保存後の削除依頼もできます。",
        ],
        "checkbox_label": "上記を確認し、評価用音声として録音・保存することに同意します。",
    }


def default_prompt_plan() -> dict[str, Any]:
    return {
        "session_name": "hai-stackchan-real-eval-v1",
        "wake_phrase": "ハイ、スタックちゃん",
        "instructions": "静かな場所で、画面の文を自然に1回読み上げてください。録音前後に少し間を置くと評価しやすくなります。",
        "consent": default_consent_terms(),
        "prompts": [
            {"id": "p001", "label": "positive", "text": "ハイ、スタックちゃん", "repeat": 4},
            {"id": "p002", "label": "positive", "text": "はい、スタックちゃん", "repeat": 4},
            {"id": "p003", "label": "positive", "text": "ハイスタックちゃん", "repeat": 3},
            {"id": "p004", "label": "positive", "text": "ハイ、ｽﾀｯｸﾁｬﾝ", "repeat": 3},
            {"id": "n001", "label": "negative", "text": "スタックちゃんと呼びました", "repeat": 3},
            {"id": "n002", "label": "negative", "text": "スタックちゃん、こんにちは", "repeat": 3},
            {"id": "n003", "label": "negative", "text": "はい、スタックチャンネル", "repeat": 3},
            {"id": "n004", "label": "negative", "text": "ハイ、スタックチャンネル", "repeat": 3},
            {"id": "n005", "label": "negative", "text": "スタッキーちゃん", "repeat": 2},
            {"id": "n006", "label": "negative", "text": "スタッフさん", "repeat": 2},
            {"id": "n007", "label": "negative", "text": "スタックあんちゃん", "repeat": 2},
            {"id": "n008", "label": "negative", "text": "ねえ、スタックチャンネルを開いて", "repeat": 2},
            {"id": "h001", "label": "holdout", "text": "Hi, Stack-chan", "repeat": 2},
            {"id": "a001", "label": "ambient", "text": "何も読まずに、2秒ほど周囲の音だけを録音してください", "repeat": 2},
        ],
    }


def _expand_prompts(plan: dict[str, Any]) -> dict[str, Any]:
    expanded: list[dict[str, Any]] = []
    for index, prompt in enumerate(plan.get("prompts", []), start=1):
        prompt_id = str(prompt.get("id") or f"prompt-{index:03d}")
        label = str(prompt.get("label", "negative"))
        if label not in _LABELS:
            raise ValueError(f"Unsupported prompt label for {prompt_id}: {label}")
        text = str(prompt.get("text", "")).strip()
        if not text:
            raise ValueError(f"Prompt text is required for {prompt_id}")
        repeat = max(1, int(prompt.get("repeat", 1)))
        for repeat_index in range(1, repeat + 1):
            row = dict(prompt)
            row.update(
                {
                    "id": prompt_id,
                    "take_id": f"{prompt_id}-t{repeat_index:02d}",
                    "label": label,
                    "text": text,
                    "repeat_index": repeat_index,
                    "repeat_total": repeat,
                }
            )
            expanded.append(row)
    result = dict(plan)
    result["prompts"] = expanded
    return result


def load_prompt_plan(path: Path | None) -> dict[str, Any]:
    plan = default_prompt_plan() if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    plan.setdefault("session_name", "hai-stackchan-real-eval-v1")
    plan.setdefault("wake_phrase", "ハイ、スタックちゃん")
    plan.setdefault("instructions", default_prompt_plan()["instructions"])
    plan.setdefault("consent", default_consent_terms())
    return _expand_prompts(plan)


def _extension_for_mime(mime_type: str) -> str:
    media_type = mime_type.split(";", 1)[0].strip().lower()
    return _MIME_EXTENSIONS.get(media_type, "bin")


def _decode_audio_base64(value: str) -> bytes:
    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=True)


def save_recording_upload(payload: dict[str, Any], *, output_root: Path, now: str | None = None) -> dict[str, Any]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("payload.prompt is required")
    label = str(prompt.get("label", ""))
    if label not in _LABELS:
        raise ValueError(f"Unsupported label: {label}")
    prompt_id = _slug(str(prompt.get("id") or prompt.get("take_id") or "prompt"), fallback="prompt")
    take_id = _slug(str(prompt.get("take_id") or f"{prompt_id}-t01"), fallback=f"{prompt_id}-t01")
    participant_id = _slug(str(payload.get("participant_id") or "anonymous"), fallback="anonymous")
    consent = payload.get("consent")
    if not isinstance(consent, dict) or consent.get("accepted") is not True:
        raise ValueError("consent.accepted is required before saving recordings")
    consent_record = {
        "accepted": True,
        "version": str(consent.get("version") or "unknown"),
        "accepted_at": consent.get("accepted_at"),
    }
    mime_type = str(payload.get("mime_type") or "application/octet-stream")
    extension = _extension_for_mime(mime_type)
    audio = _decode_audio_base64(str(payload.get("audio_base64") or ""))
    if not audio:
        raise ValueError("audio_base64 is empty")
    if len(audio) > MAX_UPLOAD_BYTES:
        raise ValueError(f"audio upload too large: {len(audio)} bytes")

    timestamp = now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_name = _slug(str(payload.get("session_name") or "default-session"), fallback="default-session")
    directory = Path(output_root) / session_name / label / prompt_id
    directory.mkdir(parents=True, exist_ok=True)
    audio_path = directory / f"{timestamp}_{participant_id}_{take_id}.{extension}"
    audio_path.write_bytes(audio)

    manifest_path = Path(output_root) / session_name / "manifest.jsonl"
    record = {
        "recorded_at": timestamp,
        "session_name": session_name,
        "participant_id": participant_id,
        "prompt_id": prompt_id,
        "take_id": take_id,
        "label": label,
        "text": str(prompt.get("text") or ""),
        "duration_ms": payload.get("duration_ms"),
        "mime_type": mime_type,
        "audio_format": extension,
        "audio_bytes": len(audio),
        "audio_path": str(audio_path),
        "client_started_at": payload.get("client_started_at"),
        "user_agent": payload.get("user_agent"),
        "consent": consent_record,
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True, "audio_path": str(audio_path), "manifest_path": str(manifest_path), "record": record}


def render_index_html(plan: dict[str, Any]) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False)
    title = html.escape(str(plan.get("session_name", "Voice collection")))
    instructions = html.escape(str(plan.get("instructions", "")))
    return f"""<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title} voice collection</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    body {{ margin: 0; background: #10131a; color: #f7f2e8; }}
    main {{ max-width: 880px; margin: 0 auto; padding: 24px; }}
    .card {{ background: #1b2030; border: 1px solid #31384d; border-radius: 20px; padding: 20px; box-shadow: 0 12px 40px #0005; }}
    .prompt {{ font-size: clamp(2rem, 8vw, 4.8rem); line-height: 1.12; margin: 28px 0; letter-spacing: .03em; }}
    .meta, .status {{ color: #b8c0d9; }}
    input, button {{ font: inherit; border-radius: 12px; border: 1px solid #515b78; padding: 12px 14px; }}
    input {{ width: min(100%, 380px); background: #0d111a; color: #f7f2e8; }}
    button {{ cursor: pointer; background: #eef2ff; color: #121521; font-weight: 700; }}
    button.secondary {{ background: #232a3d; color: #f7f2e8; }}
    button.danger {{ background: #ffccd2; color: #351016; }}
    button:disabled {{ opacity: .45; cursor: not-allowed; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 4px 10px; background: #2b3349; color: #cbd5ff; }}
    .positive {{ background: #12492f; color: #b7ffd8; }}
    .negative {{ background: #4e2631; color: #ffd1dc; }}
    .holdout {{ background: #473a16; color: #ffe5a0; }}
    .ambient {{ background: #25384e; color: #c7e2ff; }}
    audio {{ width: 100%; margin-top: 12px; }}
    ol {{ padding-left: 1.4rem; }}
    dialog {{ border: 1px solid #515b78; border-radius: 20px; background: #1b2030; color: #f7f2e8; max-width: min(720px, calc(100vw - 32px)); padding: 0; box-shadow: 0 20px 70px #0009; }}
    dialog::backdrop {{ background: #000b; }}
    .dialog-body {{ padding: 22px; }}
    .consent-list {{ margin: 16px 0; }}
    .check-row {{ display: flex; gap: 10px; align-items: flex-start; margin: 16px 0; }}
    .check-row input {{ width: auto; margin-top: .25rem; }}
  </style>
</head>
<body>
<dialog id="consent-dialog">
  <form method="dialog" class="dialog-body">
    <h2 id="consent-title"></h2>
    <p class="meta">録音を始める前に、以下を確認してください。</p>
    <ul id="consent-items" class="consent-list"></ul>
    <label class="check-row"><input id="agree-consent" type="checkbox"><span id="consent-checkbox-label"></span></label>
    <div class="row">
      <button id="accept-consent" value="accept" disabled>同意してはじめる</button>
    </div>
  </form>
</dialog>
<main>
  <h1>ｽﾀｯｸﾁｬﾝ wake word real voice collection</h1>
  <p class=\"meta\">{instructions}</p>
  <section class=\"card\">
    <label>参加者ID（名前でなくてOK）<br><input id=\"participant\" autocomplete=\"off\" placeholder=\"例: speaker-01\"></label>
    <p class=\"meta\">録音はこのサーバの <code>data/real-voice/collection/</code> に保存されます。共有前に同意を取ってください。</p>
  </section>
  <section class=\"card\" style=\"margin-top: 18px;\">
    <div class=\"row\"><span id=\"progress\" class=\"pill\"></span><span id=\"label\" class=\"pill\"></span></div>
    <div id=\"prompt\" class=\"prompt\"></div>
    <div class=\"row\">
      <button id=\"record\">録音開始</button>
      <button id=\"stop\" class=\"danger\" disabled>停止</button>
      <button id=\"upload\" disabled>保存して次へ</button>
      <button id=\"skip\" class=\"secondary\">スキップ</button>
      <button id=\"prev\" class=\"secondary\">戻る</button>
    </div>
    <audio id=\"preview\" controls></audio>
    <p id=\"status\" class=\"status\"></p>
  </section>
  <section class=\"card\" style=\"margin-top: 18px;\">
    <h2>手順</h2>
    <ol>
      <li>参加者IDを入れる</li>
      <li>表示された文を自然に1回読む</li>
      <li>再生確認して「保存して次へ」</li>
    </ol>
  </section>
</main>
<script>
const PLAN = {plan_json};
let index = Number(localStorage.getItem('voiceCollectionIndex') || 0);
let recorder = null;
let chunks = [];
let lastBlob = null;
let startedAt = null;
let consentAcceptedAt = localStorage.getItem('voiceCollectionConsentAcceptedAt') || null;

const $ = (id) => document.getElementById(id);
$('participant').value = localStorage.getItem('voiceCollectionParticipant') || '';
$('participant').addEventListener('input', () => localStorage.setItem('voiceCollectionParticipant', $('participant').value));

function hasConsent() {{ return localStorage.getItem('voiceCollectionConsentAccepted') === PLAN.consent.version; }}
function openConsentIfNeeded() {{
  const consent = PLAN.consent || {{ required: false }};
  if (!consent.required || hasConsent()) return;
  $('consent-title').textContent = consent.title || '評価用音声の収集について';
  $('consent-items').innerHTML = '';
  for (const item of consent.items || []) {{
    const li = document.createElement('li');
    li.textContent = item;
    $('consent-items').appendChild(li);
  }}
  $('consent-checkbox-label').textContent = consent.checkbox_label || '同意します';
  $('agree-consent').checked = false;
  $('accept-consent').disabled = true;
  $('consent-dialog').showModal();
}}
$('agree-consent').addEventListener('change', () => {{ $('accept-consent').disabled = !$('agree-consent').checked; }});
$('accept-consent').addEventListener('click', (event) => {{
  if (!$('agree-consent').checked) {{ event.preventDefault(); return; }}
  consentAcceptedAt = new Date().toISOString();
  localStorage.setItem('voiceCollectionConsentAccepted', PLAN.consent.version);
  localStorage.setItem('voiceCollectionConsentAcceptedAt', consentAcceptedAt);
}});

function currentPrompt() {{ return PLAN.prompts[Math.max(0, Math.min(index, PLAN.prompts.length - 1))]; }}
function render() {{
  const p = currentPrompt();
  $('progress').textContent = `${{index + 1}} / ${{PLAN.prompts.length}}`;
  $('label').textContent = p.label;
  $('label').className = `pill ${{p.label}}`;
  $('prompt').textContent = p.text;
  $('status').textContent = `take: ${{p.take_id}}`;
  $('upload').disabled = !lastBlob;
  $('prev').disabled = index === 0;
  localStorage.setItem('voiceCollectionIndex', String(index));
}}
function next() {{ index = Math.min(index + 1, PLAN.prompts.length - 1); lastBlob = null; $('preview').removeAttribute('src'); render(); }}

$('record').onclick = async () => {{
  if (!hasConsent()) {{ openConsentIfNeeded(); return; }}
  if (!$('participant').value.trim()) {{ alert('参加者IDを入力してください'); return; }}
  const stream = await navigator.mediaDevices.getUserMedia({{ audio: {{ echoCancellation: false, noiseSuppression: false, autoGainControl: false }} }});
  chunks = [];
  startedAt = new Date();
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
  recorder = new MediaRecorder(stream, {{ mimeType: mime }});
  recorder.ondataavailable = (event) => {{ if (event.data.size) chunks.push(event.data); }};
  recorder.onstop = () => {{
    lastBlob = new Blob(chunks, {{ type: recorder.mimeType }});
    $('preview').src = URL.createObjectURL(lastBlob);
    $('upload').disabled = false;
    $('record').disabled = false;
    $('stop').disabled = true;
    stream.getTracks().forEach(track => track.stop());
    $('status').textContent = `録音しました (${{Math.round(lastBlob.size / 1024)}} KB)。確認して保存してください。`;
  }};
  recorder.start();
  $('record').disabled = true;
  $('stop').disabled = false;
  $('upload').disabled = true;
  $('status').textContent = '録音中…';
}};

$('stop').onclick = () => {{ if (recorder && recorder.state !== 'inactive') recorder.stop(); }};
$('skip').onclick = next;
$('prev').onclick = () => {{ index = Math.max(0, index - 1); lastBlob = null; $('preview').removeAttribute('src'); render(); }};
$('upload').onclick = async () => {{
  if (!hasConsent()) {{ openConsentIfNeeded(); return; }}
  const p = currentPrompt();
  const reader = new FileReader();
  reader.onloadend = async () => {{
    $('status').textContent = '保存中…';
    const res = await fetch('/api/upload', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        session_name: PLAN.session_name,
        participant_id: $('participant').value,
        prompt: p,
        mime_type: lastBlob.type,
        audio_base64: String(reader.result).split(',')[1],
        duration_ms: startedAt ? Date.now() - startedAt.getTime() : null,
        client_started_at: startedAt ? startedAt.toISOString() : null,
        user_agent: navigator.userAgent,
        consent: {{
          accepted: true,
          version: PLAN.consent.version,
          accepted_at: consentAcceptedAt || localStorage.getItem('voiceCollectionConsentAcceptedAt'),
        }},
      }}),
    }});
    if (!res.ok) {{ $('status').textContent = await res.text(); return; }}
    const data = await res.json();
    $('status').textContent = `保存しました: ${{data.record.audio_path}}`;
    setTimeout(next, 450);
  }};
  reader.readAsDataURL(lastBlob);
}};
render();
openConsentIfNeeded();
</script>
</body>
</html>
"""


class VoiceCollectionHandler(BaseHTTPRequestHandler):
    plan: dict[str, Any] = {}
    output_root: Path = DEFAULT_OUTPUT_ROOT

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: HTTPStatus, body: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/prompts":
            self._send_json(HTTPStatus.OK, self.plan)
            return
        if path in {"/", "/index.html"}:
            self._send_text(HTTPStatus.OK, render_index_html(self.plan), content_type="text/html; charset=utf-8")
            return
        self._send_text(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/upload":
            self._send_text(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES * 2:
                raise ValueError(f"invalid content length: {length}")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = save_recording_upload(payload, output_root=self.output_root)
        except Exception as exc:  # keep upload errors visible in the browser
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("voice-collection: " + format % args + "\n")


def build_server(*, host: str, port: int, output_root: Path, prompt_plan: dict[str, Any]) -> ThreadingHTTPServer:
    configured_output_root = output_root
    configured_prompt_plan = prompt_plan

    class Handler(VoiceCollectionHandler):
        plan = configured_prompt_plan
        output_root = configured_output_root

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a temporary real-voice collection web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompts", type=Path, help="Optional prompt-plan JSON. Defaults to the built-in Stack-chan real-eval plan.")
    args = parser.parse_args(argv)

    plan = load_prompt_plan(args.prompts)
    server = build_server(host=args.host, port=args.port, output_root=args.output_root, prompt_plan=plan)
    print(f"Serving voice collection app at http://{args.host}:{args.port}/")
    print(f"Saving uploads under {args.output_root.resolve()}")
    server.serve_forever()


if __name__ == "__main__":
    main()
