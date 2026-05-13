# 2026-05-13 real voice collection app

## Purpose

Collect consented real voice recordings for evaluation-only wake-word holdout sets before mixing human audio into training.

The app is a small Python stdlib web server:

1. Shows one prompt at a time.
2. Records microphone audio in the browser with `MediaRecorder`.
3. Saves each take under a label/prompt directory on the server.
4. Appends upload metadata to `manifest.jsonl`.

Recorded audio is intentionally ignored by Git under `data/real-voice/`.

## Run locally

From repository root:

```bash
pnpm voice:collect
```

Equivalent direct command:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.voice_collection_server \
  --host 0.0.0.0 \
  --port 8787 \
  --output-root data/real-voice/collection
```

Open:

```text
http://127.0.0.1:8787/
```

For temporary external collection, expose the local server with ngrok or a similar tunnel:

```bash
ngrok http 8787
```

or, when ngrok is not configured:

```bash
cloudflared tunnel --url http://127.0.0.1:8787 --no-autoupdate
```

Use HTTPS tunnel URLs for phone browsers; microphone recording generally requires a secure context except on localhost.

## Data layout

Default output root:

```text
training/pipeline/data/real-voice/collection/
```

Example saved file:

```text
training/pipeline/data/real-voice/collection/hai-stackchan-real-eval-v1/negative/n005/20260513T120001Z_speaker-01_n005-t01.webm
```

Manifest:

```text
training/pipeline/data/real-voice/collection/hai-stackchan-real-eval-v1/manifest.jsonl
```

Each manifest row includes:

- `recorded_at`
- `session_name`
- `participant_id` sanitized for filenames
- `prompt_id`
- `take_id`
- `label`: `positive`, `negative`, `holdout`, or `ambient`
- `text`
- `duration_ms`
- `mime_type`
- `audio_format`
- `audio_bytes`
- `audio_path`
- `client_started_at`
- `user_agent`
- `consent`: accepted flag, terms version, and accepted timestamp

## Default prompt plan

Built-in plan: `hai-stackchan-real-eval-v1`.

It includes:

- positives: `ハイ、スタックちゃん`, `はい、スタックちゃん`, `ハイスタックちゃん`, `ハイ、ｽﾀｯｸﾁｬﾝ`
- near-miss negatives: `スタックちゃんと呼びました`, `スタックちゃん、こんにちは`, `はい、スタックチャンネル`, `ハイ、スタックチャンネル`, `スタッキーちゃん`, `スタッフさん`, `スタックあんちゃん`, `ねえ、スタックチャンネルを開いて`
- English holdout: `Hi, Stack-chan`
- ambient/no-read prompt

A custom prompt file can be supplied:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.voice_collection_server \
  --prompts config/real-voice-prompts.example.json
```

Prompt JSON shape:

```json
{
  "session_name": "hai-stackchan-real-eval-v1",
  "wake_phrase": "ハイ、スタックちゃん",
  "instructions": "静かな場所で自然に読んでください。",
  "consent": {
    "version": "real-voice-eval-v1",
    "required": true,
    "title": "評価用音声の収集について",
    "items": [
      "録音データは評価用に保存します。",
      "まずは評価用として扱い、明示的な追加確認なしに学習には使いません。",
      "公開リポジトリにはコミットしません。"
    ],
    "checkbox_label": "上記を確認し、評価用音声として録音・保存することに同意します。"
  },
  "prompts": [
    { "id": "p001", "label": "positive", "text": "ハイ、スタックちゃん", "repeat": 4 },
    { "id": "n001", "label": "negative", "text": "スタッキーちゃん", "repeat": 2 }
  ]
}
```

`repeat` expands to stable take IDs such as `p001-t01`, `p001-t02`.

## Collection guidance

Recommended first pass:

| kind | target |
| --- | ---: |
| positive wake phrase variants | 50-100 |
| accept-ish positive variants | 30-50 |
| near-miss negative | 100-200 |
| suffix/context negative | 50-100 |
| ambient/silence/conversation fragments | 50-100 |

Start with one speaker to test the pipeline, but do not use that for adoption decisions. Aim for at least three speakers before judging thresholds.

## Privacy boundary

- Show the in-app consent dialog before recording; server-side upload rejects recordings without `consent.accepted: true`.
- Get explicit consent before sharing the public URL.
- Do not commit real recordings or manifests.
- Prefer participant IDs like `speaker-01`, not real names.
- Treat public tunnel URLs as temporary and revocable.
- Stop the server/tunnel when collection is done.

## Preparing recordings for evaluation

Browser uploads are WebM/Opus on Chrome/Android. Decode them to the pipeline's 16 kHz mono PCM WAV manifest format before real microWakeWord evaluation:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.real_voice_collection summarize \
  data/real-voice/collection/hai-stackchan-real-eval-v1/manifest.jsonl

PYTHONPATH=src uv run python -m suburi_wakeword.real_voice_collection prepare-eval \
  data/real-voice/collection/hai-stackchan-real-eval-v1/manifest.jsonl \
  runs/real-voice/hai_stackchan_real_eval_v1_<speaker>_<date>/dataset.jsonl
```

`prepare-eval` requires `ffmpeg` for WebM/Opus decoding and writes `split: real_holdout` rows with both `audio_path` and `normalized_audio_path` pointing at generated 16 kHz WAV files.

## First collection smoke

The first manual collection produced 38 samples from participant `sskw`:

| label | count |
| --- | ---: |
| positive | 14 |
| negative | 20 |
| holdout | 2 |
| ambient | 2 |

Converted eval manifest:

```text
training/pipeline/runs/real-voice/hai_stackchan_real_eval_v1_sskw_20260513/dataset.jsonl
```

Audio sanity after conversion: 38/38 WAVs are 16 kHz mono 16-bit PCM; duration range 1.62-4.08 sec; no clipping-like peaks; the two ambient prompts are near-silence as expected.

Existing synthetic-CV operating candidate models do not yet transfer well to this one-speaker real holdout. At their synthetic thresholds, recall is near zero. With a wide threshold sweep, the best `expanded-5k + negative_class_weight=1.25` fold reached recall 0.714 with FAR/sample 0.375 on this tiny real set. Treat this as directional only; collect at least three speakers before changing model selection.

## Verification performed

Focused tests:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m unittest tests.test_voice_collection_server -v
PYTHONPATH=src uv run python -m unittest tests.test_real_voice_collection -v
```

Full repo CI:

```bash
pnpm run ci
```
