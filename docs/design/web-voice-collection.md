# Web Voice Collection Design

## Goal

Collect consented human recordings of **「ハイ、ｽﾀｯｸﾁｬﾝ」** through a browser app, without leaking raw voice data into Git.

## User flow

1. Landing page explains the experiment.
2. Consent page asks for explicit agreement.
3. Optional metadata form:
   - display name or anonymous
   - age band, optional
   - voice range / gender, optional
   - device/browser, auto-detected when possible
   - permission to contact for deletion/follow-up, optional
4. Recording page shows one prompt at a time.
5. User records 5〜10 takes.
6. User can listen, retry, or delete before submit.
7. Submit uploads WAV/WebM plus metadata.
8. Completion page explains deletion contact/process.

## Privacy boundary

- Do not collect unnecessary personal data.
- Store contributor metadata separately from audio blob paths.
- Generate random participant IDs.
- Keep a deletion token or contact-based deletion procedure.
- Never expose raw recordings through public static hosting.

## Recommended app stack

For the first implementation:

- Vite + React + TypeScript for the browser app
- Web Audio / MediaRecorder for capture
- Zod schema shared through `packages/dataset-manifest`
- Backend adapter selected later:
  - local filesystem for development
  - Cloudflare R2 presigned upload or private S3-compatible storage for public collection
  - avoid GitHub issues/releases for raw audio

## Recording format

Browsers may record `webm/opus`. The pipeline should normalize server-side or offline to:

- 16 kHz
- mono
- 16-bit PCM WAV

The manifest should record original format and normalized output path.

## Abuse and quality controls

- Record a short calibration/noise sample.
- Reject clips that are too short, too long, clipped, or silent.
- Ask for headphones/offline quiet room guidance, but do not overburden contributors.
- Rate-limit public submissions.
- Add manual review status: `new`, `accepted`, `rejected`, `delete_requested`.

## Minimal data manifest shape

```json
{
  "sampleId": "uuid",
  "participantId": "uuid",
  "phraseId": "hai_stackchan",
  "promptText": "ハイ、ｽﾀｯｸﾁｬﾝ",
  "take": 1,
  "source": "human_web",
  "originalMimeType": "audio/webm",
  "normalizedAudioPath": "data/processed/.../sample.wav",
  "consentVersion": "2026-05-11",
  "reviewStatus": "new",
  "createdAt": "ISO-8601"
}
```
