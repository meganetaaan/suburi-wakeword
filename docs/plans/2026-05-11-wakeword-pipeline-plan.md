# suburi-wakeword Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after the initial repository scaffold.

**Goal:** Build a reproducible local-TTS + microWakeWord training pipeline and a consented Web collection app for the Japanese wake phrase **「ハイ、ｽﾀｯｸﾁｬﾝ」**.

**Architecture:** Use a pnpm monorepo. Keep browser collection, dataset schemas, training scripts, and firmware smoke tests separate. Store code and manifests in Git; keep raw human voice and private training artifacts outside Git.

**Tech Stack:** Vite + React + TypeScript, Zod, Python training scripts, Piper/piper-sample-generator, microWakeWord, TensorFlow Lite Micro / esp-tflite-micro.

---

## Phase 0: Repository foundation

### Task 0.1: Commit documentation scaffold

**Objective:** Land README, ADRs, and design notes so later implementation has a stable target.

**Files:**
- `README.md`
- `docs/adr/0001-use-microwakeword-and-local-tts.md`
- `docs/adr/0002-human-voice-data-privacy-boundary.md`
- `docs/design/training-pipeline.md`
- `docs/design/web-voice-collection.md`
- `scripts/check-docs.mjs`

**Verify:**

```bash
pnpm install
pnpm run ci
```

Expected: docs check passes.

## Phase 1: Dataset contract

### Task 1.1: Create dataset manifest package

**Objective:** Define the schema for human/TTS samples before implementing ingestion.

**Files:**
- Create: `packages/dataset-manifest/package.json`
- Create: `packages/dataset-manifest/src/sample-manifest.ts`
- Create: `packages/dataset-manifest/src/sample-manifest.spec.ts`

**Test first:** Validate that missing `consentVersion` fails for `human_web` samples and that TTS samples require a `sourceEngine`.

**Implementation:** Use Zod schemas for:
- `SampleManifestRecord`
- `ParticipantMetadataRecord`
- `PhraseId`
- `ReviewStatus`

**Verify:**

```bash
pnpm --filter @suburi-wakeword/dataset-manifest test
pnpm --filter @suburi-wakeword/dataset-manifest typecheck
```

### Task 1.2: Add manifest CLI validation

**Objective:** Validate `dataset.jsonl` files locally before training.

**Files:**
- Create: `packages/dataset-manifest/src/cli.ts`
- Create: `packages/dataset-manifest/src/cli.spec.ts`

**Verify:**

```bash
pnpm --filter @suburi-wakeword/dataset-manifest validate fixtures/sample-dataset.jsonl
```

## Phase 2: Web voice collector

### Task 2.1: Scaffold Vite React collector

**Objective:** Create `apps/collector` with ownership-first structure.

**Files:**
- Create: `apps/collector/package.json`
- Create: `apps/collector/src/app/*`
- Create: `apps/collector/src/pages/*`
- Create: `apps/collector/src/entities/recording/*`
- Create: `apps/collector/src/shared/*`

**Verify:**

```bash
pnpm --filter @suburi-wakeword/collector test
pnpm --filter @suburi-wakeword/collector typecheck
pnpm --filter @suburi-wakeword/collector build
```

### Task 2.2: Consent-first collection flow

**Objective:** Prevent recording until explicit consent is accepted.

**Test first:** Render the recording route without consent and assert the record button is unavailable.

**Files:**
- `apps/collector/src/pages/consent-page/consent-page.tsx`
- `apps/collector/src/flows/recording-session/recording-session.tsx`

**Acceptance:** User must accept consent version before microphone access is requested.

### Task 2.3: Browser recording component

**Objective:** Capture 5〜10 takes using `MediaRecorder`.

**Test first:** Mock `navigator.mediaDevices.getUserMedia` and `MediaRecorder` to verify start/stop/retry behavior.

**Files:**
- `apps/collector/src/entities/recording/model/recording-session.ts`
- `apps/collector/src/entities/recording/widgets/recording-panel.tsx`

**Acceptance:** User can record, preview, delete, and retry each take before submission.

### Task 2.4: Upload adapter boundary

**Objective:** Keep storage provider replaceable.

**Files:**
- `apps/collector/src/entities/recording/api/upload-recording.ts`
- `apps/collector/src/entities/recording/api/upload-recording.spec.ts`

**Initial implementation:** local dev mock adapter that writes metadata to console or a dev endpoint. Do not commit raw audio.

## Phase 3: Local TTS sample generation

### Task 3.1: Python environment and TTS CLI

**Objective:** Create a repeatable local generation command.

**Files:**
- Create: `training/tts/pyproject.toml`
- Create: `training/tts/suburi_tts/generate.py`
- Create: `training/tts/suburi_tts/prompts/hai_stackchan.txt`

**Default command:**

```bash
cd training/tts
uv run suburi-tts-generate --phrase-id hai_stackchan --output ../../data/raw/tts/hai_stackchan
```

**Notes:** Use Piper first. If Japanese voices are not acceptable, add an engine abstraction before adding a second engine.

### Task 3.2: TTS license and quality report

**Objective:** Track which voice engines can be used safely.

**Files:**
- Create: `docs/research/local-japanese-tts-options.md`
- Create: `training/tts/voices/README.md`

**Acceptance:** Each candidate has license, local setup, sample quality, and training suitability noted.

## Phase 4: Augmentation and normalization

### Task 4.1: Audio normalization command

**Objective:** Convert browser/TTS outputs to 16 kHz mono WAV.

**Files:**
- Create: `training/audio/suburi_audio/normalize.py`
- Create: `training/audio/tests/test_normalize.py`

**Verify:** Use a small generated fixture and assert output format via `soundfile` or `ffprobe`.

### Task 4.2: Augmentation command

**Objective:** Generate reverb/noise/speed/gain variants.

**Files:**
- Create: `training/audio/suburi_audio/augment.py`
- Create: `training/audio/tests/test_augment.py`

**Acceptance:** Output manifest records augmentation parameters for every derived sample.

## Phase 5: microWakeWord training wrapper

### Task 5.1: Pin microWakeWord dependency

**Objective:** Make training reproducible.

**Files:**
- Create: `training/microwakeword/README.md`
- Create: `training/microwakeword/requirements.lock` or `uv.lock`
- Create: `training/microwakeword/config/hai_stackchan.yaml`

**Acceptance:** A fresh environment can reproduce a dry-run training setup.

### Task 5.2: Training runner

**Objective:** Wrap notebook logic in a CLI command.

**Files:**
- Create: `training/microwakeword/suburi_mww/train.py`
- Create: `training/microwakeword/suburi_mww/export_manifest.py`

**Output:**

```text
artifacts/<run-id>/model/stream_state_internal_quant.tflite
artifacts/<run-id>/model/hai_stackchan.json
artifacts/<run-id>/metrics/offline.json
```

## Phase 6: Evaluation

### Task 6.1: Offline threshold sweep

**Objective:** Choose probability cutoff and sliding window candidates.

**Files:**
- Create: `training/evaluation/suburi_eval/threshold_sweep.py`
- Create: `training/evaluation/tests/test_threshold_sweep.py`

**Acceptance:** Produce FAR/FRR table per threshold.

### Task 6.2: ESP32-S3 smoke app

**Objective:** Verify `.tflite` can run on ESP32-S3 before touching Stack-chan firmware.

**Files:**
- Create: `firmware/esp32s3-smoke/README.md`
- Create: `firmware/esp32s3-smoke/main/*`

**Acceptance:** Serial log prints inference scores and memory usage.

## Phase 7: Stack-chan integration proposal

### Task 7.1: Write integration ADR

**Objective:** Decide how wake detection plugs into Stack-chan.

**Files:**
- Create: `docs/adr/0003-stackchan-firmware-integration-boundary.md`

**Acceptance:** ADR covers task priority, audio capture ownership, AFE/VAD coexistence, memory, and fallback to ESP-SR WakeNet.

---

## Open questions

1. Which Japanese local TTS voice has acceptable pronunciation and redistributable generated output?
2. Will the first public collection app use Cloudflare R2, private S3, or local-only export?
3. Should the target phrase include `ねえ、ｽﾀｯｸﾁｬﾝ` as a second model or only as a negative sample?
4. What is the acceptable false accept budget for Stack-chan birthday/event demo use?
