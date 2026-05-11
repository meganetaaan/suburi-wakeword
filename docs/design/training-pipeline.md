# Training Pipeline Design

## Goal

Train a microWakeWord-compatible model for **「ハイ、ｽﾀｯｸﾁｬﾝ」** and produce artifacts that can be evaluated on ESP32-S3.

## Proposed repository layout

```text
apps/
  collector/              # Web app for human recordings
packages/
  dataset-manifest/        # Shared metadata schema and validators
training/
  tts/                     # Local TTS sample generation scripts
  augmentation/            # Noise/reverb/speed/gain augmentation scripts
  microwakeword/           # microWakeWord training wrapper and notebooks
  evaluation/              # FAR/FRR/latency evaluation scripts
firmware/
  esp32s3-smoke/           # Minimal esp-tflite-micro smoke project, later
models/
  public/                  # Releaseable manifests / tiny demo models only
  private/                 # Ignored local model outputs
```

## Data classes

### Positive samples

Phrase variants:

- `ハイ、ｽﾀｯｸﾁｬﾝ`
- `はい、ｽﾀｯｸﾁｬﾝ`
- `ハイスタックチャン`
- `ハイ、スタックちゃん`

Keep the canonical model label as `hai_stackchan`.

### Hard negative samples

Include phrases that should **not** wake the device:

- `ｽﾀｯｸﾁｬﾝ`
- `スタック`
- `ちゃん`
- `ハイ、〇〇ちゃん`
- `ねえ、スタックちゃん` if not part of the target phrase
- Japanese conversations containing `ちゃん`
- English-like `stack`, `stuck`, `stack chan`
- TV/music/room noise/speaker playback

## Local TTS plan

1. Start with Piper + `piper-sample-generator`.
2. Generate multiple speed and speaker variants.
3. Resample all generated audio to 16 kHz mono WAV.
4. If Piper Japanese voice quality is insufficient, create a comparison table for:
   - Coqui TTS
   - Style-Bert-VITS2
   - VOICEVOX only if license and generated-output usage are acceptable for training
5. Do not mix engines blindly; tag every sample with `source_engine`, `voice`, and `license_note`.

## Training flow

```text
text prompts
  -> local TTS WAV generation
  -> normalization to 16 kHz mono
  -> augmentation
  -> dataset manifest creation
  -> microWakeWord spectrogram generation
  -> training
  -> quantized streaming .tflite export
  -> model JSON manifest
  -> offline evaluation
  -> ESP32-S3 smoke evaluation
```

## Artifact contract

Each trained candidate produces:

```text
artifacts/<run-id>/
  model/stream_state_internal_quant.tflite
  model/hai_stackchan.json
  metrics/offline.json
  metrics/threshold-sweep.csv
  manifest/dataset.jsonl
  README.md
```

The public release candidate may copy only non-sensitive artifacts to `models/public/<version>/`.

## Metrics

Minimum evaluation metrics:

- False reject rate by distance and speaker group
- False accepts per hour on background audio
- Detection latency
- Tensor arena size
- ESP32-S3 RAM/CPU observations

## Initial acceptance target

For a first usable prototype:

- FRR under 10% at 1 m in a quiet room
- FAR below 1 / hour in normal room background audio
- Latency under 1 s
- Runs on ESP32-S3 without PSRAM pressure that disrupts the main Stack-chan loop

Production target should be stricter after field tests.
