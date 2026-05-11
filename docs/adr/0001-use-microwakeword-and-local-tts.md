# ADR 0001: Use microWakeWord with local TTS for the first training pipeline

## Status

Accepted

## Context

ESP-SR WakeNet TTS Pipeline V3 is not open-sourced. We need a path that lets us train and iterate on a Japanese wake phrase for ESP32-S3 without relying on Espressif to generate each model.

The target wake phrase is **「ハイ、ｽﾀｯｸﾁｬﾝ」**. The desired runtime target is ESP32-S3, and the eventual product integration should fit Stack-chan firmware rather than require an external server.

## Decision

Use **microWakeWord** as the first model-training approach and generate positive synthetic samples with a **free local TTS engine**.

Initial TTS choice:

1. Use Piper and `piper-sample-generator` as the default local TTS stack because it is already used in the microWakeWord ecosystem and can run locally.
2. Prefer Japanese Piper voices for the canonical Japanese phrase. The initial voice search should include community Japanese Piper voices such as Tsukuyomi-chan-derived examples, with license verification before use.
3. If Japanese Piper voice quality is insufficient, evaluate Coqui TTS / Style-Bert-VITS2 / other local Japanese TTS engines for sample generation as secondary engines.
4. Treat generated audio licensing as part of model release readiness.

For ESP32-S3 integration, do not assume ESPHome runtime will be embedded directly. Prefer exporting `.tflite` and integrating via `esp-tflite-micro` or a small firmware-side adapter.

## Consequences

- We can iterate without waiting for Espressif's closed pipeline.
- Model quality depends heavily on generated samples, hard negatives, and real-device threshold tuning.
- Training will not happen on ESP32-S3; ESP32-S3 runs inference only.
- The first milestone is an experiment pipeline, not a polished production wake word.
