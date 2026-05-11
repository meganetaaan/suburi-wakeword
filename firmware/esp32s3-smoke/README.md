# ESP32-S3 microWakeWord smoke app

Purpose: verify that the exported microWakeWord `.tflite` and JSON manifest can be copied into an ESP-IDF project before touching Stack-chan firmware.

This app is intentionally small. It does not vendor ESPHome's GPL runtime. The integration target is `esp-tflite-micro` plus a Stack-chan-owned streaming audio/preprocessor boundary.

## Expected artifacts

Run the pipeline first:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --output-root runs/smoke/hai_stackchan_ja \
  --samples-per-variant 1
```

Copy or symlink these files into `main/model/`:

```text
training/pipeline/runs/smoke/hai_stackchan_ja/artifacts/model/stream_state_internal_quant.tflite
training/pipeline/runs/smoke/hai_stackchan_ja/artifacts/model/hai_stackchan_ja.json
```

## Build shape

```bash
idf.py set-target esp32s3
idf.py add-dependency "esp-tflite-micro"
idf.py build flash monitor
```

## Smoke acceptance

Serial output should include:

```text
suburi wakeword smoke app
model bytes: <non-zero>
manifest bytes: <non-zero>
TODO: wire microphone + microWakeWord preprocessor + esp-tflite-micro interpreter
```

The current file is a handoff scaffold. Real inference lands after the generated `.tflite` is produced by the pinned microWakeWord trainer, not by the smoke placeholder.
