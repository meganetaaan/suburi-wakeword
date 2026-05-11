# 2026-05-11 Piper smoke run

## Command

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --output-root runs/smoke/hai_stackchan_ja_20260511_235421 \
  --samples-per-variant 2
```

## Result

The local TTS/data/training smoke pipeline completed end-to-end.

Generated ignored local outputs:

```text
training/pipeline/runs/smoke/hai_stackchan_ja_20260511_235421/
  raw/                  # 14 Piper Plus WAV files, 22050 Hz
  normalized/           # 14 normalized WAV files, 16 kHz mono
  dataset.jsonl         # sample manifest
  artifacts/
    centroid-smoke-model.npz
    metrics.json
    README.md
```

Summary:

- TTS runtime: Piper Plus source runtime (`ayutaz/piper-plus` dev) via `piper_train.infer_onnx`
- Voice/model: `ja_JP-tsukuyomi-chan-medium` / `tsukuyomi-chan-6lang-fp16.onnx`
- Positive samples: 8
- Negative samples: 6
- Total samples: 14
- Normalization: 16 kHz mono WAV
- Smoke learner: centroid classifier, not final microWakeWord
- Smoke self-evaluation accuracy: `0.6428571428571429`

## Notes

The smoke learner is intentionally weak and exists only to prove plumbing. Its metric is not a FAR/FRR estimate and should not be used to judge wake-word quality.

The important result is that local Japanese Piper Plus synthesis, manifest creation, normalization, feature extraction, model artifact writing, and operation check all ran successfully.

## Follow-up

1. Replace the centroid learner with real microWakeWord training.
2. Add better positive/negative split and threshold sweep.
3. Add synthetic audio augmentation before microWakeWord training.
4. Keep raw/generated audio ignored and publish only summaries or releaseable model artifacts after license review.
