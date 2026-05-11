# microWakeWord training handoff

This directory pins the shape of the real microWakeWord trainer integration.

Current repository state:

1. `training/pipeline` creates a normalized/split/augmented dataset manifest.
2. The smoke runner exports the same handoff paths expected from microWakeWord:
   - `artifacts/model/stream_state_internal_quant.tflite`
   - `artifacts/model/hai_stackchan_ja.json`
   - `artifacts/metrics/threshold-sweep.json`
3. The smoke runner's `.tflite` is marked as a placeholder and is rejected by the real handoff path.
4. `suburi_wakeword.microwakeword.train_microwakeword_model` now wraps the upstream microWakeWord export contract:
   - build an upstream `microwakeword.model_train_eval` command with `--test_tflite_streaming_quantized 1`
   - expect `tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
   - copy only non-placeholder TFLite bytes into `artifacts/model/`
   - write the ESPHome-compatible microWakeWord JSON manifest with `training_backend: microWakeWord`

## Upstream command shape

The adapter prepares a training config and runs the upstream entrypoint from a checked-out `OHF-Voice/micro-wake-word` source tree:

```bash
uv run python -m microwakeword.model_train_eval \
  --training_config <run>/artifacts/microwakeword-train/training_config.json \
  --train 1 \
  --test_tflite_streaming_quantized 1 \
  mixednet
```

Expected upstream export:

```text
<run>/artifacts/microwakeword-train/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

The current adapter does **not** yet generate RaggedMmap spectrogram feature folders from `dataset.jsonl`; that is the next required step before a full upstream training run can complete. The notebook expects feature directories shaped like:

```text
features_dir/
  training/*_mmap
  validation/*_mmap
  testing/*_mmap
  validation_ambient/*_mmap
  testing_ambient/*_mmap
```

Do not mix English `Hi, Stack-chan` into positive samples for the Japanese baseline. Keep it in holdout/near-miss until the Japanese-only baseline has FRR/FAR numbers.
