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
   - stage `dataset.jsonl` audio into microWakeWord feature roots
   - generate RaggedMmap spectrogram features through upstream `Clips`, `SpectrogramGeneration`, and `RaggedMmap.from_generator`
   - build an upstream `microwakeword.model_train_eval` command with `--test_tflite_streaming_quantized 1`
   - expect `tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
   - copy only non-placeholder TFLite bytes into `artifacts/model/`
   - write the ESPHome-compatible microWakeWord JSON manifest with `training_backend: microWakeWord`

## Feature generation shape

The adapter stages audio from `dataset.jsonl` into this ignored run-local structure:

```text
<run>/features/positive/training/wav/*.wav
<run>/features/positive/validation/wav/*.wav
<run>/features/negative/training/wav/*.wav
<run>/features/negative/validation/wav/*.wav
<run>/features/negative/testing/wav/*.wav     # holdout / near-miss
```

It then writes and runs:

```bash
uv run python <run>/artifacts/microwakeword-train/generate_features.py
```

The generated script uses the upstream APIs:

```python
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
```

and writes `wakeword_mmap` folders under each split. These are the feature roots referenced by `training_config.json`.

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

The current adapter now stages WAV files and generates a `generate_features.py` script that uses upstream `Clips`, `SpectrogramGeneration`, and `RaggedMmap.from_generator` to create `wakeword_mmap` feature folders before training. A full upstream training run still requires the microWakeWord/TensorFlow dependency environment to be installed under the chosen source checkout.

```text
features_dir/
  training/*_mmap
  validation/*_mmap
  testing/*_mmap
  validation_ambient/*_mmap
  testing_ambient/*_mmap
```

Do not mix English `Hi, Stack-chan` into positive samples for the Japanese baseline. Keep it in holdout/near-miss until the Japanese-only baseline has FRR/FAR numbers.
