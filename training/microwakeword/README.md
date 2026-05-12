# microWakeWord training handoff

This directory pins the shape of the real microWakeWord trainer integration for the **hai_stackchan_ja** pipeline.

Current repository state:

1. `training/pipeline` creates a normalized/split/augmented dataset manifest.
2. The smoke runner exports the same public handoff paths expected from microWakeWord:
   - `artifacts/model/stream_state_internal_quant.tflite`
   - `artifacts/model/hai_stackchan_ja.json`
   - `artifacts/metrics/threshold-sweep.json`
3. The smoke runner's `.tflite` is marked as a placeholder and is rejected by the real handoff path.
4. `suburi_wakeword.microwakeword.train_microwakeword_model` wraps the upstream microWakeWord export contract:
   - stage `dataset.jsonl` audio into microWakeWord feature roots
   - generate RaggedMmap spectrogram features from staged 16 kHz PCM WAVs
   - build an upstream `microwakeword.model_train_eval` command with `--test_tflite_streaming_quantized 1`
   - expect `tflite_stream_state_internal_quant/stream_state_internal_quant.tflite` under the upstream model train directory
   - copy only non-placeholder TFLite bytes into `artifacts/model/`
   - write the ESPHome-compatible microWakeWord JSON manifest with `training_backend: microWakeWord`

## Feature generation shape

The adapter stages audio from `dataset.jsonl` into this ignored run-local structure:

```text
<run>/features/positive/training/wav/*.wav
<run>/features/positive/validation/wav/*.wav
<run>/features/positive/testing/wav/*.wav      # smoke workaround: mirrors positive validation
<run>/features/negative/training/wav/*.wav
<run>/features/negative/validation/wav/*.wav
<run>/features/negative/testing/wav/*.wav      # holdout / near-miss
```

The positive `testing` mirror exists because upstream ROC evaluation expects positive testing samples. It is only a smoke-training workaround until a dedicated positive test split exists; English `Hi, Stack-chan` remains holdout/near-miss and is not copied into positives.

It then writes and runs an absolute-path command from the upstream checkout cwd:

```bash
uv run python <absolute-run>/artifacts/microwakeword-train/generate_features.py
```

The generated script avoids the heavier upstream `Clips`/`torchcodec` route and reads staged WAVs directly with `wave` + `numpy`, then uses upstream feature APIs:

```python
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.audio.spectrograms import SpectrogramGeneration
```

It writes `wakeword_mmap` folders under each split. These are the feature roots referenced by `training_config.json`.

## Upstream command shape

The adapter prepares a training config and runs the upstream entrypoint from a checked-out `OHF-Voice/micro-wake-word` source tree:

```bash
uv run python -m microwakeword.model_train_eval \
  --training_config <absolute-run>/artifacts/microwakeword-train/training_config.json \
  --train 1 \
  --test_tflite_streaming_quantized 1 \
  mixednet --residual_connection 0,0,0,0
```

The training config keeps generated scripts/config separate from the upstream model output directory:

```text
<run>/artifacts/microwakeword-train/training_config.json
<run>/artifacts/microwakeword-train/generate_features.py
<run>/artifacts/microwakeword-model/
```

Expected upstream export:

```text
<run>/artifacts/microwakeword-model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

Public handoff copy:

```text
<run>/artifacts/model/stream_state_internal_quant.tflite
<run>/artifacts/model/hai_stackchan_ja.json
```

## Local upstream environment used for smoke training

A successful minimal upstream smoke run used an ignored local checkout under:

```text
training/pipeline/.local-data/tools/micro-wake-word
```

Setup shape:

```bash
cd training/pipeline/.local-data/tools/micro-wake-word
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python tensorboard 'numpy<2'
```

Notes from the smoke run:

- TensorBoard is required because upstream training writes `tf.summary.scalar`.
- NumPy must currently be `<2` for upstream evaluation code that still calls `np.trapz`.
- The local ignored upstream checkout needed a smoke-only compatibility patch around ndarray `.numpy()` handling in `microwakeword/train.py`. Keep that patch out of this repository unless it is turned into an upstream PR or a documented vendored patch.

Minimal smoke training/export has produced a non-placeholder `stream_state_internal_quant.tflite` and the real handoff copy for `hai_stackchan_ja`. Treat this only as a pipeline smoke artifact, not as production wake-word quality.

Do not mix English `Hi, Stack-chan` into positive samples for the Japanese baseline. Keep it in holdout/near-miss until the Japanese-only baseline has FRR/FAR numbers.
