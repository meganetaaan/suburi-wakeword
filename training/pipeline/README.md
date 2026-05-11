# Training pipeline smoke runner

This package currently validates the local TTS/data/training plumbing for **hai_stackchan_ja**.

It is intentionally small:

- local TTS: Piper Plus Tsukuyomi-chan Japanese model
- generated samples: Japanese positive variants and hard negatives
- normalization: 16 kHz mono WAV
- model: tiny centroid classifier used only as a smoke artifact

The smoke model is **not** the final microWakeWord model. It exists to prove that local Japanese TTS generation, dataset manifest creation, audio normalization, feature extraction, training, and inference artifact writing can run end-to-end.

## Test

```bash
pnpm pipeline:test
```

or from this directory:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

## Run smoke pipeline

```bash
pnpm pipeline:smoke
```

The command writes ignored local outputs under:

```text
training/pipeline/runs/smoke/hai_stackchan_ja/
```

A timestamped run can be produced with:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --output-root runs/smoke/hai_stackchan_ja_$(date +%Y%m%d_%H%M%S) \
  --samples-per-variant 2
```

## Piper Plus runtime note

The current PyPI `piper` CLI installed from `piper-plus` can list and download Japanese Piper Plus voices, but the 2026 MB-iSTFT Japanese models require `speaker_embedding` inputs. For this smoke pipeline, the command therefore clones `ayutaz/piper-plus` `dev` under `.local-data/tools/piper-plus` and runs:

```bash
uv run python -m piper_train.infer_onnx
```

from the source runtime with `onnxruntime`, `soundfile`, and `pyopenjtalk-plus` installed into that local ignored checkout.

## Next step

Replace the centroid smoke classifier with real microWakeWord training once the TTS generation and dataset contract are stable.
