# Training pipeline smoke runner

This package validates the local TTS/data/training handoff plumbing for **hai_stackchan_ja**.

It is intentionally small:

- local TTS: Piper Plus Tsukuyomi-chan model
- generated samples: Japanese positive variants, hard negatives, and English holdout/near-miss
- normalization: 16 kHz mono WAV
- augmentation: speed, gain, reverb, background noise
- split: positive/negative train + validation, with holdout kept separate
- output: microWakeWord-style `.tflite` + JSON manifest handoff artifacts

`run_smoke_pipeline` still writes a **smoke placeholder** `.tflite` so the fast TTS/data pipeline can run without TensorFlow. The real handoff path, `suburi_wakeword.microwakeword.train_microwakeword_model`, rejects that placeholder and only publishes non-placeholder upstream microWakeWord exports. Neither smoke threshold sweep nor minimal upstream training output is a production FAR/FRR claim.

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

Key outputs:

```text
artifacts/model/stream_state_internal_quant.tflite
artifacts/model/hai_stackchan_ja.json
artifacts/metrics/threshold-sweep.json
artifacts/metrics/validation-scores.jsonl
dataset.jsonl
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

from the source runtime with `onnxruntime`, `soundfile`, `pyopenjtalk-plus`, and `g2p-en` installed into that local ignored checkout. English holdout synthesis also downloads the NLTK `averaged_perceptron_tagger_eng`, `averaged_perceptron_tagger`, and `cmudict` resources for `g2p-en`.

## lightweight cross-validation smoke

For a quick relative accuracy check across generated dataset sizes, use `suburi_wakeword.evaluation.cross_validate_manifest(...)` or `compare_training_scales(...)`. This is a nearest-centroid classifier over the repository's simple audio features; it is useful for smoke comparisons, but it is **not** production microWakeWord streaming accuracy and must not be converted to FAR/hour.

The latest samples-per-variant 1 vs 3 run is recorded in `docs/memo/2026-05-12-cross-validation-smoke.md`.

## microWakeWord upstream smoke

For the real trainer handoff, keep the upstream checkout in an ignored local tools directory:

```text
training/pipeline/.local-data/tools/micro-wake-word
```

A minimal successful environment used:

```bash
cd training/pipeline/.local-data/tools/micro-wake-word
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python tensorboard 'numpy<2'
```

Current adapter notes:

- `generate_features.py` is generated under `<run>/artifacts/microwakeword-train/` and is invoked with an absolute path from the upstream checkout cwd.
- The generated feature script reads staged 16 kHz PCM WAVs directly with `wave` + `numpy`, then calls upstream `generate_features_for_clip`, `SpectrogramGeneration`, and `RaggedMmap`; it intentionally avoids the heavier `Clips`/`torchcodec` route for smoke generation.
- `training_config.json` points at absolute positive/negative feature roots and uses `<run>/artifacts/microwakeword-model` as upstream `train_dir`.
- The `mixednet` command includes `--residual_connection 0,0,0,0` for the minimal smoke architecture.
- Upstream ROC evaluation currently needs positive testing samples, so the adapter mirrors positive validation WAVs into `features/positive/testing/wav/` until a dedicated positive test split exists. English holdout remains negative/holdout-side and is not mixed into positives.
- TensorBoard is required for upstream training summaries.
- NumPy is pinned to `<2` in the local upstream venv because upstream evaluation still calls `np.trapz`.

A minimal smoke training/export has produced:

```text
<run>/artifacts/microwakeword-model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
<run>/artifacts/model/stream_state_internal_quant.tflite
<run>/artifacts/model/hai_stackchan_ja.json
```

Those files are ignored generated artifacts. Do not commit them.
