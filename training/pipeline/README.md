# Training pipeline smoke runner

This package validates the local TTS/data/training handoff plumbing for **hai_stackchan_ja**.

It is intentionally small:

- local TTS: Piper Plus Tsukuyomi-chan model
- generated samples: Japanese positive variants, hard negatives, and English holdout/near-miss
- normalization: 16 kHz mono WAV
- augmentation: speed, gain, reverb, background noise
- split: positive/negative train + validation, with holdout kept separate
- output: microWakeWord-style `.tflite` + JSON manifest handoff artifacts

The current `.tflite` is a **smoke placeholder** that preserves the expected microWakeWord artifact contract. It is not a production trained model and its threshold sweep is not a production FAR/FRR claim.

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

## Next step

Replace `suburi_wakeword.microwakeword.train_microwakeword_smoke_model` with a full upstream microWakeWord trainer run once RaggedMmap spectrogram feature generation is wired. The real handoff adapter already rejects placeholder `.tflite` bytes and only publishes non-placeholder upstream `stream_state_internal_quant.tflite` exports via `train_microwakeword_model`.
