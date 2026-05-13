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

For a larger synthetic plan, use the expanded dataset profile. It multiplies phrase variants, hard/near negatives, and Piper Plus prosody knobs (`length_scale`, `noise_scale`, `noise_scale_w`) while preserving `voice_profile_id`, `model_id`, `speaker_id`, and `prosody_id` in `tts-jobs.jsonl` and `dataset.jsonl`:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded \
  --output-root runs/smoke/hai_stackchan_ja_expanded_$(date +%Y%m%d_%H%M%S)
```

Current dry-run counts for `expanded` with the default verified Tsukuyomi voice/model profile are 125 base TTS jobs before augmentation: 50 positive, 70 negative, and 5 English holdout. Adding more `VoiceProfile`s multiplies those counts without changing the downstream manifest schema.

The recommended first thousands-scale profile is `expanded-5k`. It expects four voice profiles and creates 912 base TTS jobs: 288 positive, 576 negative, and 48 English holdout. With the current raw + augmentation expansion this produces 6,384 utterances. Follow-up profiles add targeted negatives for experiments: `expanded-5k-hard-negatives` adds Japanese near-miss continuations from false-accept analysis, while `expanded-5k-lexical-negatives` focuses on non-accept lexical impostors such as `スタッキーちゃん` and `スタックあんちゃん`. The first hard-negative real CV run lowered recall too much, so treat these profiles as rebalance experiments rather than the default dataset.

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k \
  --voice-profile-config config/voice-profiles.example.json \
  --dry-run \
  --output-root runs/dry-run/hai_stackchan_ja_expanded5k
```

`--dry-run` writes `tts-jobs.jsonl` and `dataset-summary.json` without synthesizing WAVs. Replace `config/voice-profiles.example.json` with a local ignored config once real alternative models/speakers are available.

## Piper Plus runtime note

The current PyPI `piper` CLI installed from `piper-plus` can list and download Japanese Piper Plus voices, but the 2026 MB-iSTFT Japanese models require `speaker_embedding` inputs. For this smoke pipeline, the command therefore clones `ayutaz/piper-plus` `dev` under `.local-data/tools/piper-plus` and runs:

```bash
uv run python -m piper_train.infer_onnx
```

from the source runtime with `onnxruntime`, `soundfile`, `pyopenjtalk-plus`, and `g2p-en` installed into that local ignored checkout. English holdout synthesis also downloads the NLTK `averaged_perceptron_tagger_eng`, `averaged_perceptron_tagger`, and `cmudict` resources for `g2p-en`.

## real microWakeWord evaluation

Evaluation must use real upstream microWakeWord streaming `.tflite` inference only. Do not use proxy cross-validation, handcrafted audio features, or nearest-centroid metrics for accuracy claims.

```python
from pathlib import Path
from suburi_wakeword.evaluation import evaluate_microwakeword_manifest
from suburi_wakeword.threshold_sweep import default_thresholds

report = evaluate_microwakeword_manifest(
    Path("runs/eval/hai_stackchan_ja_samples1_20260512_093259/dataset.jsonl"),
    Path("runs/eval/hai_stackchan_ja_samples1_20260512_093259/artifacts/model/stream_state_internal_quant.tflite"),
    source_dir=Path(".local-data/tools/micro-wake-word"),
    splits=("validation", "holdout"),
    thresholds=default_thresholds(0.05, 0.95, 0.05),
)
```

The evaluator loads `microwakeword.inference.Model`, feeds each 16 kHz WAV through `predict_clip(...)`, and uses the max streaming score for threshold metrics. It rejects placeholder smoke `.tflite` files. The literal expanded-5k 2-fold real `.tflite` cross-validation run is recorded in `docs/memo/2026-05-13-real-microwakeword-cross-validation.md`; it reached mean accuracy 0.703 at threshold 0.955, 0.710 at threshold 0.96, and 0.729 at threshold 0.965. The FAR reduction sweep in `docs/memo/2026-05-13-real-microwakeword-far-reduction.md` currently favors `negative_class_weight=1.25`: at threshold 0.965, FAR/sample improved from 0.336 to 0.204 while recall stayed at 0.785. The hard-negative iteration in `docs/memo/2026-05-13-expanded5k-hard-negative-iteration.md` lowered threshold-0.965 FAR/sample further to 0.074, but recall fell to 0.517. The lexical-negative iteration in `docs/memo/2026-05-13-lexical-negative-profile.md` is safer: `lexical + negative_class_weight=1.25` at threshold 0.955 matched recall 0.785 and slightly lowered FAR/sample to 0.197, but the gain is small, so keep `expanded-5k + negative_class_weight=1.25` as the operating candidate until another rebalanced profile clearly improves equal-recall FAR. The expanded-5k real validation+holdout run with no local spectrogram sliding is recorded in `docs/memo/2026-05-12-expanded5k-no-slide-real-microwakeword-evaluation.md`; it reached 0.826 accuracy at threshold 0.90 and 0.887 at threshold 0.95 on validation+holdout real streaming `.tflite` inference. The samples-per-variant 1 vs 3 real run is recorded in `docs/memo/2026-05-12-real-microwakeword-evaluation.md`; the samples3 training-step sweep is recorded in `docs/memo/2026-05-12-real-microwakeword-training-step-sweep.md`. The older `docs/memo/2026-05-12-cross-validation-smoke.md` is historical proxy-only data and must not be used for decisions.

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
- The generated feature script writes full upstream spectrograms for each WAV. Do not add local sliding-window shortening before RaggedMmap staging; mixednet already accounts for receptive-field frame drops, and shortening clips before upstream fixed-length handling misaligns training examples with streaming `.tflite` inference.
- `training_config.json` points at absolute positive/negative feature roots and uses `<run>/artifacts/microwakeword-model` as upstream `train_dir`.
- The generated feature command and training command run as `uv run --no-sync ...` so a smoke-compatible local upstream venv (notably `numpy<2`) is not silently re-synced back to incompatible dependency versions.
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
