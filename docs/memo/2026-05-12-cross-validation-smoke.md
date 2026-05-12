# 2026-05-12 cross-validation smoke

This memo records a lightweight offline accuracy check for the current Piper-generated `hai_stackchan_ja` wake-word dataset.

## Scope

- This is a quick stratified 3-fold cross-validation smoke, not production microWakeWord streaming evaluation.
- Classifier: nearest centroid over the repository's handcrafted `extract_features(...)` audio features.
- Labels: `positive` vs non-positive. `negative` and English holdout / near-miss samples are counted as non-positive.
- Purpose: compare relative behavior when synthetic training data is increased, and catch obvious dataset separability regressions.
- Do not convert `far_per_sample` to FAR/hour without timestamped continuous-audio evaluation.

## Commands

Generated two datasets with different `--samples-per-variant` values:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --output-root runs/eval/hai_stackchan_ja_samples1_20260512_093259 \
  --samples-per-variant 1

PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --output-root runs/eval/hai_stackchan_ja_samples3_20260512_093311 \
  --samples-per-variant 3
```

Then evaluated them with:

```python
from pathlib import Path
from suburi_wakeword.evaluation import compare_training_scales, cross_validate_manifest

roots = [
    Path("runs/eval/hai_stackchan_ja_samples1_20260512_093259"),
    Path("runs/eval/hai_stackchan_ja_samples3_20260512_093311"),
]
for root in roots:
    report = cross_validate_manifest(root / "dataset.jsonl", folds=3)
    (root / "artifacts" / "metrics" / "cross-validation.json").write_text(...)

compare_training_scales([(root.name, root / "dataset.jsonl") for root in roots], folds=3)
```

## Dataset sizes

| run | records | positive | negative | holdout | train positive | train negative | validation positive | validation negative | holdout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| samples1 | 77 | 28 | 42 | 7 | 21 | 32 | 7 | 10 | 7 |
| samples3 | 231 | 84 | 126 | 21 | 63 | 94 | 21 | 32 | 21 |

Augmentation mix:

| run | original | speed | gain | reverb | background_noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| samples1 | 11 | 22 | 22 | 11 | 11 |
| samples3 | 33 | 66 | 66 | 33 | 33 |

## Results

| run | sample_count | accuracy_mean | accuracy_std | precision | recall | FRR | FAR/sample | false rejects | false accepts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| samples1 | 77 | 0.572 | 0.082 | 0.429 | 0.536 | 0.464 | 0.408 | 13 | 20 |
| samples3 | 231 | 0.506 | 0.087 | 0.361 | 0.464 | 0.536 | 0.469 | 45 | 69 |

Output files:

- `training/pipeline/runs/eval/hai_stackchan_ja_samples1_20260512_093259/artifacts/metrics/cross-validation.json`
- `training/pipeline/runs/eval/hai_stackchan_ja_samples3_20260512_093311/artifacts/metrics/cross-validation.json`
- `training/pipeline/runs/eval/cross_validation_scale_comparison.json`

These are generated artifacts and stay out of Git.

## Interpretation

Increasing `samples_per_variant` from 1 to 3 increased the generated dataset from 77 to 231 records, but this lightweight proxy accuracy dropped from about 0.57 to about 0.51. Precision/recall also decreased.

That means the current synthetic data expansion does not automatically improve separability under this simple acoustic-feature classifier. The extra length-scale variants and augmentations likely broaden both positive and non-positive distributions enough that the handcrafted features are no longer discriminative.

This should not be read as final wake-word quality. It is a smoke signal that the next useful measurements should use the real microWakeWord streaming model outputs, plus a larger and more varied negative/holdout set.

## Next measurement improvements

1. Add true microWakeWord inference scoring for validation / holdout WAVs instead of nearest-centroid proxy features.
2. Keep `positive/testing` as a real split rather than mirroring validation positives once enough samples exist.
3. Add per-augmentation and per-phrase breakdowns to identify which synthetic variants cause false accepts / false rejects.
4. Evaluate production-like FAR/hour with timestamped continuous background audio before claiming quality.
