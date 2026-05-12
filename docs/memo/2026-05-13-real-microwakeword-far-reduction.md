# 2026-05-13 real microWakeWord FAR reduction pilot

## Result

This run tested a class-weight-only FAR reduction pilot after the real 2-fold CV baseline. It is still real upstream microWakeWord training/export plus real quantized streaming `.tflite` inference; no proxy metrics are used.

Changing only `negative_class_weight` from `1.0` to `2.0` reduced false accepts, but it also reduced recall sharply. This is useful as a direction check, not yet a production setting.

## Artifacts

Baseline run:

- `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/summary.json`
- false-accept analysis: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/false_accept_analysis_threshold_0_965.json`

Negative-weight pilot:

- run root: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw2_20260513/`
- summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw2_20260513/summary.json`
- false-accept analysis: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw2_20260513/false_accept_analysis_threshold_0_965.json`

## Baseline false-accept concentration

At threshold `0.965`, baseline FAR/sample was `0.3363` with 1,469 false accepts out of 4,368 non-positive held-out samples.

The worst false-accept source texts were concentrated around phrases that include `スタックチャンネル` or `スタックちゃん` plus extra words:

| text | false accepts | samples | FA rate |
| --- | ---: | ---: | ---: |
| はい、スタックチャンネル | 146 | 168 | 0.869 |
| ハイ、スタックチャンネル | 126 | 168 | 0.750 |
| ねえ、スタックチャンネルを開いて | 105 | 168 | 0.625 |
| スタックちゃんと呼びました | 104 | 168 | 0.619 |
| ハイ、スタックはどこですか | 89 | 168 | 0.530 |
| スタックチャンネルです | 86 | 168 | 0.512 |
| スタックちゃん、こんにちは | 85 | 168 | 0.506 |
| スタックチャンネル | 85 | 168 | 0.506 |

Interpretation: the current model is not mainly failing on English holdout. It is over-accepting Japanese near-misses that contain the wake phrase prefix/body plus an extra suffix or sentence context.

## Negative class weight pilot

Command:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache \
  .local-data/tools/micro-wake-word/.venv/bin/python \
  scripts/run_real_microwakeword_cv.py \
  --manifest runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805/dataset.jsonl \
  --output runs/eval/real_microwakeword_cv_expanded5k_2fold_negw2_20260513 \
  --source-dir .local-data/tools/micro-wake-word \
  --folds 2 \
  --training-steps 500 \
  --batch-size 16 \
  --negative-class-weight 2.0 \
  --threshold 0.955 \
  --threshold 0.96 \
  --threshold 0.965 \
  --threshold 0.97 \
  --threshold 0.98 \
  --resume-existing
```

Note: the first attempt used `uv run python`, which failed during scoring because that environment did not include upstream `ai_edge_litert`. The successful run used `.local-data/tools/micro-wake-word/.venv/bin/python`.

## Baseline vs negative weight

| run | threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline weight 1.0 | 0.955 | 0.703 | 0.903 | 0.389 | 1698 | 196 |
| baseline weight 1.0 | 0.960 | 0.710 | 0.895 | 0.375 | 1639 | 211 |
| baseline weight 1.0 | 0.965 | 0.729 | 0.869 | 0.336 | 1469 | 264 |
| negative weight 2.0 | 0.955 | 0.718 | 0.645 | 0.248 | 1082 | 716 |
| negative weight 2.0 | 0.960 | 0.722 | 0.615 | 0.229 | 1000 | 777 |
| negative weight 2.0 | 0.965 | 0.729 | 0.564 | 0.195 | 852 | 879 |
| negative weight 2.0 | 0.970 | 0.732 | 0.544 | 0.182 | 793 | 919 |
| negative weight 2.0 | 0.980 | 0.744 | 0.467 | 0.128 | 558 | 1075 |

## Interpretation

`negative_class_weight=2.0` cuts FAR/sample substantially at the same threshold:

- threshold `0.965`: `0.336 -> 0.195`
- false accepts: `1469 -> 852`

But recall drops too much:

- threshold `0.965`: `0.869 -> 0.564`
- false rejects: `264 -> 879`

So class weighting is a real lever, but `2.0` is too aggressive for the current dataset/training setup.

## Recommended next run

Run a narrower class-weight sweep before changing the dataset again:

- `negative_class_weight=1.25`
- `negative_class_weight=1.5`
- thresholds around `0.955` to `0.975`

Target is not max accuracy. Prefer the operating point that lowers FAR/sample while keeping recall reasonably high, then use false-accept analysis to add/minimize the worst Japanese near-miss clusters.
