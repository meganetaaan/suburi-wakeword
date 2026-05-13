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
| negative weight 1.25 | 0.955 | 0.772 | 0.830 | 0.255 | 1113 | 342 |
| negative weight 1.25 | 0.960 | 0.780 | 0.822 | 0.239 | 1044 | 358 |
| negative weight 1.25 | 0.965 | 0.793 | 0.785 | 0.204 | 889 | 433 |
| negative weight 1.25 | 0.970 | 0.794 | 0.780 | 0.200 | 874 | 443 |
| negative weight 1.25 | 0.975 | 0.798 | 0.753 | 0.181 | 792 | 497 |
| negative weight 1.5 | 0.955 | 0.830 | 0.584 | 0.057 | 249 | 838 |
| negative weight 1.5 | 0.960 | 0.822 | 0.550 | 0.052 | 227 | 908 |
| negative weight 1.5 | 0.965 | 0.814 | 0.502 | 0.042 | 185 | 1004 |
| negative weight 1.5 | 0.970 | 0.805 | 0.460 | 0.036 | 158 | 1088 |
| negative weight 1.5 | 0.975 | 0.793 | 0.409 | 0.030 | 130 | 1192 |
| negative weight 2.0 | 0.955 | 0.718 | 0.645 | 0.248 | 1082 | 716 |
| negative weight 2.0 | 0.960 | 0.722 | 0.615 | 0.229 | 1000 | 777 |
| negative weight 2.0 | 0.965 | 0.729 | 0.564 | 0.195 | 852 | 879 |
| negative weight 2.0 | 0.970 | 0.732 | 0.544 | 0.182 | 793 | 919 |
| negative weight 2.0 | 0.980 | 0.744 | 0.467 | 0.128 | 558 | 1075 |

## Narrow sweep artifacts

Additional real `.tflite` CV runs:

- `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw125_20260513/summary.json`
- `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw125_20260513/false_accept_analysis_threshold_0_965.json`
- `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw15_20260513/summary.json`
- `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_negw15_20260513/false_accept_analysis_threshold_0_965.json`

At threshold `0.965`, `negative_class_weight=1.25` reduced FAR/sample from `0.336` to `0.204` while recall stayed at `0.785`. This is a much better trade-off than `2.0` for the current dataset. `negative_class_weight=1.5` reduced FAR/sample further to `0.042`, but recall fell to `0.502`, so it is also too aggressive if wake-word recall matters.

The best current operating candidate is therefore `negative_class_weight=1.25`, with threshold `0.965` or `0.970` depending on whether the priority is recall (`0.785`) or slightly lower FAR/sample (`0.200`). Threshold `0.975` lowers FAR/sample to `0.181`, but recall drops further to `0.753`.

## Narrow sweep false-accept concentration

At threshold `0.965`, `negative_class_weight=1.25` still leaves the same Japanese near-miss clusters on top:

| text | false accepts | samples | FA rate |
| --- | ---: | ---: | ---: |
| はい、スタックチャンネル | 104 | 168 | 0.619 |
| ハイ、スタックチャンネル | 84 | 168 | 0.500 |
| スタックちゃん、こんにちは | 66 | 168 | 0.393 |
| スタックちゃんと呼びました | 61 | 168 | 0.363 |
| ねえ、スタックチャンネルを開いて | 56 | 168 | 0.333 |

`negative_class_weight=1.5` sharply suppresses most of these, but the remaining top miss is still `ねえ、スタックチャンネルを開いて` with `60/168` false accepts. That suggests class weighting helps, but dataset/content work should target suffix/context near-misses rather than English holdout first.

## Interpretation

Class weighting is a real lever, but the useful range is narrow:

- `1.25` is the first plausible FAR/recall trade-off.
- `1.5` and `2.0` are too aggressive for recall, even though they lower false accepts.
- The next improvement should focus on the highest Japanese near-miss clusters that survive at `negative_class_weight=1.25`.

## Recommended next run

The targeted hard-negative follow-up is recorded separately in `docs/memo/2026-05-13-expanded5k-hard-negative-iteration.md`. It reduced FAR/sample but hurt recall, so do not keep adding negative weight or hard-negative count. User feedback: `ハイスタックチャンネル` is close to acceptable, so the next safer experiment is the lexical-impostor profile in `docs/memo/2026-05-13-lexical-negative-profile.md` (`スタッキーちゃん`, `スタッフさん`, `スタックあんちゃん`, inserted/context words). Use `expanded-5k + negative_class_weight=1.25` as the current operating candidate until a rebalanced profile preserves recall better.

Target is not max accuracy. Prefer the operating point that lowers FAR/sample while keeping recall reasonably high, then use false-accept analysis to add/minimize the worst Japanese near-miss clusters.
