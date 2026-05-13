# 2026-05-13 expanded-5k hard-negative iteration

## Result

This run tested a targeted hard-negative dataset iteration after the `negative_class_weight=1.25` sweep. It is still real upstream microWakeWord training/export plus real quantized streaming `.tflite` inference; no proxy metrics are used.

The hard-negative profile reduced FAR/sample substantially, but it also reduced recall. This means the added near-miss negatives are useful signal, but the current mixture is too negative-heavy to adopt as-is.

## Dataset change

New profile:

- `expanded-5k-hard-negatives`

It keeps the expanded-5k positives and holdout set, then adds eight targeted Japanese near-miss negative phrases based on real false-accept analysis:

- `ねえ、スタックチャンネル`
- `スタックチャンネルを開いて`
- `ハイ、スタックチャンネルを開いて`
- `はい、スタックチャンネルを開いて`
- `スタックちゃんです`
- `ハイ、スタックちゃんと呼びました`
- `ハイ、スタックちゃん、こんにちは`
- `はい、スタックちゃん、こんにちは`

Dry-run/base counts with four voice profiles:

| profile | base jobs | positive | negative | holdout | estimated utterances after raw+augmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded-5k | 912 | 288 | 576 | 48 | 6,384 |
| expanded-5k-hard-negatives | 1,104 | 288 | 768 | 48 | 7,728 |

Generated manifest counts:

| label | samples |
| --- | ---: |
| positive | 2,016 |
| negative | 5,376 |
| holdout | 336 |

## Artifacts

Dataset:

- `training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_hard_negatives_20260513/dataset.jsonl`
- dry-run summary: `training/pipeline/runs/dry-run/hai_stackchan_ja_expanded5k_hard_negatives_20260513/dataset-summary.json`

Real CV run:

- run root: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_hard_negatives_2fold_negw125_20260513/`
- summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_hard_negatives_2fold_negw125_20260513/summary.json`
- false-accept analysis at threshold `0.900`: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_hard_negatives_2fold_negw125_20260513/false_accept_analysis_threshold_0_900.json`
- false-accept analysis at threshold `0.965`: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_hard_negatives_2fold_negw125_20260513/false_accept_analysis_threshold_0_965.json`

## Commands

Dataset generation:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-hard-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --output-root runs/smoke/hai_stackchan_ja_expanded5k_hard_negatives_20260513
```

Real `.tflite` CV:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache \
  .local-data/tools/micro-wake-word/.venv/bin/python \
  scripts/run_real_microwakeword_cv.py \
  --manifest runs/smoke/hai_stackchan_ja_expanded5k_hard_negatives_20260513/dataset.jsonl \
  --output runs/eval/real_microwakeword_cv_expanded5k_hard_negatives_2fold_negw125_20260513 \
  --source-dir .local-data/tools/micro-wake-word \
  --folds 2 \
  --training-steps 500 \
  --batch-size 16 \
  --negative-class-weight 1.25 \
  --threshold 0.900 \
  --threshold 0.925 \
  --threshold 0.950 \
  --threshold 0.955 \
  --threshold 0.960 \
  --threshold 0.965 \
  --threshold 0.970 \
  --threshold 0.975 \
  --resume-existing
```

## Baseline vs hard-negative iteration

The comparison below uses real 2-fold `.tflite` CV and `negative_class_weight=1.25`.

| run | threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expanded-5k | 0.955 | 0.772 | 0.830 | 0.255 | 1113 | 342 |
| expanded-5k | 0.960 | 0.780 | 0.822 | 0.239 | 1044 | 358 |
| expanded-5k | 0.965 | 0.793 | 0.785 | 0.204 | 889 | 433 |
| expanded-5k | 0.970 | 0.794 | 0.780 | 0.200 | 874 | 443 |
| expanded-5k | 0.975 | 0.798 | 0.753 | 0.181 | 792 | 497 |
| hard negatives | 0.900 | 0.806 | 0.759 | 0.177 | 1010 | 486 |
| hard negatives | 0.925 | 0.815 | 0.716 | 0.150 | 854 | 573 |
| hard negatives | 0.950 | 0.821 | 0.603 | 0.102 | 584 | 801 |
| hard negatives | 0.955 | 0.821 | 0.571 | 0.091 | 522 | 865 |
| hard negatives | 0.960 | 0.820 | 0.553 | 0.086 | 489 | 902 |
| hard negatives | 0.965 | 0.820 | 0.517 | 0.074 | 421 | 973 |
| hard negatives | 0.970 | 0.819 | 0.492 | 0.065 | 374 | 1024 |
| hard negatives | 0.975 | 0.820 | 0.475 | 0.058 | 330 | 1058 |

## Interpretation

Hard negatives clearly move the model toward rejecting near-misses:

- At threshold `0.965`, FAR/sample improves from `0.204` to `0.074`.
- False accepts improve from `889` to `421`.

But recall falls too much:

- At threshold `0.965`, recall drops from `0.785` to `0.517`.
- Even lowering threshold to `0.900` only reaches recall `0.759`, while FAR/sample is `0.177`.

So the dataset direction is useful, but this profile should not replace expanded-5k as-is.

A likely issue is class balance / hard-negative density: positives stayed at 2,016 samples, while non-positive samples increased from 4,368 to 5,712. The model may be learning a stricter decision boundary than desired.

## Remaining false accepts

At threshold `0.965`, the top remaining false accepts are still suffix/context near-misses:

| text | false accepts | samples | FA rate |
| --- | ---: | ---: | ---: |
| ハイ、スタックちゃんと呼びました | 49 | 168 | 0.292 |
| ハイ、スタックチャンネル | 37 | 168 | 0.220 |
| ハイ、スタックちゃん、こんにちは | 35 | 168 | 0.208 |
| スタックちゃんと呼びました | 33 | 168 | 0.196 |
| はい、スタックちゃん、こんにちは | 32 | 168 | 0.190 |
| スタックちゃん、こんにちは | 32 | 168 | 0.190 |
| ねえ、スタックチャンネル | 31 | 168 | 0.185 |
| ねえ、スタックチャンネルを開いて | 26 | 168 | 0.155 |

## Recommended next run

Do not keep increasing negative class weight or hard-negative count. Instead, rebalance the same hard-negative idea:

1. Keep the hard-negative phrases, but reduce their multiplicity, e.g. only selected prosody variants or only raw+lighter augmentation for targeted hard negatives.
2. Or add positive variants/prosody diversity to restore recall while keeping the hard negatives.
3. Re-run real 2-fold CV with `negative_class_weight=1.0` and `1.25`, including lower thresholds around `0.875–0.950`.

The current acceptance candidate remains the previous `expanded-5k + negative_class_weight=1.25` result until a balanced hard-negative profile preserves recall better.
