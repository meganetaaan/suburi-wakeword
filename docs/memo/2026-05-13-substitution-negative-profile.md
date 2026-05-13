# 2026-05-13 substitution-negative profile

## Motivation

The `expanded-5k-lexical-negatives` profile was safer than the first hard-negative profile, but its top remaining false accepts included the newly added full `スタックちゃん + suffix/context` negatives:

- `スタックちゃんのとなり`
- `スタックちゃんではありません`
- `スタックちゃん、こんにちは`

This profile tests an even lighter variant that keeps only clear lexical substitutions and avoids newly adding contiguous `スタックちゃん + suffix/context` negatives.

## Dataset profile

Profile:

- `expanded-5k-substitution-negatives`

It keeps the existing `expanded-5k` positives, baseline negatives, and English holdout, then adds only four substitution negatives:

- `ハイ、スタッキーちゃん`
- `はい、スタッキーちゃん`
- `ハイ、スタックあんちゃん`
- `はい、スタックあんちゃん`

Note: `ハイ、スタッフさん` / `はい、スタッフさん` already exist in the expanded baseline negative set.

It deliberately does not add:

- `スタックちゃんのとなり`
- `スタックちゃんではありません`
- `ハイ、スタックチャンネルを開いて`

## Dataset generation

Dry-run command:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-substitution-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --dry-run \
  --output-root runs/dry-run/hai_stackchan_ja_expanded5k_substitution_negatives_20260513
```

Full dataset command:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-substitution-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --output-root runs/smoke/hai_stackchan_ja_expanded5k_substitution_negatives_20260513
```

Counts:

| profile | base jobs | positive | negative | holdout | estimated utterances after raw+augmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded-5k | 912 | 288 | 576 | 48 | 6,384 |
| expanded-5k-substitution-negatives | 1,008 | 288 | 672 | 48 | 7,056 |
| expanded-5k-lexical-negatives | 1,056 | 288 | 720 | 48 | 7,392 |

Generated manifest counts:

| label | samples |
| --- | ---: |
| positive | 2,016 |
| negative | 4,704 |
| holdout | 336 |

## Real CV commands

Two real 2-fold microWakeWord `.tflite` CV runs were executed. Both use real upstream training/export and real streaming `.tflite` inference; no proxy metrics are used.

```bash
cd training/pipeline
for weight_tag in 10 125; do
  if [ "$weight_tag" = "10" ]; then weight="1.0"; else weight="1.25"; fi
  out="runs/eval/real_microwakeword_cv_expanded5k_substitution_negatives_2fold_negw${weight_tag}_20260513"
  PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache \
    .local-data/tools/micro-wake-word/.venv/bin/python \
    scripts/run_real_microwakeword_cv.py \
    --manifest runs/smoke/hai_stackchan_ja_expanded5k_substitution_negatives_20260513/dataset.jsonl \
    --output "$out" \
    --source-dir .local-data/tools/micro-wake-word \
    --folds 2 \
    --training-steps 500 \
    --batch-size 16 \
    --negative-class-weight "$weight" \
    --threshold 0.875 \
    --threshold 0.900 \
    --threshold 0.925 \
    --threshold 0.950 \
    --threshold 0.955 \
    --threshold 0.960 \
    --threshold 0.965 \
    --threshold 0.970 \
    --threshold 0.975 \
    --resume-existing
done
```

Artifacts:

- dataset: `training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_substitution_negatives_20260513/dataset.jsonl`
- negw1.0 summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_substitution_negatives_2fold_negw10_20260513/summary.json`
- negw1.25 summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_substitution_negatives_2fold_negw125_20260513/summary.json`
- negw1.25 threshold-0.875 false accepts: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_substitution_negatives_2fold_negw125_20260513/false_accept_analysis_threshold_0_875.json`

## Results

Comparison against the current operating candidate and the previous lexical run:

| run | threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expanded-5k + negw1.25 | 0.965 | 0.793 | 0.785 | 0.204 | 889 | 433 |
| lexical + negw1.25 | 0.955 | 0.798 | 0.785 | 0.197 | 1058 | 433 |
| substitution + negw1.0 | 0.875 | 0.802 | 0.560 | 0.101 | 508 | 888 |
| substitution + negw1.25 | 0.875 | 0.854 | 0.695 | 0.083 | 417 | 615 |
| substitution + negw1.25 | 0.900 | 0.853 | 0.658 | 0.069 | 347 | 690 |
| substitution + negw1.25 | 0.950 | 0.831 | 0.502 | 0.037 | 189 | 1003 |

Full threshold excerpt for substitution profiles:

| run | threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| substitution + negw1.0 | 0.875 | 0.802 | 0.560 | 0.101 | 508 | 888 |
| substitution + negw1.0 | 0.900 | 0.798 | 0.507 | 0.086 | 433 | 994 |
| substitution + negw1.0 | 0.925 | 0.789 | 0.433 | 0.068 | 345 | 1144 |
| substitution + negw1.0 | 0.950 | 0.782 | 0.353 | 0.046 | 230 | 1305 |
| substitution + negw1.25 | 0.875 | 0.854 | 0.695 | 0.083 | 417 | 615 |
| substitution + negw1.25 | 0.900 | 0.853 | 0.658 | 0.069 | 347 | 690 |
| substitution + negw1.25 | 0.925 | 0.844 | 0.593 | 0.055 | 278 | 821 |
| substitution + negw1.25 | 0.950 | 0.831 | 0.502 | 0.037 | 189 | 1003 |

## Interpretation

This profile strongly reduces false accepts, but it hurts recall too much:

- Best low-threshold substitution point (`negw1.25 @ 0.875`) has FAR/sample `0.083`, but recall only `0.695`.
- It does not reach the current operating candidate recall (`0.785`) anywhere in the tested threshold range.
- `negative_class_weight=1.0` is worse than `1.25` here: recall is even lower at comparable low thresholds.

The result is counter-intuitive because this profile adds fewer negatives than `expanded-5k-lexical-negatives`. It suggests that the random/held-out split and training dynamics are sensitive enough that simply removing suffix/context negatives does not restore recall. In this run, substitution-only negatives made the classifier conservative rather than better balanced.

## Remaining false accepts

At the best substitution operating point (`substitution + negw1.25`, threshold `0.875`), top remaining false accepts were:

| text | false accepts | samples | FA rate |
| --- | ---: | ---: | ---: |
| スタックちゃんと呼びました | 88 | 168 | 0.524 |
| はい、スタックチャンネル | 58 | 168 | 0.345 |
| スタックちゃん、こんにちは | 57 | 168 | 0.339 |
| ハイ、スタックチャンネル | 45 | 168 | 0.268 |
| スタックチャンネルです | 34 | 168 | 0.202 |
| ねえ、スタックチャンネルを開いて | 33 | 168 | 0.196 |

The persistent false accepts are mostly baseline expanded negatives, not the newly added `スタッキーちゃん` / `スタックあんちゃん` substitutions.

## Decision

Do not adopt `expanded-5k-substitution-negatives`.

Current operating candidate remains:

- `expanded-5k + negative_class_weight=1.25`, threshold around `0.965`

The best alternative remains the previous lexical profile as a small-but-not-decisive improvement at equal recall:

- `expanded-5k-lexical-negatives + negative_class_weight=1.25`, threshold around `0.955`

Next useful experiment:

1. Stop adding negatives for the moment.
2. Try positive-side recovery, such as `positive_class_weight=1.25` together with `negative_class_weight=1.25`, or add positive diversity, then compare equal-recall FAR.
3. Keep reporting only real `.tflite` streaming inference metrics.
