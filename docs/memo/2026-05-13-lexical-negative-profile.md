# 2026-05-13 lexical-negative profile

## Motivation

The previous `expanded-5k-hard-negatives` profile treated phrases like `ハイ、スタックチャンネルを開いて` as hard negatives. That lowered FAR/sample, but it also hurt recall. One reason is that `ハイスタックチャンネル` / `ハイ、スタックチャンネル` sits close to an acceptable wake expression: it starts with the same wake prefix and may be a natural misrecognition or continuation.

This profile instead tests lexical impostors that should remain clearly non-wake while still being acoustically close to `スタックちゃん`.

User-proposed examples included:

- `スタッキーちゃん`
- `スタッフさん`
- `スタックあんちゃん`

## Dataset profile

Profile:

- `expanded-5k-lexical-negatives`

It keeps the existing `expanded-5k` positives, baseline negatives, and English holdout, then adds six lexical/context negatives:

- `ハイ、スタッキーちゃん`
- `はい、スタッキーちゃん`
- `ハイ、スタックあんちゃん`
- `はい、スタックあんちゃん`
- `スタックちゃんのとなり`
- `スタックちゃんではありません`

Note: `ハイ、スタッフさん` / `はい、スタッフさん` already exist in the expanded baseline negative set, so this profile does not duplicate them.

## Dataset generation

Dry-run command:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-lexical-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --dry-run \
  --output-root runs/dry-run/hai_stackchan_ja_expanded5k_lexical_negatives_20260513
```

Full dataset command:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-lexical-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --output-root runs/smoke/hai_stackchan_ja_expanded5k_lexical_negatives_20260513
```

Counts:

| profile | base jobs | positive | negative | holdout | estimated utterances after raw+augmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded-5k | 912 | 288 | 576 | 48 | 6,384 |
| expanded-5k-lexical-negatives | 1,056 | 288 | 720 | 48 | 7,392 |
| expanded-5k-hard-negatives | 1,104 | 288 | 768 | 48 | 7,728 |

Generated manifest counts:

| label | samples |
| --- | ---: |
| positive | 2,016 |
| negative | 5,040 |
| holdout | 336 |

## Real CV commands

Two real 2-fold microWakeWord `.tflite` CV runs were executed. Both use real upstream training/export and real streaming `.tflite` inference; no proxy metrics are used.

```bash
cd training/pipeline
for weight_tag in 10 125; do
  if [ "$weight_tag" = "10" ]; then weight="1.0"; else weight="1.25"; fi
  out="runs/eval/real_microwakeword_cv_expanded5k_lexical_negatives_2fold_negw${weight_tag}_20260513"
  PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache \
    .local-data/tools/micro-wake-word/.venv/bin/python \
    scripts/run_real_microwakeword_cv.py \
    --manifest runs/smoke/hai_stackchan_ja_expanded5k_lexical_negatives_20260513/dataset.jsonl \
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

- dataset: `training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_lexical_negatives_20260513/dataset.jsonl`
- negw1.0 summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_lexical_negatives_2fold_negw10_20260513/summary.json`
- negw1.25 summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_lexical_negatives_2fold_negw125_20260513/summary.json`
- negw1.25 threshold-0.955 false accepts: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_lexical_negatives_2fold_negw125_20260513/false_accept_analysis_threshold_0_955.json`

## Results

Current operating candidate for comparison:

| run | threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expanded-5k + negw1.25 | 0.955 | 0.772 | 0.830 | 0.255 | 1113 | 342 |
| expanded-5k + negw1.25 | 0.960 | 0.780 | 0.822 | 0.239 | 1044 | 358 |
| expanded-5k + negw1.25 | 0.965 | 0.793 | 0.785 | 0.204 | 889 | 433 |
| expanded-5k + negw1.25 | 0.975 | 0.798 | 0.753 | 0.181 | 792 | 497 |

Lexical profile with `negative_class_weight=1.0`:

| threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.925 | 0.695 | 0.893 | 0.379 | 2037 | 216 |
| 0.950 | 0.734 | 0.844 | 0.308 | 1655 | 314 |
| 0.955 | 0.741 | 0.831 | 0.294 | 1578 | 340 |
| 0.960 | 0.742 | 0.827 | 0.290 | 1558 | 349 |
| 0.965 | 0.759 | 0.800 | 0.256 | 1376 | 403 |
| 0.975 | 0.784 | 0.770 | 0.211 | 1132 | 464 |

Lexical profile with `negative_class_weight=1.25`:

| threshold | mean accuracy | mean recall | mean FAR/sample | false accepts | false rejects |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.925 | 0.771 | 0.847 | 0.258 | 1385 | 308 |
| 0.950 | 0.794 | 0.798 | 0.207 | 1114 | 408 |
| 0.955 | 0.798 | 0.785 | 0.197 | 1058 | 433 |
| 0.960 | 0.799 | 0.781 | 0.194 | 1043 | 442 |
| 0.965 | 0.807 | 0.748 | 0.171 | 918 | 509 |
| 0.975 | 0.810 | 0.714 | 0.153 | 825 | 576 |

## Interpretation

The lexical profile is better than the previous hard-negative profile, because it can preserve recall at the same level as the operating candidate:

- `expanded-5k + negw1.25 @ 0.965`: recall `0.785`, FAR/sample `0.204`
- `lexical + negw1.25 @ 0.955`: recall `0.785`, FAR/sample `0.197`

The improvement is real but small. It is not enough to call this a strong replacement yet:

- FAR/sample improves `0.204 -> 0.197` at the same recall.
- False accepts increase in absolute count (`889 -> 1058`) because the lexical dataset has more non-positive samples (`4,368 -> 5,376`). FAR/sample is the fairer comparison.
- At more aggressive thresholds (`0.965+`), FAR improves further, but recall drops below the current operating candidate.

`negative_class_weight=1.0` is not a good replacement: it keeps recall high but FAR/sample remains worse than the current operating candidate at comparable recall.

## Remaining false accepts

At the best same-recall point (`lexical + negw1.25`, threshold `0.955`), top remaining false accepts are:

| text | false accepts | samples | FA rate |
| --- | ---: | ---: | ---: |
| スタックちゃんのとなり | 113 | 168 | 0.673 |
| スタックちゃん、こんにちは | 90 | 168 | 0.536 |
| スタックちゃんではありません | 84 | 168 | 0.500 |
| ハイ、スタックチャンネル | 77 | 168 | 0.458 |
| スタックちゃんと呼びました | 66 | 168 | 0.393 |
| はい、スタックチャンネル | 62 | 168 | 0.369 |
| ねえ、スタックチャンネルを開いて | 51 | 168 | 0.304 |

The new lexical/context negatives are themselves now high false-accept items (`スタックちゃんのとなり`, `スタックちゃんではありません`). That suggests the model still keys strongly on the contiguous `スタックちゃん` token. The next iteration should avoid adding many more full `スタックちゃん + suffix` negatives without also improving positives/decision margin.

## Decision

Do not replace the operating candidate solely based on this run. The lexical profile is promising and much safer than the first hard-negative profile, but the gain is modest.

Current operating candidate remains:

- `expanded-5k + negative_class_weight=1.25`, threshold around `0.965`

Next useful experiment:

1. Try a lighter lexical profile that emphasizes true substitutions (`スタッキーちゃん`, `スタッフさん`, `スタックあんちゃん`) and reduces full `スタックちゃん + suffix` contexts.
2. Or keep the lexical profile but add positive diversity / positive weight to recover margin, then rerun real 2-fold CV.
3. Continue using real `.tflite` streaming inference and compare at equal-recall operating points.
