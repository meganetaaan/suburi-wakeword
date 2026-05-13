# 2026-05-13 lexical-negative profile

## Motivation

The previous `expanded-5k-hard-negatives` profile treated phrases like `ハイ、スタックチャンネルを開いて` as hard negatives. That lowered FAR/sample, but it also hurt recall. One reason is that `ハイスタックチャンネル` / `ハイ、スタックチャンネル` sits close to an acceptable wake expression: it starts with the same wake prefix and may be a natural misrecognition or continuation.

This profile instead tests lexical impostors that should remain clearly non-wake while still being acoustically close to `スタックちゃん`.

User-proposed examples included:

- `スタッキーちゃん`
- `スタッフさん`
- `スタックあんちゃん`

## Dataset profile

New profile:

- `expanded-5k-lexical-negatives`

It keeps the existing `expanded-5k` positives, baseline negatives, and English holdout, then adds six lexical/context negatives:

- `ハイ、スタッキーちゃん`
- `はい、スタッキーちゃん`
- `ハイ、スタックあんちゃん`
- `はい、スタックあんちゃん`
- `スタックちゃんのとなり`
- `スタックちゃんではありません`

Note: `ハイ、スタッフさん` / `はい、スタッフさん` already exist in the expanded baseline negative set, so this profile does not duplicate them.

## Dry-run counts

Command:

```bash
cd training/pipeline
PYTHONPATH=src uv run python -m suburi_wakeword.run_smoke_pipeline \
  --dataset-profile expanded-5k-lexical-negatives \
  --voice-profile-config config/voice-profiles.example.json \
  --dry-run \
  --output-root runs/dry-run/hai_stackchan_ja_expanded5k_lexical_negatives_20260513
```

Result:

| profile | base jobs | positive | negative | holdout | estimated utterances after raw+augmentation |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded-5k | 912 | 288 | 576 | 48 | 6,384 |
| expanded-5k-lexical-negatives | 1,056 | 288 | 720 | 48 | 7,392 |
| expanded-5k-hard-negatives | 1,104 | 288 | 768 | 48 | 7,728 |

## Expected comparison

This profile is deliberately lighter than `expanded-5k-hard-negatives`:

- fewer added negatives: 6 instead of 8;
- avoids adding `ハイ、スタックチャンネルを開いて` and similar prefix+continuation phrases;
- keeps negatives focused on lexical substitutions or inserted/context words.

The next real evaluation should compare:

1. `expanded-5k + negative_class_weight=1.25` — current operating candidate;
2. `expanded-5k-lexical-negatives + negative_class_weight=1.0`;
3. `expanded-5k-lexical-negatives + negative_class_weight=1.25`.

Use real 2-fold microWakeWord `.tflite` streaming inference only. Include lower thresholds around `0.875–0.950` because the previous hard-negative profile needed lower thresholds to recover recall.

## Decision rule

Adopt only if the lexical profile lowers FAR/sample while keeping recall near the current operating candidate. A good target would be materially below FAR/sample `0.204` while staying close to recall `0.785` at some threshold. If recall falls like the hard-negative profile, keep `expanded-5k + negative_class_weight=1.25` and rebalance again.
