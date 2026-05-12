# expanded-5k real microWakeWord evaluation (2026-05-12)

## Scope

This run uses the generated `expanded-5k` synthetic dataset and evaluates with real upstream microWakeWord streaming `.tflite` inference only. No proxy metrics are used.

## Dataset

Source run:

```text
training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805
```

Generated records:

```text
raw WAV:        912
normalized WAV: 912
augmented WAV:  5,472
dataset.jsonl:  6,384
```

Dataset labels:

```text
positive: 2,016
negative: 4,032
holdout:    336
```

Splits:

```text
train:      4,536
validation: 1,512
holdout:      336
```

Axes:

```text
voice profiles:    4
prosody variants:  6
unique sample ids: 6,384 / 6,384
```

## Real training / export

Command shape used through `RealEvalExperiment` / `run_experiments`:

```text
source_root: runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805
run_root:    runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805/artifacts/real-eval/steps100
steps:       100
batch_size:  32
model:       mixednet
```

Generated model:

```text
runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805/artifacts/real-eval/steps100/artifacts/model/stream_state_internal_quant.tflite
```

## Real evaluation

Evaluation split filter:

```text
validation + holdout
```

Evaluated samples:

```text
1,848 total = 504 positive + 1,344 non-positive
```

Results at threshold `0.5`:

| metric | value |
|---|---:|
| accuracy | 0.297 |
| precision | 0.278 |
| recall | 0.984 |
| FRR | 0.0159 |
| FAR/sample | 0.9606 |
| false rejects | 8 |
| false accepts | 1,291 |

Threshold sweep summary:

| selection | threshold | FRR | FAR/sample | false rejects | false accepts |
|---|---:|---:|---:|---:|---:|
| threshold 0.5 | 0.50 | 0.0159 | 0.9606 | 8 | 1,291 |
| min FAR then FRR | 0.55 | 1.0000 | 0.0000 | 504 | 0 |

Score ranges were still very narrow:

```text
all:      0.4980392157 - 0.5019607843
positive: 0.4980392157 - 0.5019607843
negative: 0.4980392157 - 0.5019607843
```

## Interpretation

The expanded dataset and 100 training steps moved the model slightly away from the earlier constant-over-accepting behavior, but it is still not discriminative. At threshold `0.5`, recall is high, but nearly all non-positive validation/holdout samples are accepted.

The next useful work is likely not more synthetic utterances alone. We should inspect the upstream training configuration / model capacity and class weighting, then run a less-minimal training configuration while keeping this same expanded dataset and real `.tflite` evaluation path.

Generated artifacts remain ignored and are not committed.
