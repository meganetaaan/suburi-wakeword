# real microWakeWord training-step sweep (2026-05-12)

## Scope

This experiment keeps evaluation real-only: upstream microWakeWord streaming `.tflite` inference on validation + holdout WAVs. No proxy metrics are used.

The previous real run showed the model was over-accepting at `threshold=0.5`. This sweep checks whether increasing the minimal smoke `training_steps` from `2` to `20` changes that behavior on the `samples3` dataset.

## Command

```bash
cd training/pipeline
PYTHONPATH=$PWD/src .local-data/tools/micro-wake-word/.venv/bin/python \
  scripts/run_real_microwakeword_eval.py \
  --scale samples3 \
  --training-step 2 \
  --training-step 20 \
  --output runs/eval/real_microwakeword_samples3_training_step_comparison.json
```

## Dataset

Source dataset:

```text
training/pipeline/runs/eval/hai_stackchan_ja_samples3_20260512_093311/dataset.jsonl
```

Evaluation split filter:

```text
validation + holdout
```

Evaluated samples:

```text
74 total = 21 positive + 53 non-positive
```

## Results at threshold 0.5

| experiment | training_steps | accuracy | precision | recall | FRR | FAR/sample | false rejects | false accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| samples3_steps2 | 2 | 0.284 | 0.284 | 1.000 | 0.000 | 1.000 | 0 | 53 |
| samples3_steps20 | 20 | 0.284 | 0.284 | 1.000 | 0.000 | 1.000 | 0 | 53 |

Score ranges:

| experiment | all scores | positive scores | non-positive scores |
|---|---|---|---|
| samples3_steps2 | 0.5019607843 - 0.5019607843 | 0.5019607843 - 0.5019607843 | 0.5019607843 - 0.5019607843 |
| samples3_steps20 | 0.5019607843 - 0.5058823529 | 0.5019607843 - 0.5019607843 | 0.5019607843 - 0.5058823529 |

## Interpretation

Increasing smoke training from 2 to 20 steps did not improve false accepts. The exported quantized streaming model still scores almost every validation/holdout clip at roughly the same value around `0.502`.

This suggests the current minimal synthetic setup is not yet producing a discriminative wake-word model. The next useful change is not simply more tiny training steps; it is to improve the training signal:

1. add stronger and more diverse hard negatives / background negatives,
2. run a less-minimal training configuration,
3. keep reporting only real `.tflite` threshold sweeps.

## Artifacts

Ignored generated artifacts:

```text
training/pipeline/runs/eval/real_microwakeword_samples3_training_step_comparison.json
training/pipeline/runs/eval/hai_stackchan_ja_samples3_20260512_093311/artifacts/real-eval/steps2/artifacts/metrics/real-microwakeword-evaluation.json
training/pipeline/runs/eval/hai_stackchan_ja_samples3_20260512_093311/artifacts/real-eval/steps20/artifacts/metrics/real-microwakeword-evaluation.json
```
