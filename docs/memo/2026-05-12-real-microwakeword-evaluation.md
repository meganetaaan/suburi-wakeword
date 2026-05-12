# real microWakeWord evaluation smoke (2026-05-12)

## Scope

This memo replaces the earlier lightweight cross-validation smoke for decision-making. The earlier result was proxy-only. From here on, wake word evaluation uses real upstream microWakeWord streaming `.tflite` inference only.

Evaluated splits:

- `validation`
- `holdout`

The English `Hi, Stack-chan` near-miss remains holdout/negative-side and is not mixed into positives.

## Commands

The scale comparison was run from `training/pipeline` with the ignored upstream checkout at `.local-data/tools/micro-wake-word`:

```bash
uv pip install --python .local-data/tools/micro-wake-word/.venv/bin/python 'numpy<2' tensorboard
PYTHONPATH=$PWD/src .local-data/tools/micro-wake-word/.venv/bin/python scripts/run_real_microwakeword_eval.py
```

The repository adapter invokes upstream feature generation and training as `uv run --no-sync ...` inside the upstream checkout, so the local smoke venv keeps the required `numpy<2` compatibility for upstream `np.trapz` calls.

## Artifacts

Per-run reports:

```text
training/pipeline/runs/eval/hai_stackchan_ja_samples1_20260512_093259/artifacts/metrics/real-microwakeword-evaluation.json
training/pipeline/runs/eval/hai_stackchan_ja_samples3_20260512_093311/artifacts/metrics/real-microwakeword-evaluation.json
```

Scale comparison:

```text
training/pipeline/runs/eval/real_microwakeword_scale_comparison.json
```

Generated `.tflite` models are ignored artifacts under each run's `artifacts/model/` directory and are not committed.

## Results at threshold 0.5

| scale | evaluated samples | positives | negatives | accuracy | precision | recall | FRR | FAR/sample | false rejects | false accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples1 | 24 | 7 | 17 | 0.2917 | 0.2917 | 1.0000 | 0.0000 | 1.0000 | 0 | 17 |
| samples3 | 74 | 21 | 53 | 0.2838 | 0.2838 | 1.0000 | 0.0000 | 1.0000 | 0 | 53 |

Interpretation: this minimal smoke training over-detects at threshold `0.5`: it catches every positive validation sample, but also accepts every validation/holdout negative. Increasing generated samples from 1x to 3x did not improve the real microWakeWord false accept behavior in this smoke configuration.

## Next work

- Treat this as a real inference smoke, not a production model quality claim.
- Improve negative/holdout diversity and training steps before judging threshold quality.
- Re-run the same real `.tflite` evaluator after each dataset/training change; do not reintroduce proxy CV metrics.
