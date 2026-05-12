# expanded-5k no-slide real microWakeWord evaluation (2026-05-12)

## Scope

This run uses the generated `expanded-5k` synthetic dataset and evaluates with real upstream microWakeWord streaming `.tflite` inference only. No proxy metrics are used.

## Root cause

The previous feature generator created sliding training/validation spectrogram windows by shortening each clip by up to 9 frames before upstream fixed-length padding/truncation. For the current mixednet configuration, upstream computes a 194-frame model input (`clip_duration_ms: 1500` plus 47 dropped receptive-field frames). Many positive clips are near or below that input length, so the staged training examples were often padded tail-aligned examples that did not match the full streaming chunks used during `.tflite` inference.

The real upstream config did contain both positive and negative feature roots with correct `truth` values. RaggedMmap feature statistics were non-constant, and `FeatureHandler` saw expected label counts. The failure was not missing negative data.

## Real training / export

Source run:

```text
training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805
```

Experiment run:

```text
training/pipeline/runs/eval/no_slide_steps500_20260512_codex
```

Training settings:

```text
model: mixednet
steps: 500
batch_size: 64
feature staging: one full upstream spectrogram per WAV, no local sliding_window_view
```

The generated training/export artifacts are ignored and must not be committed.

## Real evaluation

Evaluator:

```text
suburi_wakeword.evaluation.evaluate_microwakeword_manifest
upstream microwakeword.inference.Model
quantized stream_state_internal_quant.tflite
```

Evaluation split filter:

```text
validation + holdout
```

Evaluated samples:

```text
1,848 total = 504 positive + 1,344 non-positive
```

Threshold sweep:

| threshold | accuracy | precision | recall | FRR | FAR/sample | false rejects | false accepts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.442 | 0.328 | 1.000 | 0.0000 | 0.7671 | 0 | 1,031 |
| 0.85 | 0.743 | 0.515 | 0.996 | 0.0040 | 0.3519 | 2 | 473 |
| 0.90 | 0.826 | 0.612 | 0.990 | 0.0099 | 0.2359 | 5 | 317 |
| 0.95 | 0.887 | 0.722 | 0.954 | 0.0456 | 0.1376 | 23 | 185 |

The real streaming `.tflite` estimate exceeds the 0.70 target at thresholds `0.85`, `0.90`, and `0.95`. The model manifest default cutoff was raised to `0.90` so future handoff artifacts do not keep the too-permissive `0.50` operating point.

## Commands run

```bash
cd training/pipeline
PYTHONPATH=src:.local-data/tools/micro-wake-word \
  .local-data/tools/micro-wake-word/.venv/bin/python <codex experiment script>
```

The script built a fresh training plan with `MicroWakeWordTrainingConfig(training_steps=(500,), batch_size=64)`, generated upstream RaggedMmap features, trained/exported upstream microWakeWord, copied the real `.tflite` handoff artifact, and wrote:

```text
runs/eval/no_slide_steps500_20260512_codex/artifacts/metrics/real-microwakeword-evaluation.json
```

