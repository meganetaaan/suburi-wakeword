# 2026-05-13 real microWakeWord cross-validation

## Result

Literal real microWakeWord cross-validation reached mean held-out accuracy >= 0.70.

This is not the earlier validation+holdout single-split result. Each fold trained and exported a real upstream microWakeWord `.tflite`, then evaluated the held-out fold WAVs with upstream `microwakeword.inference.Model.predict_clip`.

No proxy metrics, handcrafted features, nearest-centroid scoring, or lightweight CV were used.

## Artifact

- Run root: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/`
- Summary: `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/summary.json`
- Fold metrics:
  - `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/fold1/artifacts/metrics/real-microwakeword-cv-evaluation.json`
  - `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/fold2/artifacts/metrics/real-microwakeword-cv-evaluation.json`
- Fold models:
  - `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/fold1/artifacts/model/stream_state_internal_quant.tflite`
  - `training/pipeline/runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex/fold2/artifacts/model/stream_state_internal_quant.tflite`

## Design

- Script: `training/pipeline/scripts/run_real_microwakeword_cv.py`
- Source manifest: `training/pipeline/runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805/dataset.jsonl`
- Selected records: all 6,384 expanded-5k records
- Fold count: 2
- Fold assignment: deterministic SHA-256 order, stratified by positive vs non-positive class and grouped by `source_sample_id` when present so augmented variants stay in the same fold
- Per fold training portion: 3,192 records
- Per fold held-out evaluation portion: 3,192 records
- Per fold class counts: 1,008 positives, 2,184 non-positives
- Internal upstream validation: 15% of each fold's training groups only
- Training steps: 500
- Batch size: 16
- Positive language check: `english_positive_samples = 0`

The held-out fold manifest is separate from the training manifest, so held-out records are not staged into upstream training, validation, or export-time testing features for that fold.

## Metrics

| Threshold | Fold | Accuracy | Precision | Recall | FRR | FAR/sample | False rejects | False accepts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.900 | 1 | 0.6388 | 0.4654 | 0.9663 | 0.0337 | 0.5124 | 34 | 1119 |
| 0.900 | 2 | 0.5699 | 0.4219 | 0.9782 | 0.0218 | 0.6186 | 22 | 1351 |
| 0.900 | mean | 0.6043 | 0.4436 | 0.9722 | 0.0278 | 0.5655 | 56 | 2470 |
| 0.950 | 1 | 0.7033 | 0.5168 | 0.9315 | 0.0685 | 0.4020 | 69 | 878 |
| 0.950 | 2 | 0.6805 | 0.4967 | 0.9058 | 0.0942 | 0.4235 | 95 | 925 |
| 0.950 | mean | 0.6919 | 0.5068 | 0.9187 | 0.0813 | 0.4128 | 164 | 1803 |
| 0.955 | 1 | 0.7168 | 0.5297 | 0.9206 | 0.0794 | 0.3773 | 80 | 824 |
| 0.955 | 2 | 0.6898 | 0.5051 | 0.8849 | 0.1151 | 0.4002 | 116 | 874 |
| 0.955 | mean | 0.7033 | 0.5174 | 0.9028 | 0.0972 | 0.3887 | 196 | 1698 |
| 0.960 | 1 | 0.7306 | 0.5441 | 0.9058 | 0.0942 | 0.3503 | 95 | 765 |
| 0.960 | 2 | 0.6898 | 0.5051 | 0.8849 | 0.1151 | 0.4002 | 116 | 874 |
| 0.960 | mean | 0.7102 | 0.5246 | 0.8953 | 0.1047 | 0.3752 | 211 | 1639 |
| 0.965 | 1 | 0.7437 | 0.5589 | 0.8938 | 0.1062 | 0.3255 | 107 | 711 |
| 0.965 | 2 | 0.7133 | 0.5289 | 0.8442 | 0.1558 | 0.3471 | 157 | 758 |
| 0.965 | mean | 0.7285 | 0.5439 | 0.8690 | 0.1310 | 0.3363 | 264 | 1469 |

## Commands

Initial full run trained fold 1, then failed at scoring because the pipeline `uv` environment did not include upstream `ai_edge_litert`. The fold 1 model was already exported. The final successful run used the upstream microWakeWord virtualenv for scoring and `UV_CACHE_DIR=/tmp/uv-cache` for child `uv` training/export commands:

```bash
cd training/pipeline
PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache \
  .local-data/tools/micro-wake-word/.venv/bin/python \
  scripts/run_real_microwakeword_cv.py \
  --folds 2 \
  --training-steps 500 \
  --threshold 0.90 \
  --threshold 0.95 \
  --threshold 0.955 \
  --threshold 0.96 \
  --threshold 0.965 \
  --output runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex \
  --resume-existing
```

