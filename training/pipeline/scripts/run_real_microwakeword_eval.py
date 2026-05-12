from __future__ import annotations

import json
import shutil
from pathlib import Path

from suburi_wakeword.evaluation import evaluate_microwakeword_manifest
from suburi_wakeword.microwakeword import MicroWakeWordTrainingConfig, train_microwakeword_model
from suburi_wakeword.threshold_sweep import default_thresholds


ROOTS = [
    ("samples1", Path("runs/eval/hai_stackchan_ja_samples1_20260512_093259")),
    ("samples3", Path("runs/eval/hai_stackchan_ja_samples3_20260512_093311")),
]
SOURCE_DIR = Path(".local-data/tools/micro-wake-word").resolve()
SPLITS = ("validation", "holdout")
THRESHOLDS = default_thresholds(0.05, 0.95, 0.05)
CONFIG = MicroWakeWordTrainingConfig(training_steps=(2,), batch_size=4)


def main() -> None:
    rows = []
    for scale, root in ROOTS:
        manifest = root / "dataset.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        print(f"== {scale}: train/export real microWakeWord ==", flush=True)
        for stale in (
            root / "artifacts" / "microwakeword-model",
            root / "artifacts" / "model",
        ):
            if stale.exists():
                shutil.rmtree(stale)
        train_microwakeword_model(
            manifest,
            root,
            CONFIG,
            source_dir=SOURCE_DIR,
            model_architecture="mixednet",
        )
        model = root / "artifacts" / "model" / "stream_state_internal_quant.tflite"
        print(f"== {scale}: evaluate {model} ==", flush=True)
        report = evaluate_microwakeword_manifest(
            manifest,
            model,
            threshold=0.5,
            thresholds=THRESHOLDS,
            source_dir=SOURCE_DIR,
            splits=SPLITS,
        )
        out = root / "artifacts" / "metrics" / "real-microwakeword-evaluation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "scale": scale,
            "evaluation": report["evaluation"],
            "manifest": report["manifest"],
            "model": report["model"],
            "threshold": report["threshold"],
            "evaluated_splits": report["evaluated_splits"],
            "sample_count": report["sample_count"],
            "positive_samples": report["positive_samples"],
            "negative_samples": report["negative_samples"],
            "accuracy": report["accuracy"],
            "precision": report["precision"],
            "recall": report["recall"],
            "frr": report["frr"],
            "far_per_sample": report["far_per_sample"],
            "false_rejects_total": report["false_rejects_total"],
            "false_accepts_total": report["false_accepts_total"],
        })
    summary = {
        "evaluation": "real_microwakeword_tflite_streaming_scale_comparison",
        "note": "Real upstream microWakeWord .tflite streaming inference on validation/holdout WAVs only; no proxy metrics.",
        "splits": list(SPLITS),
        "rows": rows,
    }
    out = Path("runs/eval/real_microwakeword_scale_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
