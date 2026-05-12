from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .evaluation import evaluate_microwakeword_manifest
from .microwakeword import MicroWakeWordTrainingConfig, train_microwakeword_model
from .threshold_sweep import default_thresholds


@dataclass(frozen=True)
class RealEvalExperiment:
    name: str
    source_root: Path
    run_root: Path
    training_steps: tuple[int, ...]


def build_training_step_experiments(
    roots: Iterable[tuple[str, Path]],
    *,
    training_steps: Sequence[int],
) -> list[RealEvalExperiment]:
    experiments: list[RealEvalExperiment] = []
    for scale, root in roots:
        for steps in training_steps:
            experiments.append(
                RealEvalExperiment(
                    name=f"{scale}_steps{steps}",
                    source_root=Path(root),
                    run_root=Path(root) / "artifacts" / "real-eval" / f"steps{steps}",
                    training_steps=(int(steps),),
                )
            )
    return experiments


def run_experiments(
    experiments: Sequence[RealEvalExperiment],
    *,
    source_dir: Path,
    splits: Sequence[str] = ("validation", "holdout"),
    thresholds: Sequence[float] | None = None,
    threshold: float = 0.5,
    batch_size: int = 4,
) -> dict:
    rows = []
    for experiment in experiments:
        manifest = experiment.source_root / "dataset.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        if experiment.run_root.exists():
            shutil.rmtree(experiment.run_root)
        experiment.run_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, experiment.run_root / "dataset.jsonl")

        print(
            f"== {experiment.name}: train/export real microWakeWord steps={experiment.training_steps} ==",
            flush=True,
        )
        train_microwakeword_model(
            experiment.run_root / "dataset.jsonl",
            experiment.run_root,
            MicroWakeWordTrainingConfig(training_steps=experiment.training_steps, batch_size=batch_size),
            source_dir=source_dir,
            model_architecture="mixednet",
        )
        model = experiment.run_root / "artifacts" / "model" / "stream_state_internal_quant.tflite"
        print(f"== {experiment.name}: evaluate {model} ==", flush=True)
        report = evaluate_microwakeword_manifest(
            experiment.run_root / "dataset.jsonl",
            model,
            threshold=threshold,
            thresholds=thresholds or default_thresholds(0.05, 0.95, 0.05),
            source_dir=source_dir,
            splits=splits,
        )
        metrics_path = experiment.run_root / "artifacts" / "metrics" / "real-microwakeword-evaluation.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "experiment": experiment.name,
            "source_root": str(experiment.source_root),
            "run_root": str(experiment.run_root),
            "training_steps": list(experiment.training_steps),
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
    return {
        "evaluation": "real_microwakeword_tflite_streaming_training_step_comparison",
        "note": "Real upstream microWakeWord .tflite streaming inference on validation/holdout WAVs only; no proxy metrics.",
        "splits": list(splits),
        "rows": rows,
    }


def write_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
