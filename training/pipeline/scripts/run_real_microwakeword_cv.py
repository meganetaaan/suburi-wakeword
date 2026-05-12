from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from suburi_wakeword.evaluation import evaluate_microwakeword_manifest
from suburi_wakeword.microwakeword import MicroWakeWordTrainingConfig, train_microwakeword_model


DEFAULT_MANIFEST = Path("runs/smoke/hai_stackchan_ja_expanded5k_20260512_154805/dataset.jsonl")
DEFAULT_OUTPUT = Path("runs/eval/real_microwakeword_cv_expanded5k_2fold_20260513_codex")
DEFAULT_SOURCE_DIR = Path(".local-data/tools/micro-wake-word")
DEFAULT_THRESHOLDS = (0.90, 0.95)


@dataclass(frozen=True)
class Fold:
    index: int
    train_group_keys: set[str]
    eval_group_keys: set[str]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _label_class(record: dict) -> str:
    return "positive" if record.get("label") == "positive" else "negative"


def _group_key(record: dict) -> str:
    source_sample_id = record.get("source_sample_id") or record.get("sample_id")
    return f"{_label_class(record)}:{source_sample_id}"


def _stable_digest(value: str, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _select_group_keys(records: Sequence[dict], *, seed: str, max_positive_groups: int | None, max_negative_groups: int | None) -> set[str]:
    by_class: dict[str, set[str]] = {"positive": set(), "negative": set()}
    for record in records:
        by_class[_label_class(record)].add(_group_key(record))

    selected: set[str] = set()
    limits = {"positive": max_positive_groups, "negative": max_negative_groups}
    for label_class, keys in by_class.items():
        ordered = sorted(keys, key=lambda key: _stable_digest(key, seed=seed))
        limit = limits[label_class]
        selected.update(ordered[:limit] if limit is not None else ordered)
    return selected


def _build_folds(records: Sequence[dict], *, folds: int, seed: str, selected_group_keys: set[str]) -> list[Fold]:
    by_class: dict[str, list[str]] = {"positive": [], "negative": []}
    for record in records:
        key = _group_key(record)
        if key in selected_group_keys and key not in by_class[_label_class(record)]:
            by_class[_label_class(record)].append(key)

    fold_eval_keys = [set() for _ in range(folds)]
    for label_class, keys in by_class.items():
        ordered = sorted(keys, key=lambda key: _stable_digest(key, seed=f"{seed}:fold:{label_class}"))
        for offset, key in enumerate(ordered):
            fold_eval_keys[offset % folds].add(key)

    return [
        Fold(
            index=index,
            train_group_keys=selected_group_keys - fold_eval_keys[index],
            eval_group_keys=fold_eval_keys[index],
        )
        for index in range(folds)
    ]


def _mark_training_splits(records: Sequence[dict], *, fold: Fold, validation_fraction: float, seed: str) -> list[dict]:
    train_keys = sorted(fold.train_group_keys)
    by_class: dict[str, list[str]] = {"positive": [], "negative": []}
    for key in train_keys:
        by_class[key.split(":", 1)[0]].append(key)

    validation_keys: set[str] = set()
    for label_class, keys in by_class.items():
        ordered = sorted(keys, key=lambda key: _stable_digest(key, seed=f"{seed}:fold:{fold.index}:validation:{label_class}"))
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_keys.update(ordered[:validation_count])

    output = []
    for record in records:
        key = _group_key(record)
        if key not in fold.train_group_keys:
            continue
        split = "validation" if key in validation_keys else "train"
        output.append({**record, "split": split})
    return output


def _mark_eval_splits(records: Sequence[dict], *, fold: Fold) -> list[dict]:
    split = f"cv_fold_{fold.index + 1}"
    return [{**record, "split": split} for record in records if _group_key(record) in fold.eval_group_keys]


def _counts(records: Sequence[dict]) -> dict[str, int]:
    counts = {
        "sample_count": len(records),
        "positive_samples": sum(1 for record in records if _label_class(record) == "positive"),
        "negative_samples": sum(1 for record in records if _label_class(record) == "negative"),
        "english_positive_samples": sum(
            1 for record in records if _label_class(record) == "positive" and record.get("language") == "en"
        ),
    }
    return counts


def _threshold_metrics(report: dict, thresholds: Sequence[float]) -> list[dict]:
    by_threshold = {float(row["threshold"]): row for row in report.get("threshold_sweep", [])}
    rows = []
    for threshold in thresholds:
        row = by_threshold[float(threshold)]
        positives = int(row["positive_samples"])
        negatives = int(row["negative_samples"])
        false_rejects = int(row["false_rejects"])
        false_accepts = int(row["false_accepts"])
        true_positives = positives - false_rejects
        true_negatives = negatives - false_accepts
        sample_count = positives + negatives
        rows.append(
            {
                "threshold": float(threshold),
                "sample_count": sample_count,
                "positive_samples": positives,
                "negative_samples": negatives,
                "accuracy": (true_positives + true_negatives) / max(1, sample_count),
                "precision": true_positives / max(1, true_positives + false_accepts),
                "recall": true_positives / max(1, positives),
                "frr": false_rejects / max(1, positives),
                "far_per_sample": false_accepts / max(1, negatives),
                "false_rejects_total": false_rejects,
                "false_accepts_total": false_accepts,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
            }
        )
    return rows


def _mean_metrics(rows: Sequence[dict], thresholds: Sequence[float]) -> list[dict]:
    output = []
    for threshold in thresholds:
        selected = [row for row in rows if float(row["threshold"]) == float(threshold)]
        metric_names = ("accuracy", "precision", "recall", "frr", "far_per_sample")
        mean_row = {
            "threshold": float(threshold),
            "fold_count": len(selected),
            "sample_count": sum(int(row["sample_count"]) for row in selected),
            "positive_samples": sum(int(row["positive_samples"]) for row in selected),
            "negative_samples": sum(int(row["negative_samples"]) for row in selected),
            "false_rejects_total": sum(int(row["false_rejects_total"]) for row in selected),
            "false_accepts_total": sum(int(row["false_accepts_total"]) for row in selected),
        }
        for name in metric_names:
            mean_row[f"mean_{name}"] = sum(float(row[name]) for row in selected) / max(1, len(selected))
        output.append(mean_row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Literal real microWakeWord cross-validation using trained .tflite models.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", default="suburi-real-microwakeword-cv-v1")
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--negative-class-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    parser.add_argument("--max-positive-groups", type=int, default=None)
    parser.add_argument("--max-negative-groups", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-existing", action="store_true", help="Reuse an existing fold .tflite export instead of retraining it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    thresholds = tuple(args.thresholds or DEFAULT_THRESHOLDS)
    manifest_path = args.manifest
    output_root = args.output
    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume_existing:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it or --resume-existing to reuse fold exports")
    output_root.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(manifest_path)
    selected_group_keys = _select_group_keys(
        records,
        seed=args.seed,
        max_positive_groups=args.max_positive_groups,
        max_negative_groups=args.max_negative_groups,
    )
    selected_records = [record for record in records if _group_key(record) in selected_group_keys]
    folds = _build_folds(records, folds=args.folds, seed=args.seed, selected_group_keys=selected_group_keys)

    fold_rows = []
    for fold in folds:
        fold_root = output_root / f"fold{fold.index + 1}"
        fold_root.mkdir(parents=True, exist_ok=True)
        train_records = _mark_training_splits(selected_records, fold=fold, validation_fraction=args.validation_fraction, seed=args.seed)
        eval_records = _mark_eval_splits(selected_records, fold=fold)
        train_manifest = fold_root / "train_dataset.jsonl"
        eval_manifest = fold_root / "heldout_dataset.jsonl"
        _write_jsonl(train_manifest, train_records)
        _write_jsonl(eval_manifest, eval_records)

        model = fold_root / "artifacts" / "model" / "stream_state_internal_quant.tflite"
        if args.resume_existing and model.exists():
            print(f"== fold {fold.index + 1}/{args.folds}: reuse existing real microWakeWord export {model} ==", flush=True)
        else:
            print(
                f"== fold {fold.index + 1}/{args.folds}: train/export real microWakeWord "
                f"steps={args.training_steps} train={_counts(train_records)} eval={_counts(eval_records)} ==",
                flush=True,
            )
            train_microwakeword_model(
                train_manifest,
                fold_root,
                MicroWakeWordTrainingConfig(
                    training_steps=(int(args.training_steps),),
                    batch_size=int(args.batch_size),
                    positive_class_weight=float(args.positive_class_weight),
                    negative_class_weight=float(args.negative_class_weight),
                ),
                source_dir=args.source_dir.resolve(),
                model_architecture="mixednet",
            )
        report = evaluate_microwakeword_manifest(
            eval_manifest,
            model,
            threshold=thresholds[0],
            thresholds=thresholds,
            source_dir=args.source_dir.resolve(),
            splits=(f"cv_fold_{fold.index + 1}",),
        )
        metrics_path = fold_root / "artifacts" / "metrics" / "real-microwakeword-cv-evaluation.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        threshold_rows = _threshold_metrics(report, thresholds)
        fold_summary = {
            "fold": fold.index + 1,
            "train_manifest": str(train_manifest),
            "eval_manifest": str(eval_manifest),
            "model": str(model),
            "train_counts": _counts(train_records),
            "eval_counts": _counts(eval_records),
            "threshold_metrics": threshold_rows,
        }
        fold_rows.append(fold_summary)
        print(json.dumps(fold_summary, ensure_ascii=False, indent=2), flush=True)

    flat_rows = [row for fold in fold_rows for row in fold["threshold_metrics"]]
    summary = {
        "evaluation": "real_microwakeword_literal_cross_validation_tflite_streaming",
        "note": (
            "Each fold trains and exports an upstream microWakeWord .tflite from the fold training portion, "
            "then evaluates held-out fold WAVs with microwakeword.inference.Model.predict_clip. "
            "No proxy metrics, handcrafted features, nearest-centroid scoring, or lightweight CV are used."
        ),
        "source_manifest": str(manifest_path),
        "source_dir": str(args.source_dir.resolve()),
        "output_root": str(output_root),
        "folds": args.folds,
        "seed": args.seed,
        "training_steps": args.training_steps,
        "batch_size": args.batch_size,
        "positive_class_weight": args.positive_class_weight,
        "negative_class_weight": args.negative_class_weight,
        "validation_fraction_within_training_portion": args.validation_fraction,
        "selected_counts": _counts(selected_records),
        "folds_detail": fold_rows,
        "mean_threshold_metrics": _mean_metrics(flat_rows, thresholds),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
