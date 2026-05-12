from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from suburi_wakeword.false_accept_analysis import summarize_false_accepts_by_text


DEFAULT_THRESHOLD = 0.965


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze_cv_false_accepts(cv_root: Path, *, threshold: float = DEFAULT_THRESHOLD, top_n: int = 30) -> dict:
    records: list[dict] = []
    sample_scores: list[dict] = []
    for fold_manifest in sorted(cv_root.glob("fold*/heldout_dataset.jsonl")):
        records.extend(_read_jsonl(fold_manifest))
    for metrics_path in sorted(cv_root.glob("fold*/artifacts/metrics/real-microwakeword-cv-evaluation.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        sample_scores.extend(report.get("sample_scores", []))

    rows = summarize_false_accepts_by_text(records, sample_scores, threshold=threshold)
    total_non_positive = sum(int(row["sample_count"]) for row in rows)
    total_false_accepts = sum(int(row["false_accepts"]) for row in rows)
    return {
        "note": "False-accept analysis from real microWakeWord streaming .tflite CV sample scores; no proxy metrics.",
        "cv_root": str(cv_root),
        "threshold": float(threshold),
        "non_positive_samples": total_non_positive,
        "false_accepts": total_false_accepts,
        "far_per_sample": total_false_accepts / max(1, total_non_positive),
        "top_n": int(top_n),
        "top_false_accept_texts": rows[:top_n],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze real microWakeWord CV false accepts by source text.")
    parser.add_argument("cv_root", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyze_cv_false_accepts(args.cv_root, threshold=args.threshold, top_n=args.top_n)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
