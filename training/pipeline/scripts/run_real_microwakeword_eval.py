from __future__ import annotations

import argparse
import json
from pathlib import Path

from suburi_wakeword.real_eval_runner import build_training_step_experiments, run_experiments, write_summary
from suburi_wakeword.threshold_sweep import default_thresholds


DEFAULT_ROOTS = [
    ("samples1", Path("runs/eval/hai_stackchan_ja_samples1_20260512_093259")),
    ("samples3", Path("runs/eval/hai_stackchan_ja_samples3_20260512_093311")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate real microWakeWord .tflite experiments.")
    parser.add_argument(
        "--scale",
        choices=("samples1", "samples3"),
        action="append",
        help="Dataset scale to evaluate. Defaults to both samples1 and samples3.",
    )
    parser.add_argument(
        "--training-step",
        type=int,
        action="append",
        dest="training_steps",
        help="Training step count to evaluate. Can be passed multiple times. Defaults to 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/eval/real_microwakeword_training_step_comparison.json"),
        help="Where to write the comparison JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.scale or [])
    roots = [(name, root) for name, root in DEFAULT_ROOTS if not selected or name in selected]
    experiments = build_training_step_experiments(roots, training_steps=args.training_steps or (2,))
    summary = run_experiments(
        experiments,
        source_dir=Path(".local-data/tools/micro-wake-word").resolve(),
        splits=("validation", "holdout"),
        thresholds=default_thresholds(0.05, 0.95, 0.05),
    )
    write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
