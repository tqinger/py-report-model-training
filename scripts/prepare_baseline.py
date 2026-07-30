from __future__ import annotations

import argparse
from pathlib import Path

from tcm_qwen_eval.dataset import (
    audit_examples,
    default_data_dir,
    grouped_split,
    load_examples,
    select_baseline_examples,
    write_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TCM SFT data and choose deterministic baseline samples.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_selection.json"))
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    examples = load_examples(args.data_dir)
    selected = select_baseline_examples(examples, args.samples_per_task, args.seed)
    write_selection(args.output, selected, grouped_split(examples, args.seed))
    print(f"Loaded {len(examples)} examples across {len({item.task for item in examples})} tasks.")
    print(f"Selected {len(selected)} baseline examples -> {args.output}")
    for row in audit_examples(examples):
        print(row)


if __name__ == "__main__":
    main()
