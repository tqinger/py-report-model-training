"""Create a small, reproducible conversations subset for an end-to-end smoke test."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

CONVERSATION_FILENAME = re.compile(r"tongue-(?P<combination>\d+)-r(?P<round>0[1-9]|10)-without")
EXPECTED_ROUNDS = set(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small, complete-per-combination conversations dataset for smoke tests."
    )
    parser.add_argument("--source", type=Path, default=Path("data/conversations"))
    parser.add_argument("--output", type=Path, default=Path("data/smoke/conversations"))
    parser.add_argument("--combinations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _group_files(source: Path) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(source.glob("*.json")):
        match = CONVERSATION_FILENAME.fullmatch(path.stem)
        if not match:
            raise ValueError(f"{path}: unexpected conversations filename")
        combination = match["combination"]
        round_number = int(match["round"])
        if round_number in groups[combination]:
            raise ValueError(f"{path}: duplicate r{round_number:02d} for combination {combination}")
        groups[combination][round_number] = path

    for combination, rounds in groups.items():
        if set(rounds) != EXPECTED_ROUNDS:
            raise ValueError(f"{combination}: expected exactly one file for every round r01 through r10")
    if not groups:
        raise ValueError(f"No conversation JSON files found in {source}")
    return groups


def prepare_smoke_data(source: Path, output: Path, combinations: int, seed: int) -> list[str]:
    if combinations <= 0:
        raise ValueError("combinations must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")

    groups = _group_files(source)
    if combinations > len(groups):
        raise ValueError(f"Requested {combinations} combinations, but source contains only {len(groups)}")
    selected = sorted(
        groups,
        key=lambda combination: hashlib.sha256(f"{seed}:{combination}".encode()).hexdigest(),
    )[:combinations]

    output.mkdir(parents=True, exist_ok=True)
    for combination in selected:
        for round_number in sorted(groups[combination]):
            source_path = groups[combination][round_number]
            shutil.copy2(source_path, output / source_path.name)
    return selected


def main() -> None:
    args = parse_args()
    selected = prepare_smoke_data(args.source, args.output, args.combinations, args.seed)
    print(f"Prepared {len(selected) * 10} samples from {len(selected)} combinations in {args.output}")
    print(f"Selected combinations: {', '.join(selected)}")


if __name__ == "__main__":
    main()
