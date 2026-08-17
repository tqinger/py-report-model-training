"""Utilities for finding and validating resumable Trainer checkpoints."""

from __future__ import annotations

import re
from pathlib import Path

LATEST_CHECKPOINT = "latest"
_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")
_TRAINING_STATE_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def resolve_resume_checkpoint(resume_from_checkpoint: str | None, output_dir: Path) -> str | None:
    """Resolve and validate an optional explicit or latest Trainer checkpoint."""
    if resume_from_checkpoint is None:
        return None

    if resume_from_checkpoint == LATEST_CHECKPOINT:
        candidates = (
            path
            for path in output_dir.iterdir()
            if path.is_dir() and _CHECKPOINT_NAME.fullmatch(path.name)
        )
        try:
            checkpoint_path = max(candidates, key=lambda path: int(path.name.removeprefix("checkpoint-")))
        except ValueError as error:
            raise SystemExit(f"No checkpoint found in output directory: {output_dir}") from error
    else:
        checkpoint_path = Path(resume_from_checkpoint)

    if not checkpoint_path.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint_path}")
    missing_files = [
        filename for filename in _TRAINING_STATE_FILES if not (checkpoint_path / filename).is_file()
    ]
    if not (checkpoint_path / "adapter_config.json").is_file():
        missing_files.append("adapter_config.json")
    if not any((checkpoint_path / filename).is_file() for filename in _ADAPTER_WEIGHT_FILES):
        missing_files.append("adapter model weights")
    if missing_files:
        raise SystemExit(
            "Checkpoint is not recoverable because required files are missing "
            f"({', '.join(missing_files)}): {checkpoint_path}"
        )
    return str(checkpoint_path)
