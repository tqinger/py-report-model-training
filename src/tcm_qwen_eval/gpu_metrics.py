"""GPU telemetry collection for resumable QLoRA training runs."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

GPU_QUERY_FIELDS = (
    "index",
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "fan.speed",
    "clocks.current.sm",
    "clocks.current.memory",
)
GPU_METRIC_FIELDS = (
    "gpu_index",
    "name",
    "temperature_c",
    "utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "power_limit_w",
    "fan_speed_percent",
    "sm_clock_mhz",
    "memory_clock_mhz",
)


def _numeric_value(value: str) -> int | float | str | None:
    """Convert nvidia-smi numeric output while retaining unavailable values."""
    normalized = value.strip()
    if normalized in {"", "N/A", "[N/A]"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return normalized


def query_gpu_metrics() -> list[dict[str, Any]]:
    """Return current telemetry for every GPU reported by ``nvidia-smi``."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")

    metrics: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(GPU_METRIC_FIELDS):
            raise RuntimeError(
                "nvidia-smi returned an unexpected number of telemetry columns: "
                f"expected {len(GPU_METRIC_FIELDS)}, got {len(row)}"
            )
        metrics.append(
            {
                field: value.strip() if field == "name" else _numeric_value(value)
                for field, value in zip(GPU_METRIC_FIELDS, row, strict=True)
            }
        )
    return metrics


class GPUMetricsCallback(TrainerCallback):
    """Append one GPU telemetry JSON record for each Trainer logging step."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "gpu_metrics.jsonl"
        self._last_logged_step: int | None = None

    def _write_metrics(self, state: Any) -> None:
        step = int(state.global_step)
        if step == self._last_logged_step:
            return

        payload: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "global_step": step,
            "epoch": state.epoch,
        }
        try:
            payload["gpus"] = query_gpu_metrics()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # Telemetry must never make an otherwise valid training run fail.
            payload["gpus"] = []
            payload["error"] = str(error)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._last_logged_step = step

    def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if getattr(state, "is_world_process_zero", True):
            self._write_metrics(state)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if getattr(state, "is_world_process_zero", True):
            self._write_metrics(state)
        return control
