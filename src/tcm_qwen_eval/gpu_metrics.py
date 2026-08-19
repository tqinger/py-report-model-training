"""GPU telemetry collection for resumable QLoRA training runs."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
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
CPU_TEMPERATURE_SENSOR_KEYWORDS = (
    "coretemp",
    "cpu",
    "k10temp",
    "zenpower",
    "soc_thermal",
)
CPU_TEMPERATURE_LABEL_KEYWORDS = ("ccd", "core", "cpu", "package", "tctl", "tdie")


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


def _cpu_temperatures_celsius() -> list[dict[str, str | float | None]]:
    """Return CPU temperatures exposed by the operating system, when available."""
    sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
    if sensors_temperatures is None:
        return []

    try:
        sensor_groups = sensors_temperatures(fahrenheit=False)
    except (OSError, NotImplementedError, psutil.Error):
        return []

    temperatures: list[dict[str, str | float | None]] = []
    for sensor_name, readings in sensor_groups.items():
        normalized_sensor_name = sensor_name.lower()
        for reading in readings:
            label = reading.label or None
            normalized_label = (label or "").lower()
            if not (
                any(keyword in normalized_sensor_name for keyword in CPU_TEMPERATURE_SENSOR_KEYWORDS)
                or any(keyword in normalized_label for keyword in CPU_TEMPERATURE_LABEL_KEYWORDS)
            ):
                continue
            temperatures.append(
                {
                    "sensor": sensor_name,
                    "label": label,
                    "current_c": reading.current,
                    "high_c": reading.high,
                    "critical_c": reading.critical,
                }
            )
    return temperatures


def query_cpu_metrics(process: psutil.Process | None = None) -> dict[str, Any]:
    """Return CPU utilization, clock, memory, and available temperature telemetry."""
    training_process = process or psutil.Process()
    frequency = psutil.cpu_freq()
    memory = psutil.virtual_memory()
    return {
        "system_utilization_percent": psutil.cpu_percent(interval=None),
        "process_utilization_percent": training_process.cpu_percent(interval=None),
        "frequency_current_mhz": frequency.current if frequency else None,
        "system_memory_utilization_percent": memory.percent,
        "system_memory_used_mib": round(memory.used / 1024**2, 2),
        "system_memory_total_mib": round(memory.total / 1024**2, 2),
        "temperatures_c": _cpu_temperatures_celsius(),
    }


class GPUMetricsCallback(TrainerCallback):
    """Append GPU and CPU telemetry for each Trainer logging step."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "gpu_metrics.jsonl"
        self._last_logged_step: int | None = None
        self._process: psutil.Process | None = None
        try:
            process = psutil.Process()
            # psutil calculates CPU utilization between consecutive calls. Prime it at
            # callback construction so the first logging step is meaningful as well.
            psutil.cpu_percent(interval=None)
            process.cpu_percent(interval=None)
            self._process = process
        except (OSError, psutil.Error):
            # CPU telemetry is optional and must not prevent training from starting.
            pass

    def _write_metrics(self, state: Any) -> None:
        step = int(state.global_step)
        if step == self._last_logged_step:
            return

        payload: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "global_step": step,
            "epoch": state.epoch,
        }
        errors: list[str] = []
        try:
            payload["cpu"] = query_cpu_metrics(self._process)
        except (OSError, psutil.Error) as error:
            payload["cpu"] = {}
            errors.append(f"CPU telemetry: {error}")
        try:
            payload["gpus"] = query_gpu_metrics()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # Telemetry must never make an otherwise valid training run fail.
            payload["gpus"] = []
            errors.append(str(error))
        if errors:
            payload["error"] = "; ".join(errors)

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
