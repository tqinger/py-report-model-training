import json
import subprocess
from types import SimpleNamespace

import pytest

from tcm_qwen_eval.gpu_metrics import (
    GPUMetricsCallback,
    query_cpu_metrics,
    query_gpu_metrics,
)


def test_query_cpu_metrics_records_system_and_process_utilization(
    monkeypatch: pytest.MonkeyPatch,
):
    class Process:
        def cpu_percent(self, *, interval: None) -> float:
            assert interval is None
            return 12.5

    class Frequency:
        current = 4500.0

    class Memory:
        percent = 62.5
        used = 8 * 1024**3
        total = 16 * 1024**3

    class Temperature:
        label = "Package id 0"
        current = 75.0
        high = 95.0
        critical = 100.0

    class UnrelatedTemperature:
        label = "Composite"
        current = 40.0
        high = None
        critical = None

    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.psutil.cpu_percent",
        lambda *, interval: 37.5,
    )
    monkeypatch.setattr("tcm_qwen_eval.gpu_metrics.psutil.cpu_freq", lambda: Frequency())
    monkeypatch.setattr("tcm_qwen_eval.gpu_metrics.psutil.virtual_memory", lambda: Memory())
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.psutil.sensors_temperatures",
        lambda *, fahrenheit: {
            "coretemp": [Temperature()],
            "nvme": [UnrelatedTemperature()],
        },
        raising=False,
    )

    assert query_cpu_metrics(Process()) == {
        "system_utilization_percent": 37.5,
        "process_utilization_percent": 12.5,
        "frequency_current_mhz": 4500.0,
        "system_memory_utilization_percent": 62.5,
        "system_memory_used_mib": 8192.0,
        "system_memory_total_mib": 16384.0,
        "temperatures_c": [
            {
                "sensor": "coretemp",
                "label": "Package id 0",
                "current_c": 75.0,
                "high_c": 95.0,
                "critical_c": 100.0,
            }
        ],
    }


def test_query_gpu_metrics_parses_nvidia_smi_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            stdout="0, NVIDIA RTX, 61, 94, 1234, 16384, 251.50, 320.00, 42, 2100, 9501\n",
        ),
    )

    assert query_gpu_metrics() == [
        {
            "gpu_index": 0,
            "name": "NVIDIA RTX",
            "temperature_c": 61,
            "utilization_percent": 94,
            "memory_used_mib": 1234,
            "memory_total_mib": 16384,
            "power_draw_w": 251.5,
            "power_limit_w": 320.0,
            "fan_speed_percent": 42,
            "sm_clock_mhz": 2100,
            "memory_clock_mhz": 9501,
        }
    ]


def test_gpu_metrics_callback_appends_and_deduplicates_steps(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_cpu_metrics",
        lambda process: {
            "system_utilization_percent": 37.5,
            "process_utilization_percent": 12.5,
            "frequency_current_mhz": 4500.0,
            "system_memory_utilization_percent": 62.5,
            "system_memory_used_mib": 8192.0,
            "system_memory_total_mib": 16384.0,
            "temperatures_c": [],
        },
    )
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_gpu_metrics",
        lambda: [{"gpu_index": 0, "temperature_c": 61}],
    )
    callback = GPUMetricsCallback(tmp_path)
    state = SimpleNamespace(global_step=50, epoch=0.5, is_world_process_zero=True)
    control = object()

    assert callback.on_log(None, state, control) is control
    assert callback.on_train_end(None, state, control) is control

    rows = [json.loads(line) for line in (tmp_path / "gpu_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["global_step"] == 50
    assert rows[0]["cpu"] == {
        "system_utilization_percent": 37.5,
        "process_utilization_percent": 12.5,
        "frequency_current_mhz": 4500.0,
        "system_memory_utilization_percent": 62.5,
        "system_memory_used_mib": 8192.0,
        "system_memory_total_mib": 16384.0,
        "temperatures_c": [],
    }
    assert rows[0]["gpus"] == [{"gpu_index": 0, "temperature_c": 61}]


def test_gpu_metrics_callback_records_unavailable_telemetry(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_cpu_metrics",
        lambda process: {
            "system_utilization_percent": 37.5,
            "process_utilization_percent": 12.5,
            "frequency_current_mhz": 4500.0,
            "system_memory_utilization_percent": 62.5,
            "system_memory_used_mib": 8192.0,
            "system_memory_total_mib": 16384.0,
            "temperatures_c": [],
        },
    )
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_gpu_metrics",
        lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    )
    callback = GPUMetricsCallback(tmp_path)
    state = SimpleNamespace(global_step=1, epoch=0.1, is_world_process_zero=True)

    callback.on_log(None, state, None)

    row = json.loads((tmp_path / "gpu_metrics.jsonl").read_text())
    assert row["gpus"] == []
    assert "nvidia-smi" in row["error"]


def test_gpu_metrics_callback_records_unavailable_cpu_telemetry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_cpu_metrics",
        lambda process: (_ for _ in ()).throw(OSError("CPU metrics unavailable")),
    )
    monkeypatch.setattr(
        "tcm_qwen_eval.gpu_metrics.query_gpu_metrics",
        lambda: [{"gpu_index": 0, "temperature_c": 61}],
    )
    callback = GPUMetricsCallback(tmp_path)
    state = SimpleNamespace(global_step=1, epoch=0.1, is_world_process_zero=True)

    callback.on_log(None, state, None)

    row = json.loads((tmp_path / "gpu_metrics.jsonl").read_text())
    assert row["cpu"] == {}
    assert row["gpus"] == [{"gpu_index": 0, "temperature_c": 61}]
    assert "CPU metrics unavailable" in row["error"]
