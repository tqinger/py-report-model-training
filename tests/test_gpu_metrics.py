import json
import subprocess
from types import SimpleNamespace

import pytest

from tcm_qwen_eval.gpu_metrics import GPUMetricsCallback, query_gpu_metrics


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
    assert rows[0]["gpus"] == [{"gpu_index": 0, "temperature_c": 61}]


def test_gpu_metrics_callback_records_unavailable_telemetry(tmp_path, monkeypatch: pytest.MonkeyPatch):
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
