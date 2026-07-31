# 训练启动命令

以下命令均应在项目根目录 `D:\shanjiyun\py-report-model-training` 的 PowerShell 中运行。前台训练需要设置 `PYTHONPATH`；后台启动器会自动处理该环境变量。

## 首次准备

```powershell
uv sync --group dev
uv run python scripts/download_qwen3_weights.py --size 4B
```

若权重尚未下载，也可在前台命令末尾添加 `--allow-download`，或在后台命令末尾添加 `-AllowDownload`。

## 原始舌象数据

数据源：`data/tongue-analysis`（训练命令的数据根目录为 `data`）

### 前台启动

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train_tongue_qlora.py `
  --config configs/tongue_qlora.toml `
  --data-dir data `
  --output-dir artifacts/qwen3-4b-tongue-qlora
```

### 后台启动

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora.toml `
  -DataDir data `
  -OutputDir artifacts/qwen3-4b-tongue-qlora
```

## 全组合对话数据

数据源：`data/conversations`；每个组合固定按 `r01`–`r08` 训练、`r09` 验证、`r10` 测试。

### 前台启动

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train_tongue_qlora.py `
  --config configs/tongue_qlora_conversations.toml `
  --data-dir data/conversations `
  --output-dir artifacts/qwen3-4b-tongue-conversations-qlora
```

### 后台启动

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-qlora
```

## 冒烟测试数据

先准备 100 条冒烟数据（10 个完整组合）：

```powershell
uv run python scripts/prepare_conversations_smoke_data.py
```

### 前台启动

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train_tongue_qlora.py `
  --config configs/tongue_qlora_conversations.toml `
  --data-dir data/smoke/conversations `
  --output-dir artifacts/qwen3-4b-tongue-conversations-smoke
```

### 后台启动

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/smoke/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-smoke
```

## 后台日志与恢复训练

启动器会输出 PID，并在 `artifacts/logs/` 写入同一时间戳的 `.out.log` 和 `.err.log`。实时查看训练进度：

```powershell
Get-Content -LiteralPath artifacts\logs\tongue_qlora_<timestamp>.out.log -Tail 50 -Wait
```

从 checkpoint 恢复全组合训练：

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-qlora `
  -ResumeFromCheckpoint artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-<step>
```
