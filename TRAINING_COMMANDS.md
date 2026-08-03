# 训练启动命令

以下命令均应在项目根目录运行。Windows PowerShell 的前台训练需要设置 `PYTHONPATH`；其 `.ps1` 后台启动器会自动处理该环境变量。Linux Bash 不支持该 `.ps1` 启动器，应使用下文的 `nohup` 命令。

## 首次准备

```powershell
uv sync --group dev
uv run python scripts/download_qwen3_weights.py --size 4B
```

项目固定使用 PyTorch CUDA 12.8 wheel，可同时支持 Windows 与 Linux；RTX 50 系列服务器应在同步后确认 `torch.cuda.get_arch_list()` 包含 `sm_120`：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
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
PYTHONPATH=src uv run python scripts/train_tongue_qlora.py \
    --config configs/tongue_qlora_conversations.toml \
    --data-dir data/conversations \
    --output-dir artifacts/qwen3-4b-tongue-conversations-qlora


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

## Linux Bash 后台启动

Linux 上不要执行 `./scripts/start_tongue_qlora_training.ps1`。使用 `nohup` 启动后，终端关闭也不会中断训练；`$!` 是后台进程 PID。

### 原始舌象数据

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_tongue_qlora.py \
  --config configs/tongue_qlora.toml \
  --data-dir data \
  --output-dir artifacts/qwen3-4b-tongue-qlora \
  >"artifacts/logs/tongue_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/tongue_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
```

### 全组合对话数据

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_tongue_qlora.py \
  --config configs/tongue_qlora_conversations.toml \
  --data-dir data/conversations \
  --output-dir artifacts/qwen3-4b-tongue-conversations-qlora \
  >"artifacts/logs/tongue_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/tongue_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/tongue_qlora_${timestamp}.out.log"
```

### 冒烟测试数据

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_tongue_qlora.py \
  --config configs/tongue_qlora_conversations.toml \
  --data-dir data/smoke/conversations \
  --output-dir artifacts/qwen3-4b-tongue-conversations-smoke \
  >"artifacts/logs/tongue_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/tongue_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
```

## 后台日志与恢复训练

启动器会输出 PID，并在 `artifacts/logs/` 写入同一时间戳的 `.out.log` 和 `.err.log`。实时查看训练进度：

```powershell
Get-Content -LiteralPath artifacts\logs\tongue_qlora_<timestamp>.out.log -Tail 50 -Wait
```

Linux Bash 实时查看日志：

```bash
tail -f artifacts/logs/tongue_qlora_<timestamp>.out.log
```

从 checkpoint 恢复全组合训练：

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-qlora `
  -ResumeFromCheckpoint artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-<step>
```
