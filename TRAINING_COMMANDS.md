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

## 体质分析（Qwen3-4B）

数据源：`data/constitution-analysis`。训练入口会将每条 JSON 响应作为一个综合体质分析任务，并按体质辨识信息来源分组，以 80/10/10 切分训练、验证和测试集。

### Windows PowerShell 前台训练

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train_constitution_qlora.py `
  --config configs/constitution_qlora.toml `
  --data-dir data/constitution-analysis `
  --output-dir artifacts/qwen3-4b-constitution-qlora
```

### Windows PowerShell 后台训练

```powershell
.\scripts\start_constitution_qlora_training.ps1 `
  -Config configs/constitution_qlora.toml `
  -DataDir data/constitution-analysis `
  -OutputDir artifacts/qwen3-4b-constitution-qlora
```

### Ubuntu Bash 前台训练

```bash
PYTHONPATH=src uv run python scripts/train_constitution_qlora.py \
  --config configs/constitution_qlora.toml \
  --data-dir data/constitution-analysis \
  --output-dir artifacts/qwen3-4b-constitution-qlora
```

### Ubuntu Bash 后台训练

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_constitution_qlora.py \
  --config configs/constitution_qlora.toml \
  --data-dir data/constitution-analysis \
  --output-dir artifacts/qwen3-4b-constitution-qlora \
  >"artifacts/logs/constitution_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/constitution_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/constitution_qlora_${timestamp}.out.log"
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

### Qwen3-1.7B（后台启动并记录日志）

```powershell
uv run python scripts/download_qwen3_weights.py --size 1.7B

.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations_1_7b.toml `
  -DataDir data/conversations `
  -Model Qwen/Qwen3-1.7B `
  -OutputDir artifacts/qwen3-1.7b-tongue-conversations-qlora
```

启动器会将本次训练的标准输出和错误输出分别写入 `artifacts/logs/` 下带时间戳的 `.out.log`、`.err.log` 文件。

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

PYTHONPATH=src uv run python -u scripts/train_tongue_qlora.py \
    --config configs/tongue_qlora_conversations.toml \
    --data-dir data/conversations \
    --output-dir artifacts/qwen3-4b-tongue-conversations-qlora

### 舌象体质 50K

数据源：`data/tongue_constitution_50k`。该数据集已固定划分 `train.jsonl` 和 `val.jsonl`，训练入口会原样使用，不再重新切分。

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_tongue_constitution_qlora.py \
  --config configs/tongue_constitution_50k_qlora.toml \
  --data-dir data/tongue_constitution_50k \
  --output-dir artifacts/qwen3-4b-tongue-constitution-50k-qlora \
  >"artifacts/logs/tongue_constitution_50k_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/tongue_constitution_50k_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/tongue_constitution_50k_qlora_${timestamp}.out.log"
```

### 五态 20

数据源：`data/wutai_20`。该数据集已固定划分 `train.jsonl` 和 `val.jsonl`，训练入口会原样使用，不再重新切分。

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_wutai_qlora.py \
  --config configs/wutai_20_qlora.toml \
  --data-dir data/wutai_20 \
  --output-dir artifacts/qwen3-4b-wutai-20-qlora \
  >"artifacts/logs/wutai_20_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/wutai_20_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wutai_20_qlora_${timestamp}.out.log"
```

### 全诊报告 50K

数据源：`data/holistic_50k`。该数据集已固定划分 `train.jsonl` 和 `val.jsonl`，训练入口会原样使用，不再重新切分。

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_holistic_qlora.py \
  --config configs/holistic_50k_qlora.toml \
  --data-dir data/holistic_50k \
  --output-dir artifacts/qwen3-4b-holistic-50k-qlora \
  >"artifacts/logs/holistic_50k_qlora_${timestamp}.out.log" \
  2>"artifacts/logs/holistic_50k_qlora_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/holistic_50k_qlora_${timestamp}.out.log"
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

### Ubuntu：恢复预切分 JSONL 训练任务

恢复时必须沿用原来的 `--output-dir`；只传入 `--resume-from-checkpoint` 会自动选择其中最新的完整 checkpoint。以下命令分别恢复舌象体质、五态、全诊报告训练：

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_tongue_constitution_qlora.py \
  --config configs/tongue_constitution_50k_qlora.toml \
  --data-dir data/tongue_constitution_50k \
  --output-dir artifacts/qwen3-4b-tongue-constitution-50k-qlora \
  --resume-from-checkpoint \
  >"artifacts/logs/tongue_constitution_50k_resume_${timestamp}.out.log" \
  2>"artifacts/logs/tongue_constitution_50k_resume_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
```

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_wutai_qlora.py \
  --config configs/wutai_20_qlora.toml \
  --data-dir data/wutai_20 \
  --output-dir artifacts/qwen3-4b-wutai-20-qlora \
  --resume-from-checkpoint \
  >"artifacts/logs/wutai_20_resume_${timestamp}.out.log" \
  2>"artifacts/logs/wutai_20_resume_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
```

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train_holistic_qlora.py \
  --config configs/holistic_50k_qlora.toml \
  --data-dir data/holistic_50k \
  --output-dir artifacts/qwen3-4b-holistic-50k-qlora \
  --resume-from-checkpoint \
  >"artifacts/logs/holistic_50k_resume_${timestamp}.out.log" \
  2>"artifacts/logs/holistic_50k_resume_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
```

若要恢复指定 checkpoint，将无参数的 `--resume-from-checkpoint` 替换为例如：

```bash
--resume-from-checkpoint artifacts/qwen3-4b-wutai-20-qlora/checkpoint-500
```

从 checkpoint 恢复全组合训练：

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-qlora `
  -ResumeFromCheckpoint artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-<step>
```

`training.save_steps` 控制完整、可恢复 checkpoint 的保存间隔（优化器 step）。
`save_total_limit = 1` 会在保存新 checkpoint 时自动删除上一个；若最佳验证 checkpoint
不同，Trainer 会额外保留它。不需要填写 step 编号时，使用 `-ResumeLatest` 自动从最新
保留的 checkpoint 恢复：

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -Config configs/tongue_qlora_conversations.toml `
  -DataDir data/conversations `
  -OutputDir artifacts/qwen3-4b-tongue-conversations-qlora `
  -ResumeLatest
```
