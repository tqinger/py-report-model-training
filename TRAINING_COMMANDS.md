# QLoRA 训练命令

所有命令均在项目根目录运行。配置文件只存放训练超参数；数据目录、模型缓存、输出目录和恢复 checkpoint 均通过命令行指定。

## 1. 环境准备

首次在 Windows PowerShell 或 Ubuntu Bash 环境中执行：

```bash
uv sync --group dev
uv run python scripts/maintenance/download_qwen3_weights.py --size 4B
```

项目使用 PyTorch CUDA 12.8 wheel。训练前确认 CUDA 可用，并检查 Ubuntu 服务器的架构列表包含实际 GPU 的计算能力：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

默认从 `artifacts/hf_cache` 离线读取模型。缓存缺失且允许从 Hugging Face 下载时，在任一训练命令末尾追加 `--allow-download`。

## 2. 康养 LoRA compact

数据根目录为 `data/wellness_lora_compact/export`。四项数据都已提供固定的 `train.jsonl` / `val.jsonl`，训练时不会重新切分。配置集中在 `configs/wellness_lora_compact/`。

| 任务 | 数据目录 | 训练入口 | Ubuntu 配置 | Windows 配置 |
| --- | --- | --- | --- | --- |
| 舌象康养 | `tongue` | `train_wellness_tongue_qlora.py` | `tongue_qlora.toml` | `tongue_qlora_windows.toml` |
| 舌象体质康养 | `tongue_constitution` | `train_tongue_constitution_qlora.py` | `tongue_constitution_qlora.toml` | `tongue_constitution_qlora_windows.toml` |
| 舌象体质康养计划 v2 | `tongue-constitution-plan-v2/export/tongue_constitution` | `train_tongue_constitution_qlora.py` | `tongue_constitution_plan_v2_qlora.toml` | `tongue_constitution_qlora_windows.toml` |
| 五态康养 | `wutai` | `train_wutai_qlora.py` | `wutai_qlora.toml` | `wutai_qlora_windows.toml` |
| 四诊康养 | `holistic` | `train_holistic_qlora.py` | `holistic_qlora.toml` | `holistic_qlora_windows.toml` |

Ubuntu 配置面向当前服务器配方；Windows 配置面向 RTX 4060 Ti 16GB，使用 `dataloader_num_workers = 0` 和较小批大小。四项任务都使用 Qwen3-4B，输出目录以任务名区分；不要用相同输出目录混合不同平台的训练状态。

### 2.1 Windows PowerShell

先执行一次 `$env:PYTHONPATH = "src"`；随后从下列命令中选择一个任务运行。

#### 舌象康养

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_wellness_tongue_qlora.py `
  --config configs/wellness_lora_compact/tongue_qlora_windows.toml `
  --data-dir data/wellness_lora_compact/export/tongue `
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-qlora-windows
```

#### 舌象体质康养

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_tongue_constitution_qlora.py `
  --config configs/wellness_lora_compact/tongue_constitution_qlora_windows.toml `
  --data-dir data/wellness_lora_compact/export/tongue_constitution `
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-constitution-qlora-windows
```

#### 舌象体质康养计划 v2

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_tongue_constitution_qlora.py `
  --config configs/wellness_lora_compact/tongue_constitution_qlora_windows.toml `
  --data-dir data/wellness_lora_compact/tongue-constitution-plan-v2/export/tongue_constitution `
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-constitution-plan-v2-qlora-windows
```

#### 五态康养

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_wutai_qlora.py `
  --config configs/wellness_lora_compact/wutai_qlora_windows.toml `
  --data-dir data/wellness_lora_compact/export/wutai `
  --output-dir artifacts/qwen3-4b-wellness-compact-wutai-qlora-windows
```

#### 四诊康养

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_holistic_qlora.py `
  --config configs/wellness_lora_compact/holistic_qlora_windows.toml `
  --data-dir data/wellness_lora_compact/export/holistic `
  --output-dir artifacts/qwen3-4b-wellness-compact-holistic-qlora-windows
```

### 2.2 Ubuntu Bash

以下命令均以 `nohup` 在后台启动，并将标准输出和错误输出分开写入 `artifacts/logs/`。若要前台运行，移除 `nohup`、重定向、`</dev/null &` 和其后的 PID 输出即可。

#### 舌象康养

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train/train_wellness_tongue_qlora.py \
  --config configs/wellness_lora_compact/tongue_qlora.toml \
  --data-dir data/wellness_lora_compact/export/tongue \
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-qlora \
  >"artifacts/logs/wellness_compact_tongue_${timestamp}.out.log" \
  2>"artifacts/logs/wellness_compact_tongue_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wellness_compact_tongue_${timestamp}.out.log"
```

#### 舌象体质康养

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train/train_tongue_constitution_qlora.py \
  --config configs/wellness_lora_compact/tongue_constitution_qlora.toml \
  --data-dir data/wellness_lora_compact/export/tongue_constitution \
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-constitution-qlora \
  >"artifacts/logs/wellness_compact_tongue_constitution_${timestamp}.out.log" \
  2>"artifacts/logs/wellness_compact_tongue_constitution_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wellness_compact_tongue_constitution_${timestamp}.out.log"
```

#### 舌象体质康养计划 v2

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train/train_tongue_constitution_qlora.py \
  --config configs/wellness_lora_compact/tongue_constitution_plan_v2_qlora.toml \
  --data-dir data/wellness_lora_compact/tongue-constitution-plan-v2/export/tongue_constitution \
  --output-dir artifacts/qwen3-4b-wellness-compact-tongue-constitution-plan-v2-qlora \
  >"artifacts/logs/wellness_compact_tongue_constitution_plan_v2_${timestamp}.out.log" \
  2>"artifacts/logs/wellness_compact_tongue_constitution_plan_v2_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wellness_compact_tongue_constitution_plan_v2_${timestamp}.out.log"
```

#### 五态康养

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train/train_wutai_qlora.py \
  --config configs/wellness_lora_compact/wutai_qlora.toml \
  --data-dir data/wellness_lora_compact/export/wutai \
  --output-dir artifacts/qwen3-4b-wellness-compact-wutai-qlora \
  >"artifacts/logs/wellness_compact_wutai_${timestamp}.out.log" \
  2>"artifacts/logs/wellness_compact_wutai_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wellness_compact_wutai_${timestamp}.out.log"
```

#### 四诊康养

```bash
timestamp=$(date +%Y%m%d-%H%M%S)
mkdir -p artifacts/logs
PYTHONPATH=src nohup uv run python -u scripts/train/train_holistic_qlora.py \
  --config configs/wellness_lora_compact/holistic_qlora.toml \
  --data-dir data/wellness_lora_compact/export/holistic \
  --output-dir artifacts/qwen3-4b-wellness-compact-holistic-qlora \
  >"artifacts/logs/wellness_compact_holistic_${timestamp}.out.log" \
  2>"artifacts/logs/wellness_compact_holistic_${timestamp}.err.log" \
  </dev/null &
echo "PID: $!"
echo "Log: artifacts/logs/wellness_compact_holistic_${timestamp}.out.log"
```

## 3. 既有训练任务

下表保留现有医疗版任务的入口、数据和默认配置。它们的配置集中在 `configs/medical_lora/`；Windows 前台运行时先设置 `$env:PYTHONPATH = "src"`，Ubuntu 前台运行时在命令前加 `PYTHONPATH=src`。各任务可沿用第 2 节的命令形式替换脚本、配置、数据与输出目录。

| 任务 | 训练入口 | 配置 | 数据目录 | 默认输出目录 |
| --- | --- | --- | --- | --- |
| 体质分析 | `train_constitution_qlora.py` | `configs/medical_lora/constitution_qlora.toml` | `data/constitution-analysis` | `artifacts/qwen3-4b-constitution-qlora` |
| 原始舌象 | `train_tongue_qlora.py` | `configs/medical_lora/tongue_qlora.toml` | `data` | `artifacts/qwen3-4b-tongue-qlora` |
| 舌象组合对话 | `train_tongue_qlora.py` | `configs/medical_lora/tongue_qlora_conversations.toml` | `data/conversations` | `artifacts/qwen3-4b-tongue-conversations-qlora` |
| 舌象组合对话（1.7B） | `train_tongue_qlora.py` | `configs/medical_lora/tongue_qlora_conversations_1_7b.toml` | `data/conversations` | `artifacts/qwen3-1.7b-tongue-conversations-qlora` |
| 病历润色 | `train_case_polish_qlora.py` | `configs/medical_lora/case_polish_qlora_windows.toml` | `data/case-polish-lora` | `artifacts/qwen3-4b-case-polish-lora-windows` |
| 舌象体质 50K | `train_tongue_constitution_qlora.py` | `configs/medical_lora/tongue_constitution_50k_qlora.toml` | `data/tongue_constitution_50k` | `artifacts/qwen3-4b-tongue-constitution-50k-qlora` |
| 五态 20 | `train_wutai_qlora.py` | `configs/medical_lora/wutai_20_qlora.toml` | `data/wutai_20` | `artifacts/qwen3-4b-wutai-20-qlora` |
| 全诊报告 50K | `train_holistic_qlora.py` | `configs/medical_lora/holistic_50k_qlora.toml` | `data/holistic_50k` | `artifacts/qwen3-4b-holistic-50k-qlora` |

### Windows 后台启动器

以下两个既有 PowerShell 启动器会创建独立后台进程和带时间戳的日志文件：

```powershell
.\scripts\train\start_constitution_qlora_training.ps1
.\scripts\train\start_tongue_qlora_training.ps1
```

两者都支持 `-Config`、`-DataDir`、`-Model`、`-OutputDir`、`-CacheDir`、`-AllowDownload`、`-ResumeFromCheckpoint` 与 `-ResumeLatest` 参数。Ubuntu 不要执行 `.ps1` 启动器，应使用第 2 节所示的 `nohup` 方式。

### 舌象组合烟雾数据

```bash
uv run python scripts/maintenance/prepare_conversations_smoke_data.py
```

将上表中“舌象组合对话”的数据目录改为 `data/smoke/conversations`，输出目录改为 `artifacts/qwen3-4b-tongue-conversations-smoke` 即可运行烟雾训练。

## 4. 训练运维

### 查看日志

Windows PowerShell：

```powershell
Get-Content -LiteralPath artifacts\logs\<task>_<timestamp>.out.log -Tail 50 -Wait
```

Ubuntu Bash：

```bash
tail -f artifacts/logs/<task>_<timestamp>.out.log
```

### 恢复训练

必须沿用原来的 `--output-dir`，这样训练器才能找到完整 checkpoint。无值传入 `--resume-from-checkpoint` 会自动选择该目录中的最新完整 checkpoint；也可显式指定路径：

```bash
--resume-from-checkpoint
--resume-from-checkpoint artifacts/qwen3-4b-wellness-compact-wutai-qlora/checkpoint-500
```

将上述参数追加到相应的 Windows 或 Ubuntu 启动命令即可。`save_steps` 控制可恢复 checkpoint 的间隔；`save_total_limit = 1` 保留最新恢复 checkpoint，最佳验证 checkpoint 会按 Trainer 的规则额外保留。

### GPU 指标

每次记录训练日志时，训练入口会向输出目录中的 `gpu_metrics.jsonl` 追加 GPU、CPU 和内存指标；指标采集失败不会中断训练。

```bash
tail -f artifacts/qwen3-4b-wellness-compact-wutai-qlora/gpu_metrics.jsonl
```

## 5. 简化提示词体检报告分部模型（Windows）

三个命令均使用 Qwen3-4B，直接从原始 JSON 数据目录按四诊源信息分组，以固定种子划分 99% 训练集和 1% 验证集。先在 PowerShell 中执行一次 `$env:PYTHONPATH = "src"`，再按需启动下列独立 Adapter 训练。

### 第一、二部分：形体与心理

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_holistic_section_qlora.py `
  --config configs/medical_lora/holistic_report_parts_1_2_qlora_windows.toml `
  --data-dir data/medicine/holistic-tcm-report_简化提示词/第一二部分_形体与心理 `
  --output-dir artifacts/qwen3-4b-holistic-report-parts-1-2-qlora-windows
```

### 第三、四部分：气血与脏腑

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_holistic_section_qlora.py `
  --config configs/medical_lora/holistic_report_parts_3_4_qlora_windows.toml `
  --data-dir data/medicine/holistic-tcm-report_简化提示词/第三四部分_气血与脏腑 `
  --output-dir artifacts/qwen3-4b-holistic-report-parts-3-4-qlora-windows
```

### 第五部分：日常调护

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/train/train_holistic_section_qlora.py `
  --config configs/medical_lora/holistic_report_part_5_qlora_windows.toml `
  --data-dir data/medicine/holistic-tcm-report_简化提示词/第五部分_日常调护 `
  --output-dir artifacts/qwen3-4b-holistic-report-part-5-qlora-windows
```
