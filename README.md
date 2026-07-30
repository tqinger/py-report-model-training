# Qwen3 舌象 QLoRA 训练与评测

## 舌象 QLoRA 微调

以本地 `data/tongue-analysis` 的两项舌象子任务合并训练一个 `Qwen/Qwen3-4B` Adapter。训练保留原始 system 和 user 消息作为上下文，但只对 assistant 回复计算损失；同一舌象来源不会被拆到不同数据集。

```powershell
# 首次使用或依赖更新后执行
uv sync --group dev

# 在项目根目录启动训练
$env:PYTHONPATH = "src"
uv run python scripts/train_tongue_qlora.py

# 训练完成后，在独立测试集上生成自动指标与人工评分表
$env:PYTHONPATH = "src"
uv run python scripts/evaluate_tongue_qlora.py
```

### 后台训练与进度日志（Windows PowerShell）

需要关闭终端后继续训练时，用以下启动器代替前台训练命令：

```powershell
.\scripts\start_tongue_qlora_training.ps1
```

启动器会将训练作为独立后台进程运行，终端关闭不会中断训练。每次启动会在 `artifacts/logs/` 生成一对带时间戳的日志文件：`tongue_qlora_*.out.log` 记录训练进度，`tongue_qlora_*.err.log` 记录警告、错误和异常堆栈。训练输出未缓冲写入，因此可以在另一个 PowerShell 窗口实时查看进度：

```powershell
# 将 <timestamp> 替换为启动器打印出的时间戳
Get-Content -LiteralPath artifacts\logs\tongue_qlora_<timestamp>.out.log -Tail 50 -Wait

# 需要排查异常时，查看错误日志
Get-Content -LiteralPath artifacts\logs\tongue_qlora_<timestamp>.err.log -Tail 50 -Wait
```

日志开始时会写入模型、输出目录、各数据集样本数和日志间隔；随后显示可训练参数量，以及每 `logging_steps`（默认 10）步记录的 `loss`、`grad_norm`、`learning_rate`、`epoch`。每个 epoch 结束还会记录验证集指标和 checkpoint 保存信息。启动器会打印后台 PID、两份日志的完整路径，可用 `Get-Process -Id <PID>` 确认训练仍在运行。

它支持与训练脚本相同的常用运行参数，例如从 checkpoint 恢复或在模型未缓存时允许下载：

```powershell
.\scripts\start_tongue_qlora_training.ps1 `
  -ResumeFromCheckpoint artifacts\qwen3-4b-tongue-qlora\checkpoint-120 `
  -AllowDownload
```

Adapter、训练配置、切分清单和最佳验证 checkpoint 均写入 `artifacts/qwen3-4b-tongue-qlora/`。评测会生成 `artifacts/qwen3-4b-tongue-qlora-evaluation.xlsx`：其中的“结果表”仅含动态输入、Qwen3-4B 基座/QLoRA 输出与空白人工评分列。

### 默认超参数

| 项目 | 默认值 | 作用 |
| --- | --- | --- |
| 基座模型 | `Qwen/Qwen3-4B` | 仅加载预训练权重，不合并或修改基座权重 |
| 量化 | 4-bit NF4 + 双重量化 | 降低显存占用，适配 16GB 显存 |
| 计算精度 | BF16 | 量化层计算与 LoRA 训练精度 |
| LoRA 目标层 | `q/k/v/o_proj`、`gate/up/down_proj` | 同时适配注意力与 MLP 层 |
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` | 控制 Adapter 容量与正则化强度 |
| 最大序列长度 | `2048` | system、user 与 assistant 合并后的 token 上限 |
| 单卡 batch size | `1` | 降低单次显存峰值 |
| 梯度累积 | `8` | 有效 batch size 为 8 条样本 |
| 学习率 | `5e-5` | 小数据集下较保守的 QLoRA 初始学习率 |
| warmup | `5%` | 降低训练开始阶段的不稳定性 |
| 最大 epoch | `3` | 限制小数据集过拟合风险 |
| 优化器 | `paged_adamw_8bit` | 减少优化器状态显存 |
| 梯度检查点 | 启用 | 以训练速度换取显存空间 |
| 数据切分 | 固定种子 `20260729`、按舌象来源 80/10/10 | 防止同一舌象的两个子任务泄漏到验证或测试集 |
| 最佳模型选择 | 最低验证集 loss | 每个 epoch 保存 checkpoint，最终保留验证损失最低的 Adapter |

可按需覆盖关键参数，例如减少训练轮数：

```powershell
$env:PYTHONPATH = "src"
Copy-Item configs/tongue_qlora.toml configs/tongue_qlora_local.toml
# Edit training.num_train_epochs and training.learning_rate in the copied TOML file.
uv run python scripts/train_tongue_qlora.py --config configs/tongue_qlora_local.toml
```

模型默认只从本机 Hugging Face 缓存读取；缓存缺失时在命令末尾添加 `--allow-download`。

### 下载基座权重

使用下载脚本可将指定参数量的 Qwen3 权重下载到训练和评测共用的 `artifacts/hf_cache`。支持 `0.6B`、`1.7B` 和 `4B`，默认下载训练所需的 `4B`：

```powershell
uv run python scripts/download_qwen3_weights.py --size 4B
```

下载较小模型时替换参数量，例如：

```powershell
uv run python scripts/download_qwen3_weights.py --size 1.7B
```

## 环境

所有环境操作均使用 `uv`。项目固定 Python 3.12，并从 PyTorch CUDA 12.6 索引安装 GPU 版 PyTorch：

```powershell
uv python install 3.12
uv sync --group dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

若驱动与 CUDA 12.6 wheel 不兼容，先升级 NVIDIA 驱动；不要退回系统 Python 或 CPU 版 PyTorch。

## 数据与微调约束

- 训练仅使用 `data/tongue-analysis` 中的两项舌象任务。
- 同一舌象来源不可被拆散到训练、验证或测试集。
- 建议设置为 NF4、BF16 compute、rank 16、alpha 32、dropout 0.05、最大长度 2048、batch size 1 加梯度累积。
