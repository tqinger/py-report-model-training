# 数据集与 Adapter 对照

训练数据保存在本地 `data/`，该目录被 Git 忽略，避免提交医疗样本与大文件。本文件记录运行所需的相对路径、用途和训练入口；迁移到新机器时应按此表恢复数据目录。

| 数据目录 | 任务 | 训练入口 | 默认 Adapter 输出 |
| --- | --- | --- | --- |
| `data/tongue-analysis` | 舌象分析 | `scripts/train/train_tongue_qlora.py` | `artifacts/qwen3-4b-tongue-qlora` |
| `data/constitution-analysis` | 体质分析 | `scripts/train/train_constitution_qlora.py` | `artifacts/qwen3-4b-constitution-qlora` |
| `data/holistic_50k` | 全诊报告 | `scripts/train/train_holistic_qlora.py` | `artifacts/qwen3-4b-holistic-50k-qlora` |
| `data/tongue_constitution_50k` | 舌象体质 | `scripts/train/train_tongue_constitution_qlora.py` | `artifacts/qwen3-4b-tongue-constitution-50k-qlora` |
| `data/wutai_20` | 五态报告 | `scripts/train/train_wutai_qlora.py` | `artifacts/qwen3-4b-wutai-20-qlora` |
| `data/case-polish-lora` | 病历润色 | `scripts/train/train_case_polish_qlora.py` | `artifacts/qwen3-4b-case-polish-lora-windows` |
| `data/wellness_lora_compact/export` | 康养系列任务 | 对应 `scripts/train/train_*_qlora.py` | `artifacts/qwen3-4b-wellness-compact-*` |

## 简化提示词体检报告分部数据

根目录为 `data/medicine/holistic-tcm-report_简化提示词`。三个目录分别训练独立 Qwen3-4B QLoRA Adapter；训练时不复制源数据，而是按四诊源信息分组，以固定种子划分 99% 训练集和 1% 验证集。

| 数据子目录 | 模型部分 | 配置 | 默认输出 |
| --- | --- | --- | --- |
| `第一二部分_形体与心理` | 形体与心理 | `configs/medical_lora/holistic_report_parts_1_2_qlora_windows.toml` | `artifacts/qwen3-4b-holistic-report-parts-1-2-qlora-windows` |
| `第三四部分_气血与脏腑` | 气血与脏腑 | `configs/medical_lora/holistic_report_parts_3_4_qlora_windows.toml` | `artifacts/qwen3-4b-holistic-report-parts-3-4-qlora-windows` |
| `第五部分_日常调护` | 日常调护 | `configs/medical_lora/holistic_report_part_5_qlora_windows.toml` | `artifacts/qwen3-4b-holistic-report-part-5-qlora-windows` |

对应命令见 [`TRAINING_COMMANDS.md`](../TRAINING_COMMANDS.md)。
