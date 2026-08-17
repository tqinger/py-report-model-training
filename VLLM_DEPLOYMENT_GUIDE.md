# Qwen3-4B 舌象与体质 LoRA 的 vLLM 部署手册

本手册说明如何将本项目训练得到的 Qwen3-4B 舌象和体质 LoRA Adapter 用**一个** vLLM 实例部署为 OpenAI 兼容 API，并覆盖：本机验证、FP8 基座转换、systemd 运维、切换新权重、迁移到阿里云 ECS、停止服务和排障。

> 当前已验证环境：Ubuntu 24.04、NVIDIA GeForce RTX 5090 D（32 GiB）、NVIDIA 驱动 595.84、Python 3.12、vLLM 0.26.0。实际部署使用了 `--enforce-eager`，以绕过该组合下图编译初始化时出现的 CUDA 段错误。

## 1. 部署对象与目录

当前服务器上的已验证路径如下。其他服务器可以使用不同根目录，但服务文件中必须全部改为对应的**绝对 Linux 路径**。

```text
项目根目录
/home/tan/py-report-model-training

Qwen3-4B 基座快照
/home/tan/py-report-model-training/artifacts/hf_cache/
  models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

舌象 LoRA Adapter（rank 16）
/home/tan/py-report-model-training/artifacts/
  qwen3-4b-tongue-conversations-qlora/checkpoint-10176

体质 LoRA Adapter（rank 16）
/home/tan/py-report-model-training/artifacts/
  qwen3-4b-constitution-qlora/checkpoint-2324
```

两个 Adapter 均基于同一份 Qwen3-4B 基座、使用 rank 16。Adapter 的 `adapter_config.json` 可能保留了训练机器上的 Windows 相对路径；部署时不要依赖该字段。始终将基座路径作为 `vllm serve` 的第一个参数，并通过 `--lora-modules` 显式提供 Adapter 路径。

本手册把 API 中的 Adapter 名称固定为 `tongue-qlora` 和 `constitution-qlora`。每次请求只选择其中一个 Adapter：

```json
{"model": "tongue-qlora"}
```

```json
{"model": "constitution-qlora"}
```

`--max-loras 2` 允许引擎在同一批次中调度这两个不同 Adapter；它不表示单个请求会同时叠加两个 Adapter。

## 2. 当前服务器：安装并启动服务

### 2.1 前置检查

```bash
nvidia-smi
df -h
```

确认 GPU 可见、磁盘有足够空间，并确认基座目录、两个 Adapter 目录内分别有模型权重和 `adapter_config.json`。当前服务按约 3,000 次/天的低并发场景配置为 `max_model_len=4096`、`gpu_memory_utilization=0.75`、`max_num_seqs=4`；显存较小或峰值并发较高时应重新压测。

### 2.2 创建独立 vLLM 环境

不要把推理依赖安装到训练用的 `.venv`。以下命令以 `tan` 用户为例：

```bash
# 若 uv 尚未安装，请按 https://docs.astral.sh/uv/ 的安装方式安装。
$HOME/.local/bin/uv venv --python 3.12 --seed $HOME/.venvs/vllm-qwen3
$HOME/.local/bin/uv pip install \
  --python $HOME/.venvs/vllm-qwen3/bin/python \
  vllm --torch-backend=auto

# 固化实际安装版本，供迁移和回滚使用。
$HOME/.local/bin/uv pip freeze \
  --python $HOME/.venvs/vllm-qwen3/bin/python \
  > $HOME/.venvs/vllm-qwen3/requirements.lock

$HOME/.venvs/vllm-qwen3/bin/vllm --version
```

`--torch-backend=auto` 会根据机器上的 NVIDIA 驱动选择合适的 PyTorch/CUDA 后端。不要复制其他机器的 `site-packages` 目录；应在目标 GPU 机器上重新安装。

### 2.3 生成可部署的 FP8 共享基座（可选）

FP8 部署仍保留两个 LoRA Adapter。转换对象是共同的完整 Qwen3-4B 基座，**不是** `checkpoint-2324` 或 `checkpoint-10176`；若将 Adapter 合并为完整模型，就无法在同一个服务中按请求切换两项能力。

在部署机新建与 vLLM 环境分离的 llm-compressor 环境。vLLM 官方建议将两者分开安装：

```bash
$HOME/.local/bin/uv venv --python 3.12 --seed $HOME/.venvs/llm-compressor
$HOME/.local/bin/uv pip install \
  --python $HOME/.venvs/llm-compressor/bin/python \
  llmcompressor

# <项目根目录> 内的脚本；输出目录必须为新建或空目录。
BASE_BF16=/home/tan/py-report-model-training/artifacts/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
FP8_BASE=/home/tan/models/releases/qwen3-4b-fp8-20260812

$HOME/.venvs/llm-compressor/bin/python \
  /home/tan/py-report-model-training/scripts/convert_qwen3_4b_base_to_fp8.py \
  --model "$BASE_BF16" \
  --output-dir "$FP8_BASE" \
  --device cuda:0
```

脚本使用无需校准数据的 `FP8_DYNAMIC` 量化：Linear 层权重为 FP8，激活在运行时按 token 动态量化；`lm_head` 保留原精度。它会校验输入是完整 Hugging Face 基座、拒绝 LoRA Adapter 目录，并在完成后检查输出的权重和量化元数据。请保留脚本最终输出的 `fp8_model_dir` 与 `quantization_config` 记录。

FP8 W8A8 需要 Ada Lovelace（计算能力 8.9）或 Hopper 及更新架构才能获得硬件加速；在不支持的 GPU 上不要将此流程作为生产部署方案。转换完成后使用下文的双 Adapter 冒烟请求验收，再切换生产服务。

### 2.4 先在前台验证（可选）

在创建 systemd 服务前，可用下面的命令前台启动。确认成功后按 `Ctrl+C` 停止，再继续下一节。

```bash
export PATH="$HOME/.venvs/vllm-qwen3/bin:$PATH"

$HOME/.venvs/vllm-qwen3/bin/vllm serve \
  /home/tan/py-report-model-training/artifacts/hf_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --host 127.0.0.1 --port 8000 \
  --enable-lora \
  --lora-modules tongue-qlora=/home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-10176 constitution-qlora=/home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-2324 \
  --max-lora-rank 16 --max-loras 2 \
  --gpu-memory-utilization 0.75 --max-model-len 4096 --max-num-seqs 4 \
  --no-enable-prefix-caching --enforce-eager
```

上面的命令使用 BF16 基座。如已完成 FP8 转换，只替换 `vllm serve` 后的第一个路径为 `$FP8_BASE`，其余两个 `--lora-modules` 和服务参数保持不变：

```bash
$HOME/.venvs/vllm-qwen3/bin/vllm serve "$FP8_BASE" \
  --host 127.0.0.1 --port 8000 \
  --enable-lora \
  --lora-modules tongue-qlora=/home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-10176 constitution-qlora=/home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-2324 \
  --max-lora-rank 16 --max-loras 2 \
  --gpu-memory-utilization 0.75 --max-model-len 4096 --max-num-seqs 4 \
  --no-enable-prefix-caching --enforce-eager
```

`--enforce-eager` 会关闭 torch.compile 和 CUDA Graph 优化，吞吐会低于图编译模式，但它是当前 RTX 5090 D + vLLM 0.26.0 组合的稳定设置。将来升级 vLLM、PyTorch 或驱动后，可在独立验证环境中移除此参数重新压测；未验证前不要直接移除。

#### 当前服务启动参数说明

| 参数 | 当前值 | 作用与调整建议 |
| --- | --- | --- |
| `--gpu-memory-utilization` | `0.75` | 允许 vLLM 最多规划使用约 75% 的可用 GPU 显存，主要用于模型后的 KV Cache。当前低并发场景无需预留 90%；峰值并发增加前应通过压测再提高。 |
| `--max-model-len` | `4096` | 单次请求允许的最大上下文长度（输入 token 与输出 token 合计上限）。值越大，单请求可处理的内容越长，但 KV Cache 占用越多；需要长病历或长输出时再升回 `8192`。 |
| `--max-num-seqs` | `4` | 调度器一次最多并行处理的序列数量。适合约 3,000 次/天的低并发场景；请求超过此数量会排队，不代表每天只能处理 4 次。 |
| `--max-loras` | `2` | 启动时预留两个 LoRA 的调度容量，以允许 `tongue-qlora` 与 `constitution-qlora` 同批请求。单条请求仍只选择一个 `model`。 |
| `--max-lora-rank` | `16` | 两个当前 Adapter 的实际 rank。应设为将加载 Adapter 中的最大值，设得更大会浪费内存。 |
| `--no-enable-prefix-caching` | 已启用 | 关闭跨请求的 Prefix Cache 复用。每个正在生成的请求仍必须使用临时 KV Cache；该参数不会关闭请求内 KV Cache。 |
| `--enforce-eager` | 已启用 | 禁用 torch.compile 和 CUDA Graph，改为即时执行。当前用于规避 RTX 5090 D + vLLM 0.26.0 的图编译初始化段错误；它会牺牲部分性能，但不影响模型语义和 LoRA 效果。 |

这四项属于**服务启动参数**，位于 systemd unit 的 `ExecStart=`。修改后必须执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart tongue-vllm.service
```

### 2.5 创建 systemd 服务

保存以下文件为 `/home/tan/tongue-vllm.service`，并按实际路径替换模型与 Adapter 目录：

```ini
[Unit]
Description=Qwen3-4B Tongue and Constitution LoRA vLLM API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tan
Group=tan
WorkingDirectory=/home/tan/py-report-model-training
Environment=PATH=/home/tan/.venvs/vllm-qwen3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=HF_HOME=/home/tan/py-report-model-training/artifacts/hf_cache
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/home/tan/.venvs/vllm-qwen3/bin/vllm serve /home/tan/models/base-current --host 127.0.0.1 --port 8000 --enable-lora --lora-modules tongue-qlora=/home/tan/models/tongue-qlora-current constitution-qlora=/home/tan/models/constitution-qlora-current --max-lora-rank 16 --max-loras 2 --gpu-memory-utilization 0.75 --max-model-len 4096 --max-num-seqs 4 --no-enable-prefix-caching --enforce-eager
Restart=on-failure
RestartSec=10
TimeoutStartSec=0
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

其中 `PATH` 不能省略：vLLM 的 FlashInfer 运行时需要调用虚拟环境中的 `ninja`。首次启用前创建服务使用的稳定链接；FP8 部署将 `base-current` 指向上一节的 `$FP8_BASE`。若部署 BF16，请将该变量改为第 1 节所列的原始基座快照：

```bash
mkdir -p /home/tan/models
ln -sfn "$FP8_BASE" /home/tan/models/base-current
ln -sfn /home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-10176 \
  /home/tan/models/tongue-qlora-current
ln -sfn /home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-2324 \
  /home/tan/models/constitution-qlora-current
```

然后安装、启用并查看状态：

```bash
sudo install -o root -g root -m 644 \
  /home/tan/tongue-vllm.service /etc/systemd/system/tongue-vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now tongue-vllm.service
sudo systemctl status tongue-vllm.service --no-pager
```

### 2.6 验收 API

服务只绑定 `127.0.0.1`，因此以下命令必须在服务器上执行：

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tongue-qlora",
    "messages": [
      {
        "role": "system",
        "content": "你是严谨的中医舌诊助手。仅依据给定舌象和患者信息，先概括病机及印证，再给1–2条一致的日常调养建议；不得作疾病确诊或夸大推断。不得补充未提供或矛盾的舌象；仅有齿痕舌时不得称为胖大舌。"
      },
      {
        "role": "user",
        "content": "请生成舌面综合分析与日常调养建议。\n患者信息：性别女，年龄53岁\n舌象特征：淡红舌、正常舌形、薄白苔"
      }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 320,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

`/v1/models` 应同时包含 `tongue-qlora` 和 `constitution-qlora`。接着执行体质报告冒烟请求：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "constitution-qlora",
    "messages": [
      {"role": "system", "content": "你是中医体质调理顾问。表达通俗、专业、温和；不作疾病诊断。"},
      {"role": "user", "content": "根据以下体质辨识信息生成 JSON，只输出 JSON：\n{\n  \"analysis_summary\": \"一句综合概括\",\n  \"analysis_symptoms\": \"体质表现段落\",\n  \"analysis_advice\": \"调理方向段落\"\n}\n\n要求：仅使用输入列出的体质，三个字段均不可为空。\n\n体质辨识信息：\n结论：您的主要体质为气虚质，存在阴虚质、痰湿质倾向\n主体质（main_type）：QIXU（气虚质）\n兼挟体质：无\n倾向体质：阴虚质、痰湿质"}
    ],
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 384,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

FP8 部署验收标准是：`/v1/models` 同时返回两个 Adapter，舌象和体质两条冒烟请求均返回非空回复，且体质请求可以解析为包含三个非空字段的 JSON。本项目的评测推理禁用了 Qwen3 thinking，在线调用也建议传入 `chat_template_kwargs.enable_thinking=false`，以保持行为一致。

### 2.7 每次请求的推理参数

以下 JSON 是调用方在每次 `POST /v1/chat/completions` 中传入的参数示例。它们只影响当前请求，修改时不需要重启服务：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是严谨的中医舌诊助手。仅依据给定舌象和患者信息，先概括病机及印证，再给1–2条一致的日常调养建议；不得作疾病确诊或夸大推断。不得补充未提供或矛盾的舌象；仅有齿痕舌时不得称为胖大舌。"
    },
    {
      "role": "user",
      "content": "请生成舌面综合分析与日常调养建议。\n患者信息：性别女，年龄53岁\n舌象特征：淡红舌、正常舌形、薄白苔"
    }
  ]
}
```

上述内容是 `messages` 字段示例。实际请求 `/v1/chat/completions` 时，还需要在外层补上模型名和可选采样参数，例如：

```json
{
  "model": "tongue-qlora",
  "temperature": 0.2,
  "top_p": 0.9,
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

- `temperature`：随机性；值越低，回答越稳定、可复现性越高。医疗类结构化输出通常从 `0` 到 `0.3` 开始验证。
- `top_p`：核采样范围；只在累积概率最高的 token 范围内采样。较低值会进一步收敛输出风格。
- `max_tokens`：本次最多生成的输出 token 数，不等于输入加输出的上下文上限；仍受 `--max-model-len` 限制。
- `enable_thinking=false`：关闭 Qwen3 thinking 输出，与当前评测推理保持一致。

对 `data/conversations` 的 76,320 条训练目标回复，使用当前 Qwen3 tokenizer 统计得到：P50 为 187 token、P95 为 222、P99 为 239、最大值为 305。因此默认设置为 `max_tokens=320`，可覆盖全部训练样本并保留少量生成余量；仅在允许更长报告时再提高到 `384` 或 `512`。

若请求未传 `temperature`、`top_p` 等采样参数，当前 Qwen3 基座的 `generation_config.json` 会提供默认值：`temperature=0.6`、`top_p=0.95`、`top_k=20`。

## 3. 日常启停、日志和资源检查

```bash
# 服务状态和最近日志
sudo systemctl status tongue-vllm.service --no-pager
sudo journalctl -u tongue-vllm.service -n 100 --no-pager
sudo journalctl -u tongue-vllm.service -f

# 重启：切换权重、修改配置后必须执行
sudo systemctl restart tongue-vllm.service

# 正常停止：会释放模型和 GPU 显存
sudo systemctl stop tongue-vllm.service

# 停止并取消开机自启
sudo systemctl disable --now tongue-vllm.service

# 恢复开机自启并立即启动
sudo systemctl enable --now tongue-vllm.service

# 确认 8000 仅在回环地址监听，并查看显存
ss -ltn | grep ':8000'
nvidia-smi
```

不要使用 `kill -9` 作为常规停止方式；优先使用 `systemctl stop`，它能正确结束 API Server 和 EngineCore 子进程。仅在进程卡死且 `systemctl stop` 超时后再排查 PID 和强制终止。

## 4. 安全切换模型版本（两个 LoRA 或 FP8 基座）

vLLM 只会在服务启动时加载 LoRA。因此切换任一 Adapter 或基座一定需要重启服务。切换前先确认新 Adapter 与当前 Qwen3-4B 基座匹配，并至少检查：

```bash
NEW_ADAPTER=/home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-<新步数>
test -f "$NEW_ADAPTER/adapter_config.json"
test -f "$NEW_ADAPTER/adapter_model.safetensors" -o -f "$NEW_ADAPTER/adapter_model.bin"
cat "$NEW_ADAPTER/adapter_config.json"
```

### 4.1 推荐：三个稳定路径 + 符号链接

systemd 应始终使用一个基座路径和两个 Adapter 稳定路径：

```text
vllm serve /home/tan/models/base-current ...
--lora-modules tongue-qlora=/home/tan/models/tongue-qlora-current constitution-qlora=/home/tan/models/constitution-qlora-current
```

首次创建当前版本链接并修改一次服务文件：

```bash
mkdir -p /home/tan/models
ln -sfn \
  /home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-10176 \
  /home/tan/models/tongue-qlora-current
ln -sfn \
  /home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-2324 \
  /home/tan/models/constitution-qlora-current

sudoedit /etc/systemd/system/tongue-vllm.service
# 替换基座与两个 --lora-modules 路径为上述稳定链接。
sudo systemctl daemon-reload
sudo systemctl restart tongue-vllm.service
```

之后切换舌象 Adapter 时，记录旧路径、原子替换链接、重启并完成第 2.6 节的两个 API 验收：

```bash
OLD_ADAPTER=$(readlink -f /home/tan/models/tongue-qlora-current)
NEW_ADAPTER=/home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-<新步数>

ln -s "$NEW_ADAPTER" /home/tan/models/tongue-qlora-next
mv -Tf /home/tan/models/tongue-qlora-next /home/tan/models/tongue-qlora-current
sudo systemctl restart tongue-vllm.service
curl -fsS http://127.0.0.1:8000/v1/models
```

若新版本加载或效果异常，回滚为旧链接并重启：

```bash
ln -s "$OLD_ADAPTER" /home/tan/models/tongue-qlora-next
mv -Tf /home/tan/models/tongue-qlora-next /home/tan/models/tongue-qlora-current
sudo systemctl restart tongue-vllm.service
```

切换体质 Adapter 时使用同样的流程，只替换对应的变量与稳定链接：

```bash
OLD_ADAPTER=$(readlink -f /home/tan/models/constitution-qlora-current)
NEW_ADAPTER=/home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-<新步数>
test -f "$NEW_ADAPTER/adapter_config.json"
test -f "$NEW_ADAPTER/adapter_model.safetensors" -o -f "$NEW_ADAPTER/adapter_model.bin"
ln -s "$NEW_ADAPTER" /home/tan/models/constitution-qlora-next
mv -Tf /home/tan/models/constitution-qlora-next /home/tan/models/constitution-qlora-current
sudo systemctl restart tongue-vllm.service
```

如需回滚，使用已记录的 `OLD_ADAPTER` 替换 `constitution-qlora-current` 并重启。

### 4.2 切换 BF16/FP8 共享基座

两个 LoRA 只能加载到与训练时相同的 Qwen3-4B 架构和 tokenizer 上。FP8 产物是同一基座的量化版本，因而保留两个 LoRA 参数；不要将某个 Adapter 合并成专用整模，否则不能在此服务中同时路由舌象和体质请求。

在不停现有服务的情况下，先检查新 FP8 基座产物：

```bash
OLD_BASE=$(readlink -f /home/tan/models/base-current)
NEW_BASE=/home/tan/models/releases/qwen3-4b-fp8-<版本>

test -f "$NEW_BASE/config.json"
test -f "$NEW_BASE/tokenizer_config.json"
test -n "$(find "$NEW_BASE" -maxdepth 1 -name '*.safetensors' -print -quit)"
# 应输出量化元数据；若为空，不要切换。
grep -n 'quantization_config' "$NEW_BASE/config.json"

ln -s "$NEW_BASE" /home/tan/models/base-next
mv -Tf /home/tan/models/base-next /home/tan/models/base-current

sudo systemctl restart tongue-vllm.service
curl -fsS http://127.0.0.1:8000/v1/models
```

若服务无法加载或任一第 2.6 节冒烟请求失败，恢复旧基座链接并重启：

```bash
ln -s "$OLD_BASE" /home/tan/models/base-next
mv -Tf /home/tan/models/base-next /home/tan/models/base-current
sudo systemctl restart tongue-vllm.service
```

## 5. 迁移到阿里云 ECS

### 5.1 实例与安全基线

1. 选择 Linux GPU ECS，优先选择带预装 NVIDIA 驱动的 GPU 镜像；若没有预装驱动，按实例系列安装适用于计算场景的 NVIDIA Tesla 驱动。
2. 建议使用至少 100 GiB 系统盘/数据盘，并为模型、vLLM 依赖、缓存和日志预留空间。
3. 在目标实例运行 `nvidia-smi`；没有 GPU 或驱动版本不兼容时，不要继续安装 vLLM。
4. 安全组采用白名单：SSH `22` 仅允许管理员固定公网 IP；公网服务仅开放 `443`；**不要开放 vLLM 的 `8000` 端口**。若只供 VPC 内业务调用，443 也只授权对应 VPC 网段或业务安全组。

阿里云安全组默认拒绝入站流量，应只增加业务所需端口和来源。官方参考：

- [ECS 安全组的使用场景与原则](https://www.alibabacloud.com/help/en/ecs/user-guide/security-groups-for-different-use-cases)
- [GPU ECS 驱动安装指南](https://www.alibabacloud.com/help/en/egs/user-guide/installation-guideline-for-nvidia-drivers)
- [带预装 NVIDIA 驱动的 Alibaba Cloud Linux 镜像](https://www.alibabacloud.com/help/en/ecs/user-guide/alibaba-cloud-linux-3-with-pre-installed-nvidia-gpu-drivers)

### 5.2 传输 FP8 基座和两个 Adapter

在新 ECS 上建立目标根目录。下面以从当前服务器复制为例，实际将主机名、端口和用户替换为你的值：

```bash
DEPLOY_ROOT=/home/<云端用户>/tongue-vllm
mkdir -p "$DEPLOY_ROOT/models"

# 传输完成的 FP8 基座目录；量化产物必须完整保留 config、tokenizer 与 safetensors。
rsync -aP -e 'ssh -p 22' \
  tan@<源服务器>:/home/tan/models/releases/qwen3-4b-fp8-20260812 \
  "$DEPLOY_ROOT/models/"

rsync -aP -e 'ssh -p 22' \
  tan@<源服务器>:/home/tan/py-report-model-training/artifacts/qwen3-4b-tongue-conversations-qlora/checkpoint-10176 \
  "$DEPLOY_ROOT/artifacts/"

rsync -aP -e 'ssh -p 22' \
  tan@<源服务器>:/home/tan/py-report-model-training/artifacts/qwen3-4b-constitution-qlora/checkpoint-2324 \
  "$DEPLOY_ROOT/artifacts/"
```

也可以通过对象存储或公司制品库传输，但必须对传输后的文件执行 `sha256sum` 或至少在目标机启动 vLLM 并完成两个 API 冒烟请求。不要把基座权重或 Adapter 上传到公开存储桶。

### 5.3 云端安装与服务

在 ECS 上重复第 2 节：创建独立 vLLM 环境、重新安装 vLLM、创建 systemd 服务。服务文件中的 `User`、`Group`、`WorkingDirectory`、`HF_HOME`、基座路径和 Adapter 路径都必须替换为云端实际值。

云端仍保持：

```text
vLLM: 127.0.0.1:8000
公网入口: HTTPS 反向代理或企业 API 网关
```

不要将 `--host` 改为 `0.0.0.0` 后直接把 8000 端口暴露公网。生产环境应由 Nginx/API 网关终止 TLS，并实施认证、限流、审计日志和请求体大小限制。例如 Nginx 的基础反向代理形态如下；证书、认证和域名必须按实际环境配置：

```nginx
server {
    listen 443 ssl http2;
    server_name llm.example.com;

    ssl_certificate     /etc/letsencrypt/live/llm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/llm.example.com/privkey.pem;
    client_max_body_size 10m;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }
}
```

至少选择一种认证方式：企业 API 网关/SSO、Nginx `auth_request`、或 vLLM 的 `--api-key`。密钥不得写入 Git、镜像或 shell 历史；应交给密钥管理服务或仅 root 可读的环境文件。安全组只应放行反向代理对外提供的端口。

## 6. 常见问题

| 现象 | 优先处理方式 |
| --- | --- |
| `No such file or directory: 'ninja'` | 确认 service 中的 `Environment=PATH=.../.venvs/vllm-qwen3/bin:...` 存在，并确认 `$HOME/.venvs/vllm-qwen3/bin/ninja` 可执行。 |
| GPU 初始化时出现 `Segfault`、`cuCtxSynchronize` 或引擎启动失败 | 保留 `--enforce-eager`；先以同一 venv 运行最小 PyTorch CUDA 张量测试，再考虑升级/回退 vLLM、PyTorch 或驱动。 |
| 显存不足或启动时 KV cache 分配失败 | 降低 `--gpu-memory-utilization`、`--max-model-len`、`--max-num-seqs`，并确认没有其他 GPU 任务。 |
| `/v1/models` 缺少 `tongue-qlora` 或 `constitution-qlora` | 检查 `--enable-lora`、两个 `--lora-modules <名称>=<路径>`、Adapter 文件和 `journalctl` 错误。 |
| 外部机器连不上 API | 若服务监听 `127.0.0.1`，这是预期安全行为；请通过 SSH 隧道、反向代理或网关访问，而不是直接暴露 8000。 |
| 新 Adapter 或 FP8 基座效果异常 | 立即按第 4 节恢复旧 Adapter/基座符号链接并重启；保留对应评测结果、模型路径和依赖锁定文件。 |

## 7. 版本和变更记录建议

每次上线或切换权重时记录以下信息：上线时间、基座路径或模型提交、Adapter checkpoint、Adapter 的 `adapter_config.json`、vLLM 版本、`requirements.lock`、GPU/驱动版本、服务文件校验结果，以及一条固定的 API 冒烟请求结果。这样可在云端复现问题并快速回滚。

vLLM 官方文档：

- [GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/)
- [FP8 W8A8 与 llm-compressor](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)
- [OpenAI 兼容服务器](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
