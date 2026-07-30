from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tcm_qwen_eval.dataset import audit_examples, default_data_dir, load_examples, read_selection
from tcm_qwen_eval.reporting import write_workbook

DEFAULT_MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")


def format_check(task: str, output: str) -> tuple[bool, str]:
    text = output.strip()
    if not text:
        return False, "输出为空"
    forbidden_markdown = any(token in text for token in ("```", "## ", "**"))
    if task == "constitution_summary":
        return 25 <= len(text) <= 40 and "\n" not in text, "要求单句且 25–40 字"
    if task == "tongue_daily_advice":
        return 60 <= len(text) <= 80 and not any(char.isdigit() for char in text), "要求 60–80 字且无编号"
    if task == "report_part_5":
        required = ("五、日常调护建议", "1. 饮食：", "2. 起居：", "3. 情志：")
        return all(part in text for part in required), "要求第五部分标题及饮食/起居/情志三项"
    if task.startswith(("case_", "tongue_", "constitution_")):
        return not forbidden_markdown, "要求无 Markdown"
    return not forbidden_markdown, "检查无 Markdown 控制符"


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    prompt_messages = messages[:2]
    try:
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def load_model(model_name: str, load_in_4bit: bool):
    model_options: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if load_in_4bit:
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_options["device_map"] = "auto"
    else:
        model_options["device_map"] = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_options)
    # Qwen checkpoints carry sampling defaults. Greedy evaluation must explicitly clear them,
    # otherwise recent Transformers versions emit a warning even though they are ignored.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = render_prompt(tokenizer, messages)
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(
        **model_inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0, model_inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible zero-shot Qwen3 baseline evaluation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--selection", type=Path, default=Path("artifacts/baseline_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen3_baseline.xlsx"))
    parser.add_argument("--model", action="append", default=[], help="Repeat to override the default model list.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use only if BF16 loading exceeds VRAM.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command in the uv CUDA environment.")
    if not args.selection.exists():
        raise SystemExit(f"Selection file missing: {args.selection}. Run scripts/prepare_baseline.py first.")

    selected = read_selection(args.selection)
    models = tuple(args.model) if args.model else DEFAULT_MODELS
    generations: list[dict[str, Any]] = []
    for model_name in models:
        print(f"Loading {model_name} ...", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model, tokenizer = load_model(model_name, args.load_in_4bit)
        except torch.OutOfMemoryError as exc:
            raise SystemExit(f"{model_name} BF16 load exceeded VRAM; retry with --load-in-4bit. {exc}") from exc
        for index, example in enumerate(selected, start=1):
            started = time.perf_counter()
            output = generate(model, tokenizer, example.messages, args.max_new_tokens)
            latency = round(time.perf_counter() - started, 3)
            passed, note = format_check(example.task, output)
            generations.append(
                {
                    "model": model_name,
                    "example_id": example.id,
                    "output": output,
                    "format_pass": passed,
                    "format_note": note,
                    "output_chars": len(output),
                    "latency_seconds": latency,
                    "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                }
            )
            print(f"  [{index}/{len(selected)}] {example.task}: {latency:.1f}s", flush=True)
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    selected_rows = [
        {
            "id": example.id,
            "domain": example.domain,
            "task": example.task,
            "group_id": example.group_id,
            "user": example.user,
            "reference": example.reference,
        }
        for example in selected
    ]
    write_workbook(
        args.output,
        audit_examples(load_examples(args.data_dir)),
        selected_rows,
        generations,
        {
            "生成参数": f"do_sample=False, max_new_tokens={args.max_new_tokens}, thinking=False",
            "权重精度": "4-bit NF4" if args.load_in_4bit else "BF16",
            "运行设备": torch.cuda.get_device_name(0),
        },
    )
    print(f"Workbook written to {args.output}")


if __name__ == "__main__":
    main()
