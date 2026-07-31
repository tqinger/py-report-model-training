from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tcm_qwen_eval.dataset import load_examples
from tcm_qwen_eval.tongue_qlora import (
    DEFAULT_SEED,
    chat_prompt,
    choose_manual_examples,
    resolve_model_source,
    split_tongue_examples,
    tongue_format_check,
    write_json,
)
from tcm_qwen_eval.tongue_reporting import write_tongue_evaluation_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Qwen3-4B tongue QLoRA adapter on its held-out split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/qwen3-4b-tongue-qlora"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen3-4b-tongue-qlora-evaluation.xlsx"))
    parser.add_argument("--predictions-output", type=Path, default=Path("artifacts/qwen3-4b-tongue-qlora/test_predictions.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads when cache is missing.")
    return parser.parse_args()


def load_model(model_name: str, adapter_dir: Path, cache_dir: Path, allow_download: bool):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=not allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_source = resolve_model_source(model_name, cache_dir)
    base = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=not allow_download,
        cache_dir=cache_dir,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer, model_source


@torch.inference_mode()
def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = chat_prompt(tokenizer, messages)
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
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command in the uv CUDA environment.")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory is missing: {args.adapter}")

    test_examples = split_tongue_examples(load_examples(args.data_dir), args.seed)["test"]
    manual_examples = choose_manual_examples(test_examples, args.seed)
    manual_ids = {example.id for example in manual_examples}
    model, tokenizer, model_source = load_model(args.model, args.adapter, args.cache_dir, args.allow_download)

    predictions: list[dict[str, Any]] = []
    adapter_outputs: dict[str, str] = {}
    for index, example in enumerate(test_examples, start=1):
        started = time.perf_counter()
        output = generate(model, tokenizer, example.messages, args.max_new_tokens)
        passed, note = tongue_format_check(example.task, output)
        prediction = {
            "example_id": example.id,
            "task": example.task,
            "output": output,
            "format_pass": passed,
            "format_note": note,
            "output_chars": len(output),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        predictions.append(prediction)
        if example.id in manual_ids:
            adapter_outputs[example.id] = output
        print(f"[{index}/{len(test_examples)}] {example.task}", flush=True)

    base_outputs: dict[str, str] = {}
    with model.disable_adapter():
        for index, example in enumerate(manual_examples, start=1):
            base_outputs[example.id] = generate(model, tokenizer, example.messages, args.max_new_tokens)
            print(f"[base {index}/{len(manual_examples)}] {example.task}", flush=True)

    source_data_dir = args.data_dir
    if args.data_dir.name != "conversations":
        source_data_dir /= "tongue-analysis"

    write_json(args.predictions_output, predictions)
    write_tongue_evaluation_workbook(
        args.output,
        manual_examples,
        base_outputs,
        adapter_outputs,
        predictions,
        source_data_dir,
        {
            "基座模型": args.model,
            "基座权重来源": model_source,
            "Adapter": str(args.adapter),
            "生成参数": f"do_sample=False, max_new_tokens={args.max_new_tokens}, thinking=False",
            "测试集": "按舌象来源分组的独立 10% 测试集",
        },
    )
    print(json.dumps({"evaluation_workbook": str(args.output), "predictions": len(predictions)}, ensure_ascii=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
