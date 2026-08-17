from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from tcm_qwen_eval.checkpoints import LATEST_CHECKPOINT, resolve_resume_checkpoint
from tcm_qwen_eval.dataset import Example, load_examples
from tcm_qwen_eval.gpu_metrics import GPUMetricsCallback
from tcm_qwen_eval.tongue_qlora import (
    CausalDataCollator,
    TokenizedSFTDataset,
    encode_sft_example,
    resolve_model_source,
    split_manifest,
    split_tongue_examples,
    token_length_summary,
    write_json,
)
from tcm_qwen_eval.training_config import load_tongue_qlora_config

SplitExamples = Callable[[list[Example], int], dict[str, list[Example]]]
LoadPreSplitExamples = Callable[[Path], dict[str, list[Example]]]


def parse_args(
    *,
    description: str,
    default_config: Path,
    default_data_dir: Path,
    default_output_dir: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="TOML file that defines all QLoRA training hyperparameters.",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=LATEST_CHECKPOINT,
        metavar="CHECKPOINT",
        help=(
            "Resume from CHECKPOINT. Omit CHECKPOINT to resume automatically from the "
            "latest checkpoint in --output-dir."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/hf_cache"))
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads when cache is missing.")
    return parser.parse_args()


def main(
    *,
    description: str = "Fine-tune a Qwen3-4B tongue adapter with QLoRA.",
    default_config: Path = Path("configs/tongue_qlora.toml"),
    default_data_dir: Path = Path("data"),
    default_output_dir: Path = Path("artifacts/qwen3-4b-tongue-qlora"),
    split_examples: SplitExamples = split_tongue_examples,
    load_pre_split_examples: LoadPreSplitExamples | None = None,
) -> None:
    args = parse_args(
        description=description,
        default_config=default_config,
        default_data_dir=default_data_dir,
        default_output_dir=default_output_dir,
    )
    config = load_tongue_qlora_config(args.config)
    training = config.training
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command in the uv CUDA environment.")
    set_seed(training.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)

    if load_pre_split_examples is None:
        splits = split_examples(load_examples(args.data_dir), training.seed)
        manifest = split_manifest(splits, training.seed)
    else:
        splits = load_pre_split_examples(args.data_dir)
        manifest = split_manifest(splits, training.seed)
        manifest["split_strategy"] = "provided train.jsonl/val.jsonl"
        manifest["data_dir"] = str(args.data_dir)
    write_json(args.output_dir / "split_manifest.json", manifest)
    model_source = resolve_model_source(args.model, args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source, cache_dir=args.cache_dir, local_files_only=not args.allow_download
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    records = {
        name: [encode_sft_example(tokenizer, example, training.max_length) for example in examples]
        for name, examples in splits.items()
    }
    write_json(
        args.output_dir / "run_config.json",
        {
            "config_path": str(args.config),
            "model": args.model,
            "model_source": model_source,
            "hyperparameters": asdict(config),
            "token_lengths": {name: token_length_summary(items) for name, items in records.items()},
            "gpu_metrics_path": str(args.output_dir / "gpu_metrics.jsonl"),
        },
    )
    print(
        json.dumps(
            {
                "event": "run_prepared",
                "model": args.model,
                "output_dir": str(args.output_dir),
                "examples": {name: len(items) for name, items in records.items()},
                "logging_steps": training.logging_steps,
                "save_steps": training.save_steps,
                "save_total_limit": training.save_total_limit,
                "resume_from_checkpoint": resume_checkpoint,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=quantization,
        torch_dtype=getattr(torch, config.quantization.torch_dtype),
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=not args.allow_download,
        cache_dir=args.cache_dir,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training.gradient_checkpointing
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy=training.eval_strategy,
        save_strategy=training.save_strategy,
        save_steps=training.save_steps,
        # A resumable checkpoint requires the optimizer, scheduler, RNG, and Trainer state.
        save_only_model=False,
        logging_strategy=training.logging_strategy,
        logging_steps=training.logging_steps,
        # Plain log records are reliable in redirected/background output; tqdm
        # otherwise emits its progress display to stderr.
        disable_tqdm=True,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=training.metric_for_best_model,
        greater_is_better=training.greater_is_better,
        learning_rate=training.learning_rate,
        num_train_epochs=training.num_train_epochs,
        lr_scheduler_type=training.lr_scheduler_type,
        lr_scheduler_kwargs=training.lr_scheduler_kwargs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        warmup_ratio=training.warmup_ratio,
        max_grad_norm=training.max_grad_norm,
        optim=training.optim,
        bf16=training.bf16,
        tf32=training.tf32,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": training.gradient_checkpointing_use_reentrant
        },
        report_to=training.report_to,
        eval_steps=training.save_steps if training.eval_strategy == "steps" else None,
        seed=training.seed,
        data_seed=training.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedSFTDataset(records["train"]),
        eval_dataset=TokenizedSFTDataset(records["validation"]),
        data_collator=CausalDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[GPUMetricsCallback(args.output_dir)],
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    evaluation = trainer.evaluate()
    trainer.log_metrics("validation", evaluation)
    trainer.save_metrics("validation", evaluation)
    trainer.save_state()
    print(json.dumps({"adapter_dir": str(args.output_dir), "validation": evaluation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
