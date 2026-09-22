"""Train a Qwen3-4B QLoRA adapter on the supplied case-polish SFT split."""

from functools import partial
from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.dataset import load_jsonl_sft_splits

if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B case-polish adapter with QLoRA.",
        default_config=Path("configs/medical_lora/case_polish_qlora_windows.toml"),
        default_data_dir=Path("data/case-polish-lora"),
        default_output_dir=Path("artifacts/qwen3-4b-case-polish-lora-windows"),
        load_pre_split_examples=partial(
            load_jsonl_sft_splits,
            domain="case-polish",
            task="case_polish",
        ),
    )
