"""Train a Qwen3-4B QLoRA adapter on the compact tongue-wellness SFT split."""

from functools import partial
from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.dataset import load_jsonl_sft_splits


if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B compact tongue-wellness adapter with QLoRA.",
        default_config=Path("configs/wellness_lora_compact/tongue_qlora.toml"),
        default_data_dir=Path("data/wellness_lora_compact/export/tongue"),
        default_output_dir=Path("artifacts/qwen3-4b-wellness-compact-tongue-qlora"),
        load_pre_split_examples=partial(
            load_jsonl_sft_splits,
            domain="tongue-analysis",
            task="tongue_wellness_advice",
        ),
    )
