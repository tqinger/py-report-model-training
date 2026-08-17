"""Train a Qwen3-4B QLoRA adapter on the supplied five-state SFT split."""

from functools import partial
from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.dataset import load_jsonl_sft_splits

if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B five-state wellness adapter with QLoRA.",
        default_config=Path("configs/wutai_20_qlora.toml"),
        default_data_dir=Path("data/wutai_20"),
        default_output_dir=Path("artifacts/qwen3-4b-wutai-20-qlora"),
        load_pre_split_examples=partial(
            load_jsonl_sft_splits,
            domain="wutai",
            task="wutai_wellness_analysis",
        ),
    )
