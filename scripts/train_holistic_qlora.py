"""Train a Qwen3-4B QLoRA adapter on the supplied holistic-report SFT split."""

from functools import partial
from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.dataset import load_jsonl_sft_splits

if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B holistic TCM-report adapter with QLoRA.",
        default_config=Path("configs/holistic_50k_qlora.toml"),
        default_data_dir=Path("data/holistic_50k"),
        default_output_dir=Path("artifacts/qwen3-4b-holistic-50k-qlora"),
        load_pre_split_examples=partial(
            load_jsonl_sft_splits,
            domain="holistic-tcm-report",
            task="holistic_tcm_report",
        ),
    )
