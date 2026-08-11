"""Train a Qwen3-4B QLoRA adapter on the constitution-analysis dataset."""

from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.tongue_qlora import split_constitution_examples

if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B constitution-analysis adapter with QLoRA.",
        default_config=Path("configs/constitution_qlora.toml"),
        default_data_dir=Path("data/constitution-analysis"),
        default_output_dir=Path("artifacts/qwen3-4b-constitution-qlora"),
        split_examples=split_constitution_examples,
    )
