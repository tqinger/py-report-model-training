"""Train one simplified-prompt holistic-report section adapter with QLoRA."""

from pathlib import Path

from train_tongue_qlora import main

from tcm_qwen_eval.dataset import load_holistic_section_sft_splits


if __name__ == "__main__":
    main(
        description="Fine-tune a Qwen3-4B adapter for one holistic TCM-report section.",
        default_config=Path("configs/medical_lora/holistic_report_parts_1_2_qlora_windows.toml"),
        default_data_dir=Path("data/medicine/holistic-tcm-report_简化提示词/第一二部分_形体与心理"),
        default_output_dir=Path("artifacts/qwen3-4b-holistic-report-parts-1-2-qlora-windows"),
        load_pre_split_examples=load_holistic_section_sft_splits,
        pre_split_strategy="source-grouped 99/1 split from raw section JSON files",
    )
