"""Unit tests for the deployment-only FP8 base-model conversion guards."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = Path("scripts/convert_qwen3_4b_base_to_fp8.py")
SPEC = spec_from_file_location("convert_qwen3_4b_base_to_fp8", SCRIPT)
assert SPEC and SPEC.loader
converter = module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def write_base_model(directory: Path) -> None:
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"placeholder")


def test_require_base_model_accepts_complete_model(tmp_path: Path) -> None:
    write_base_model(tmp_path)

    converter.require_base_model(tmp_path)


def test_require_base_model_rejects_lora_adapter(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"placeholder")

    with pytest.raises(SystemExit, match="LoRA adapter"):
        converter.require_base_model(tmp_path)


def test_require_empty_output_rejects_existing_content(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="new or empty"):
        converter.require_empty_output(tmp_path)


def test_verify_output_requires_fp8_metadata(tmp_path: Path) -> None:
    write_base_model(tmp_path)

    with pytest.raises(RuntimeError, match="quantization_config"):
        converter.verify_output(tmp_path)


def test_verify_output_accepts_quantized_checkpoint(tmp_path: Path) -> None:
    write_base_model(tmp_path)
    (tmp_path / "config.json").write_text(
        '{"quantization_config": {"format": "float-quantized", "scheme": "FP8_DYNAMIC"}}',
        encoding="utf-8",
    )

    converter.verify_output(tmp_path)
