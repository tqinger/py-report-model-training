"""Export a shared Qwen3-4B base model as a vLLM-compatible FP8 checkpoint.

This script deliberately quantizes only the full Qwen3-4B base model.  Keep
task-specific LoRA adapters separate so one vLLM service can route requests to
multiple adapters (for example, tongue-qlora and constitution-qlora).

Run it in a dedicated llm-compressor environment on the deployment host; do not
install llm-compressor into the training or vLLM environment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REQUIRED_MODEL_FILES = ("config.json", "tokenizer_config.json")
ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a full Qwen3-4B Hugging Face base model to an FP8_DYNAMIC checkpoint."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Local directory of the full BF16/FP16 Qwen3-4B Hugging Face base model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty deployment directory for the converted FP8 model.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for quantization (default: cuda:0).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Safetensors files to convert in parallel (default: 1).",
    )
    return parser.parse_args()


def require_base_model(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"Base model directory does not exist: {model_dir}")
    if any((model_dir / marker).is_file() for marker in ADAPTER_MARKERS):
        raise SystemExit(
            f"{model_dir} appears to be a LoRA adapter, not a full base model. "
            "Pass the shared Qwen3-4B base-model directory to --model."
        )
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"{model_dir} is not a complete Hugging Face model directory; missing: {', '.join(missing)}"
        )
    if not list(model_dir.glob("*.safetensors")):
        raise SystemExit(f"{model_dir} has no safetensors model weights.")


def require_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory must be new or empty to prevent mixing model versions: {output_dir}"
        )


def verify_output(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"FP8 export is incomplete; missing: {', '.join(missing)}")
    if not list(output_dir.glob("*.safetensors")):
        raise RuntimeError("FP8 export is incomplete; no safetensors weights were written.")

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    if not quantization:
        raise RuntimeError("FP8 export is missing quantization_config in config.json.")
    print(
        json.dumps(
            {
                "fp8_model_dir": str(output_dir),
                "quantization_config": quantization,
                "safetensors_files": len(list(output_dir.glob("*.safetensors"))),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")
    require_base_model(model_dir)
    require_empty_output(output_dir)

    try:
        from llmcompressor import model_free_ptq
    except ImportError as error:
        raise SystemExit(
            "llmcompressor is required. Create a dedicated conversion environment and install it with "
            "'pip install llmcompressor'."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting shared base model to FP8_DYNAMIC: {model_dir}", flush=True)
    try:
        model_free_ptq(
            model_stub=model_dir,
            save_directory=output_dir,
            scheme="FP8_DYNAMIC",
            ignore=["lm_head"],
            device=args.device,
            max_workers=args.max_workers,
        )
    except Exception:
        if output_dir.exists() and not any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
        raise
    verify_output(output_dir)


if __name__ == "__main__":
    main()
