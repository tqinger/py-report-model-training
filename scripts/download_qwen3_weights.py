"""Download a selected Qwen3 base model into the shared Hugging Face cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

QWEN3_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Qwen3 base model into the cache used by training and evaluation."
    )
    parser.add_argument(
        "--size",
        choices=QWEN3_MODELS,
        default="4B",
        help="Model parameter size to download (default: 4B).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/hf_cache"),
        help="Hugging Face cache directory shared with training and evaluation.",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face model revision, branch, or tag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = QWEN3_MODELS[args.size]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} to {args.cache_dir} ...", flush=True)
    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    print(f"Download complete: {snapshot_path}")


if __name__ == "__main__":
    main()
