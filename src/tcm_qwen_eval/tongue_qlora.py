"""Data preparation and evaluation helpers for the Qwen3 tongue QLoRA adapter."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tcm_qwen_eval.dataset import (
    CONVERSATION_FILENAME,
    TONGUE_COMBINED_TASK,
    TONGUE_DOMAIN,
    Example,
    grouped_split,
    select_representative_examples,
)

TONGUE_TASKS = ("tongue_integrated_analysis", "tongue_daily_advice", TONGUE_COMBINED_TASK)
DEFAULT_SEED = 20260729


def tongue_examples(examples: list[Example]) -> list[Example]:
    """Return only supported tongue-analysis tasks."""
    selected = [example for example in examples if example.domain == TONGUE_DOMAIN]
    unexpected = {example.task for example in selected} - set(TONGUE_TASKS)
    if unexpected:
        raise ValueError(f"Unexpected tongue tasks: {sorted(unexpected)}")
    if not selected:
        raise ValueError("No tongue-analysis examples found")
    return selected


def _conversation_round(example: Example) -> int:
    filename = example.id.rsplit("/", maxsplit=1)[-1]
    match = CONVERSATION_FILENAME.fullmatch(filename)
    if not match:
        raise ValueError(f"{example.id}: expected a conversations dataset filename")
    return int(match["round"])


def _split_conversation_rounds(examples: list[Example]) -> dict[str, list[Example]]:
    """Keep every combination in every split by assigning its ten rounds deterministically."""
    by_group: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_group[example.group_id].append(example)

    expected_rounds = set(range(1, 11))
    result = {"train": [], "validation": [], "test": []}
    for group_id, group_examples in sorted(by_group.items()):
        rounds = {_conversation_round(example) for example in group_examples}
        if len(group_examples) != 10 or rounds != expected_rounds:
            raise ValueError(
                f"{group_id}: expected exactly one sample for each round r01 through r10"
            )
        for example in group_examples:
            split_name = "train" if _conversation_round(example) <= 8 else "validation"
            if _conversation_round(example) == 10:
                split_name = "test"
            result[split_name].append(example)
    return result


def split_tongue_examples(examples: list[Example], seed: int = DEFAULT_SEED) -> dict[str, list[Example]]:
    """Split original data by source, or exhaustive conversation data by its numbered rounds."""
    selected = tongue_examples(examples)
    if {example.task for example in selected} == {TONGUE_COMBINED_TASK}:
        return _split_conversation_rounds(selected)

    groups = grouped_split(selected, seed)
    result = {
        split_name: [example for example in selected if example.group_id in group_ids]
        for split_name, group_ids in groups.items()
    }
    all_ids = set().union(*(set(group_ids) for group_ids in groups.values()))
    if all_ids != {example.group_id for example in selected}:
        raise ValueError("Tongue split does not cover every source group")
    return result


def split_manifest(splits: dict[str, list[Example]], seed: int) -> dict[str, Any]:
    """Return an auditable, deterministic record of the selected split strategy."""
    tasks = {example.task for rows in splits.values() for example in rows}
    combined_only = tasks == {TONGUE_COMBINED_TASK}
    return {
        "seed": seed,
        "domain": TONGUE_DOMAIN,
        "tasks": sorted(tasks),
        "split_strategy": (
            "per-combination rounds r01-r08/train, r09/validation, r10/test"
            if combined_only
            else "source-grouped 80/10/10"
        ),
        "splits": {
            name: {
                "example_count": len(items),
                "group_count": len({item.group_id for item in items}),
                "group_ids": sorted({item.group_id for item in items}),
                "examples": [asdict(item) for item in items],
            }
            for name, items in splits.items()
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_model_source(model_name: str, cache_dir: Path) -> str:
    """Prefer an existing Hugging Face snapshot, falling back to the repository id."""
    direct_path = Path(model_name)
    if direct_path.is_dir():
        return str(direct_path)
    snapshot_root = cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir()) if snapshot_root.is_dir() else []
    return str(snapshots[-1]) if snapshots else model_name


def chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render the exact inference prefix, with Qwen3 thinking disabled."""
    prompt_messages = messages[:2]
    try:
        return tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def encode_sft_example(tokenizer: Any, example: Example, max_length: int) -> dict[str, list[int]]:
    """Encode one SFT example while masking all system and user prompt tokens."""
    prompt_ids = _token_ids(tokenizer, chat_prompt(tokenizer, example.messages))
    eos_token = tokenizer.eos_token or ""
    answer_ids = _token_ids(tokenizer, f"{example.reference}{eos_token}")
    if not answer_ids:
        raise ValueError(f"{example.id}: assistant answer encoded to no tokens")
    if len(prompt_ids) >= max_length:
        raise ValueError(f"{example.id}: prompt is {len(prompt_ids)} tokens, exceeding {max_length}")

    input_ids = (prompt_ids + answer_ids)[:max_length]
    answer_length = len(input_ids) - len(prompt_ids)
    if answer_length <= 0:
        raise ValueError(f"{example.id}: max length leaves no answer tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
    }


class TokenizedSFTDataset(Dataset):
    """Small in-memory token dataset suitable for a few thousand SFT records."""

    def __init__(self, records: list[dict[str, list[int]]]) -> None:
        if not records:
            raise ValueError("Tokenized dataset must not be empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


class CausalDataCollator:
    """Right-pad causal-LM inputs while preserving -100 prompt-label masking."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        padded: dict[str, list[list[int]]] = defaultdict(list)
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            padded["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            padded["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            padded["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in padded.items()}


def token_length_summary(records: list[dict[str, list[int]]]) -> dict[str, int]:
    lengths = sorted(len(record["input_ids"]) for record in records)
    if not lengths:
        raise ValueError("Cannot summarize an empty token collection")
    return {
        "min": lengths[0],
        "p50": lengths[(len(lengths) - 1) // 2],
        "p95": lengths[round((len(lengths) - 1) * 0.95)],
        "max": lengths[-1],
    }


def choose_manual_examples(test_examples: list[Example], seed: int = DEFAULT_SEED) -> list[Example]:
    """Choose five source-distinct, held-out examples per tongue subtask."""
    return select_representative_examples(test_examples, samples_per_task=5, seed=seed)


def tongue_format_check(task: str, output: str) -> tuple[bool, str]:
    """Check only constraints that are safe to evaluate automatically."""
    text = output.strip()
    if not text:
        return False, "输出为空"
    if any(marker in text for marker in ("```", "## ", "**")):
        return False, "包含 Markdown 标记"
    if task == "tongue_daily_advice":
        has_numbering = bool(re.search(r"(^|\n)\s*\d+[.、)]", text))
        return 60 <= len(text) <= 80 and not has_numbering, "要求 60–80 字且无编号"
    if task == "tongue_integrated_analysis":
        sentence_count = len(re.findall(r"[。！？]", text))
        no_title = not text.startswith(("舌面综合分析", "综合分析"))
        return 100 <= len(text) <= 120 and 2 <= sentence_count <= 3 and no_title, "要求 100–120 字、2–3 句、无标题"
    if task == TONGUE_COMBINED_TASK:
        has_advice = any(marker in text for marker in ("建议", "宜", "应"))
        return 120 <= len(text) <= 600 and has_advice, "要求包含分析与调养建议，且长度为 120–600 字"
    raise ValueError(f"Unsupported tongue task: {task}")
