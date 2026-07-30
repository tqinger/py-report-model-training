from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EXPECTED_ROLES = ("system", "user", "assistant")
ACTUAL_TRAINING_DATA_DIR = Path(r"D:\shanjiyun\py-report-system\data\training\llm_calls")


def default_data_dir() -> Path:
    """Use the report system's source data when available; keep a local fallback for portability."""
    configured = os.environ.get("TCM_TRAINING_DATA_DIR")
    if configured:
        return Path(configured)
    if ACTUAL_TRAINING_DATA_DIR.is_dir():
        return ACTUAL_TRAINING_DATA_DIR
    return Path("data")


@dataclass(frozen=True)
class Example:
    id: str
    domain: str
    task: str
    group_id: str
    messages: list[dict[str, str]]

    @property
    def system(self) -> str:
        return self.messages[0]["content"]

    @property
    def user(self) -> str:
        return self.messages[1]["content"]

    @property
    def reference(self) -> str:
        return self.messages[2]["content"]


def _task_name(domain: str, user: str) -> str:
    first_line = user.splitlines()[0].strip()
    task_markers = {
        "case-polish": (
            ("前后重复", "deduplicate_polish"),
            ("主病", "chief_complaint"),
            ("疾病条目", "present_illness_polish"),
        ),
        "constitution-analysis": (
            ("体质表现", "constitution_manifestation"),
            ("调理方向", "constitution_regulation"),
            ("一句综合概括", "constitution_summary"),
        ),
        "holistic-tcm-report": (
            ("第一、二部分", "report_parts_1_2"),
            ("第三、四部分", "report_parts_3_4"),
            ("第五部分", "report_part_5"),
        ),
        "tongue-analysis": (
            ("日常调养建议", "tongue_daily_advice"),
            ("综合分析", "tongue_integrated_analysis"),
        ),
    }
    for marker, name in task_markers[domain]:
        if marker in first_line:
            return name
    raise ValueError(f"Cannot classify task for {domain}: {first_line!r}")


def _group_source(domain: str, user: str) -> str:
    """Extract the source facts shared by task variants for split grouping."""
    markers = {
        "case-polish": ("疾病条目", "主病症状描述", "待去重文稿"),
        "constitution-analysis": ("体质辨识信息：",),
        "holistic-tcm-report": ("四诊信息：",),
        # The age/sex reference occurs before the tongue label and is part of the source
        # context for both tongue tasks. Keep it in the fingerprint rather than grouping
        # all patients with the same tongue features together.
        "tongue-analysis": ("参考信息（性别参考：",),
    }
    starts = [user.rfind(marker) for marker in markers[domain]]
    start = max(starts)
    # The task prompts place source facts after their final labelled marker. Fall back to the
    # complete user message so malformed data remains deterministically grouped and auditable.
    source = user[start:] if start >= 0 else user
    return "\n".join(line.strip() for line in source.splitlines() if line.strip())


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_examples(data_dir: Path) -> list[Example]:
    examples: list[Example] = []
    for domain_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        domain = domain_dir.name
        for path in sorted(domain_dir.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            messages = record.get("messages")
            if not isinstance(messages, list) or tuple(item.get("role") for item in messages) != EXPECTED_ROLES:
                raise ValueError(f"{path}: expected system/user/assistant messages")
            normalized = [
                {"role": message["role"], "content": str(message.get("content", "")).strip()}
                for message in messages
            ]
            if any(not message["content"] for message in normalized):
                raise ValueError(f"{path}: empty message content")
            user = normalized[1]["content"]
            task = _task_name(domain, user)
            group_id = f"{domain}-{_hash(_group_source(domain, user))[:16]}"
            examples.append(
                Example(
                    id=f"{domain}/{path.stem}",
                    domain=domain,
                    task=task,
                    group_id=group_id,
                    messages=normalized,
                )
            )
    if not examples:
        raise ValueError(f"No JSON examples found in {data_dir}")
    return examples


def select_representative_examples(
    examples: list[Example], samples_per_task: int = 5, seed: int = 20260729
) -> list[Example]:
    """Select deterministic, source-distinct examples for manual review."""
    by_task: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_task[example.task].append(example)

    selected: list[Example] = []
    for task, candidates in sorted(by_task.items()):
        # One representative per group prevents repeated source facts from dominating a task.
        representatives: dict[str, Example] = {}
        for candidate in sorted(candidates, key=lambda item: item.id):
            representatives.setdefault(candidate.group_id, candidate)
        ranked = sorted(
            representatives.values(),
            key=lambda item: _hash(f"{seed}:{task}:{item.id}"),
        )
        if len(ranked) < samples_per_task:
            raise ValueError(f"{task} has only {len(ranked)} distinct groups")
        selected.extend(ranked[:samples_per_task])
    return selected


def grouped_split(
    examples: list[Example], seed: int = 20260729
) -> dict[str, list[str]]:
    """Return a deterministic 80/10/10 split at source-group level for later SFT."""
    group_ids = sorted({item.group_id for item in examples})
    random.Random(seed).shuffle(group_ids)
    total = len(group_ids)
    train_end = round(total * 0.8)
    validation_end = train_end + round(total * 0.1)
    return {
        "train": group_ids[:train_end],
        "validation": group_ids[train_end:validation_end],
        "test": group_ids[validation_end:],
    }
