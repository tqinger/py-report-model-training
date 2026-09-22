from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EXPECTED_ROLES = ("system", "user", "assistant")
USER_ASSISTANT_ROLES = ("user", "assistant")
SUPPORTED_ROLE_SEQUENCES = (EXPECTED_ROLES, USER_ASSISTANT_ROLES)
ACTUAL_TRAINING_DATA_DIR = Path(r"D:\shanjiyun\py-report-system\data\training\llm_calls")
CONVERSATIONS_DIRECTORY = "conversations"
TONGUE_DOMAIN = "tongue-analysis"
TONGUE_COMBINED_TASK = "tongue_combined_analysis"
CONSTITUTION_COMBINED_TASK = "constitution_combined_analysis"
CONVERSATION_FILENAME = re.compile(r"tongue-(?P<combination>\d+)-r(?P<round>0[1-9]|10)-without")
SUPPORTED_DOMAINS = frozenset(
    {"case-polish", "constitution-analysis", "holistic-tcm-report", TONGUE_DOMAIN}
)
HOLISTIC_REPORT_SECTION_TASKS = frozenset(
    {"report_parts_1_2", "report_parts_3_4", "report_part_5"}
)


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
        return self.messages[0]["content"] if self.messages[0]["role"] == "system" else ""

    @property
    def user(self) -> str:
        return self.messages[-2]["content"]

    @property
    def reference(self) -> str:
        return self.messages[-1]["content"]


def _task_name(domain: str, user: str) -> str:
    first_line = user.splitlines()[0].strip()
    if domain == "constitution-analysis" and first_line.startswith("根据以下体质辨识信息生成 JSON"):
        return CONSTITUTION_COMBINED_TASK
    task_markers = {
        "case-polish": (
            ("任务：润色现病史", "present_illness_polish"),
            ("任务：生成主诉", "chief_complaint"),
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


def _normalized_messages(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or not all(isinstance(item, dict) for item in messages)
        or tuple(item.get("role") for item in messages) not in SUPPORTED_ROLE_SEQUENCES
    ):
        raise ValueError(f"{path}: expected system/user/assistant or user/assistant messages")
    normalized = [
        {"role": message["role"], "content": str(message.get("content", "")).strip()}
        for message in messages
    ]
    if any(not message["content"] for message in normalized):
        raise ValueError(f"{path}: empty message content")
    return normalized


def _normalized_jsonl_messages(path: Path):
    """Yield validated Chat SFT messages from a UTF-8 JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not valid SFT records")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            messages = record.get("messages") if isinstance(record, dict) else None
            if (
                not isinstance(messages, list)
                or any(not isinstance(item, dict) for item in messages)
                or tuple(item.get("role") for item in messages) not in SUPPORTED_ROLE_SEQUENCES
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected system/user/assistant or user/assistant messages"
                )
            normalized = [
                {"role": message["role"], "content": str(message.get("content", "")).strip()}
                for message in messages
            ]
            if any(not message["content"] for message in normalized):
                raise ValueError(f"{path}:{line_number}: empty message content")
            yield line_number, normalized


def load_jsonl_sft_splits(
    data_dir: Path,
    *,
    domain: str,
    task: str,
) -> dict[str, list[Example]]:
    """Load a supplied train/validation Chat SFT split without re-splitting it.

    The generated datasets in this project already provide ``train.jsonl`` and
    ``val.jsonl``.  Preserving that split makes training reproducible and avoids
    moving records out of the supplied validation set.
    """
    split_paths = {
        "train": data_dir / "train.jsonl",
        "validation": data_dir / "val.jsonl",
    }
    result: dict[str, list[Example]] = {}
    for split_name, path in split_paths.items():
        if not path.is_file():
            raise ValueError(f"{data_dir}: missing required {path.name}")
        examples: list[Example] = []
        for line_number, messages in _normalized_jsonl_messages(path):
            user = messages[-2]["content"]
            examples.append(
                Example(
                    id=f"{data_dir.name}/{split_name}/{line_number:06d}",
                    domain=domain,
                    task=task,
                    group_id=f"{domain}-{_hash(user)[:16]}",
                    messages=messages,
                )
            )
        if not examples:
            raise ValueError(f"{path}: contains no SFT records")
        result[split_name] = examples
    return result


def load_holistic_section_sft_splits(
    data_dir: Path,
    *,
    seed: int = 20260817,
    validation_fraction: float = 0.01,
) -> dict[str, list[Example]]:
    """Load one raw holistic-report section and create a source-grouped train/val split.

    The supplied simplified-prompt data keeps one Chat SFT record per JSON file
    and separates report sections into sibling directories.  A section adapter
    must train on only one task, while records sharing the same four-diagnosis
    source facts must remain in the same split.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"{data_dir}: contains no JSON SFT records")
    examples = _load_domain_examples("holistic-tcm-report", paths)
    tasks = {example.task for example in examples}
    if len(tasks) != 1 or not tasks <= HOLISTIC_REPORT_SECTION_TASKS:
        raise ValueError(
            f"{data_dir}: expected exactly one holistic report section task, got {sorted(tasks)}"
        )

    group_ids = sorted({example.group_id for example in examples})
    if len(group_ids) < 2:
        raise ValueError(f"{data_dir}: at least two distinct source groups are required")
    random.Random(seed).shuffle(group_ids)
    validation_count = max(1, round(len(group_ids) * validation_fraction))
    validation_groups = set(group_ids[:validation_count])
    splits = {
        "train": [example for example in examples if example.group_id not in validation_groups],
        "validation": [example for example in examples if example.group_id in validation_groups],
    }
    if not splits["train"] or not splits["validation"]:
        raise ValueError(f"{data_dir}: split produced an empty train or validation set")
    return splits


def _load_conversation_examples(data_dir: Path) -> list[Example]:
    """Load the flat, exhaustive tongue-combination conversation dataset."""
    examples: list[Example] = []
    for path in sorted(data_dir.glob("*.json")):
        match = CONVERSATION_FILENAME.fullmatch(path.stem)
        if not match:
            raise ValueError(
                f"{path}: expected a filename like tongue-00000-r01-without.json"
            )
        examples.append(
            Example(
                id=f"{TONGUE_DOMAIN}/{path.stem}",
                domain=TONGUE_DOMAIN,
                task=TONGUE_COMBINED_TASK,
                group_id=f"{TONGUE_DOMAIN}-{match['combination']}",
                messages=_normalized_messages(path),
            )
        )
    return examples


def _load_domain_examples(domain: str, paths: list[Path]) -> list[Example]:
    """Load the JSON examples for one of the legacy report domains."""
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported training-data domain: {domain}")
    examples: list[Example] = []
    for path in paths:
        normalized = _normalized_messages(path)
        user = normalized[-2]["content"]
        examples.append(
            Example(
                id=f"{domain}/{path.stem}",
                domain=domain,
                task=_task_name(domain, user),
                group_id=f"{domain}-{_hash(_group_source(domain, user))[:16]}",
                messages=normalized,
            )
        )
    return examples


def load_examples(data_dir: Path) -> list[Example]:
    """Load a legacy data root, one legacy domain, or the flat conversations dataset."""
    if data_dir.name == CONVERSATIONS_DIRECTORY:
        examples = _load_conversation_examples(data_dir)
    else:
        direct_paths = sorted(data_dir.glob("*.json"))
        if direct_paths:
            examples = _load_domain_examples(data_dir.name, direct_paths)
        else:
            examples = []
            for domain_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
                # Conversations are an alternative data root, not part of the legacy multi-domain set.
                if domain_dir.name == CONVERSATIONS_DIRECTORY:
                    continue
                # Data roots can also contain smoke fixtures and separate task directories.
                # Only the four established legacy domains participate in this combined loader.
                if domain_dir.name not in SUPPORTED_DOMAINS:
                    continue
                # Generated JSONL datasets are loaded by ``load_jsonl_sft_splits`` so their
                # supplied train/validation split remains intact.  Do not mistake summary.json
                # for a legacy single-record training example when scanning the old data root.
                if (domain_dir / "train.jsonl").is_file() or (domain_dir / "val.jsonl").is_file():
                    continue
                examples.extend(
                    _load_domain_examples(domain_dir.name, sorted(domain_dir.glob("*.json")))
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
