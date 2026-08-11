import json
from pathlib import Path

from tcm_qwen_eval.dataset import grouped_split, load_examples
from tcm_qwen_eval.tongue_qlora import (
    split_constitution_examples,
    split_manifest,
    split_tongue_examples,
)


def _write_conversation(path: Path, round_number: int) -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "系统提示"},
            {
                "role": "user",
                "content": (
                    "请生成舌面综合分析与日常调养建议。\n"
                    f"患者信息：性别女，年龄{20 + round_number}岁\n"
                    "舌象特征：淡红舌、正常舌形、薄白苔"
                ),
            },
            {"role": "assistant", "content": "舌面综合分析与日常调养建议。"},
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_data_loads_and_has_expected_tasks():
    examples = load_examples(Path("data"))
    assert len(examples) == 18725
    assert len({example.task for example in examples}) == 9


def test_grouped_split_is_complete_and_disjoint():
    examples = load_examples(Path("data"))
    split = grouped_split(examples)
    all_groups = set().union(*map(set, split.values()))
    assert len(all_groups) == len({example.group_id for example in examples})
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))


def test_conversations_data_uses_each_combination_in_all_three_splits(tmp_path: Path):
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    for round_number in range(1, 11):
        _write_conversation(
            conversations / f"tongue-00000-r{round_number:02d}-without.json", round_number
        )

    examples = load_examples(conversations)
    splits = split_tongue_examples(examples)

    assert {example.task for example in examples} == {"tongue_combined_analysis"}
    assert {example.group_id for example in examples} == {"tongue-analysis-00000"}
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [8, 1, 1]
    assert {example.id.rsplit("-r", maxsplit=1)[1][:2] for example in splits["train"]} == {
        f"{number:02d}" for number in range(1, 9)
    }
    assert splits["validation"][0].id.endswith("-r09-without")
    assert splits["test"][0].id.endswith("-r10-without")


def test_constitution_directory_loads_and_splits_without_source_leakage():
    examples = load_examples(Path("data/constitution-analysis"))
    splits = split_constitution_examples(examples)
    groups = {name: {example.group_id for example in rows} for name, rows in splits.items()}

    assert len(examples) == 11610
    assert {example.domain for example in examples} == {"constitution-analysis"}
    assert {example.task for example in examples} == {"constitution_combined_analysis"}
    assert sum(len(rows) for rows in splits.values()) == len(examples)
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    assert split_manifest(splits, 20260729)["domain"] == "constitution-analysis"
