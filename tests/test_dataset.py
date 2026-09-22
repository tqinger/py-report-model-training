import json
from pathlib import Path

from tcm_qwen_eval.dataset import (
    HOLISTIC_REPORT_SECTION_TASKS,
    SUPPORTED_DOMAINS,
    grouped_split,
    load_holistic_section_sft_splits,
    load_examples,
    load_jsonl_sft_splits,
)
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
    expected_count = sum(
        len(list((Path("data") / domain).glob("*.json"))) for domain in SUPPORTED_DOMAINS
    )
    assert len(examples) == expected_count
    assert {"constitution_combined_analysis", "tongue_daily_advice", "tongue_integrated_analysis"} <= {
        example.task for example in examples
    }


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


def test_jsonl_sft_loader_preserves_the_supplied_train_validation_split(tmp_path: Path):
    data_dir = tmp_path / "generated-sft"
    data_dir.mkdir()
    train = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Training"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    validation = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Validation"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    (data_dir / "train.jsonl").write_text(json.dumps(train, ensure_ascii=False) + "\n", encoding="utf-8")
    (data_dir / "val.jsonl").write_text(json.dumps(validation, ensure_ascii=False) + "\n", encoding="utf-8")

    splits = load_jsonl_sft_splits(
        data_dir, domain="generated", task="generated_analysis"
    )

    assert list(splits) == ["train", "validation"]
    assert [example.user for example in splits["train"]] == ["Training"]
    assert [example.user for example in splits["validation"]] == ["Validation"]
    assert splits["train"][0].id.endswith("train/000001")
    assert splits["validation"][0].task == "generated_analysis"


def test_jsonl_sft_loader_supports_system_free_user_assistant_records(tmp_path: Path):
    data_dir = tmp_path / "system-free-sft"
    data_dir.mkdir()
    record = {
        "messages": [
            {"role": "user", "content": "任务：生成主诉"},
            {"role": "assistant", "content": "胸痛3天"},
        ]
    }
    for filename in ("train.jsonl", "val.jsonl"):
        (data_dir / filename).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    splits = load_jsonl_sft_splits(data_dir, domain="case-polish", task="chief_complaint")

    example = splits["train"][0]
    assert example.system == ""
    assert example.user == "任务：生成主诉"
    assert example.reference == "胸痛3天"


def test_holistic_section_loader_creates_a_deterministic_source_grouped_split(tmp_path: Path):
    data_dir = tmp_path / "第一二部分_形体与心理"
    data_dir.mkdir()
    for index in range(100):
        payload = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {
                    "role": "user",
                    "content": (
                        "请生成报告第一、二部分。\n"
                        "四诊信息：\n"
                        f"患者编号：{index}"
                    ),
                },
                {"role": "assistant", "content": "一、形体特征\n核心病机：示例"},
            ]
        }
        (data_dir / f"{index:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    first = load_holistic_section_sft_splits(data_dir)
    second = load_holistic_section_sft_splits(data_dir)

    assert set(first) == {"train", "validation"}
    assert [len(first[name]) for name in ("train", "validation")] == [99, 1]
    assert {item.id for item in first["validation"]} == {item.id for item in second["validation"]}
    assert {item.task for item in first["train"]} == {"report_parts_1_2"}
    assert "report_parts_1_2" in HOLISTIC_REPORT_SECTION_TASKS
    assert {item.group_id for item in first["train"]}.isdisjoint(
        {item.group_id for item in first["validation"]}
    )


def test_holistic_section_loader_rejects_mixed_section_tasks(tmp_path: Path):
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    for index, prompt in enumerate(("请生成报告第一、二部分。", "请生成报告第三、四部分。")):
        payload = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": f"{prompt}\n四诊信息：患者编号：{index}"},
                {"role": "assistant", "content": "示例输出"},
            ]
        }
        (data_dir / f"{index}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    try:
        load_holistic_section_sft_splits(data_dir)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("mixed section tasks must be rejected")
