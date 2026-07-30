from pathlib import Path

from tcm_qwen_eval.dataset import (
    audit_examples,
    grouped_split,
    load_examples,
    select_baseline_examples,
)


def test_data_loads_and_has_expected_tasks():
    examples = load_examples(Path("data"))
    assert len(examples) == 7664
    assert len({example.task for example in examples}) == 11


def test_baseline_selection_is_group_distinct_and_complete():
    selected = select_baseline_examples(load_examples(Path("data")))
    assert len(selected) == 55
    for task in {example.task for example in selected}:
        task_examples = [example for example in selected if example.task == task]
        assert len(task_examples) == 5
        assert len({example.group_id for example in task_examples}) == 5


def test_grouped_split_is_complete_and_disjoint():
    examples = load_examples(Path("data"))
    split = grouped_split(examples)
    all_groups = set().union(*map(set, split.values()))
    assert len(all_groups) == len({example.group_id for example in examples})
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert len(audit_examples(examples)) == 4
