from pathlib import Path

from tcm_qwen_eval.dataset import grouped_split, load_examples


def test_data_loads_and_has_expected_tasks():
    examples = load_examples(Path("data"))
    assert len(examples) == 7664
    assert len({example.task for example in examples}) == 11


def test_grouped_split_is_complete_and_disjoint():
    examples = load_examples(Path("data"))
    split = grouped_split(examples)
    all_groups = set().union(*map(set, split.values()))
    assert len(all_groups) == len({example.group_id for example in examples})
    assert not (set(split["train"]) & set(split["validation"]))
    assert not (set(split["train"]) & set(split["test"]))
