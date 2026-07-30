from pathlib import Path

from tcm_qwen_eval.dataset import load_examples
from tcm_qwen_eval.dynamic_input import extract_dynamic_input, task_label


def test_extracts_nonempty_dynamic_data_for_every_training_sample():
    examples = load_examples(Path("data"))
    assert {task_label(example) for example in examples}
    for example in examples:
        value = extract_dynamic_input(example)
        assert value
        assert "你是一位" not in value


def test_dynamic_input_excludes_fixed_report_template_sections():
    example = next(
        item
        for item in load_examples(Path("data"))
        if item.task == "report_parts_3_4"
    )
    value = extract_dynamic_input(example)
    assert value.startswith("年龄：")
    assert "严格按以下格式输出" not in value
