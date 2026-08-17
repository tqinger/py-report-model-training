from pathlib import Path

from tcm_qwen_eval.dataset import Example, load_examples
from tcm_qwen_eval.dynamic_input import extract_dynamic_input, task_label


def test_extracts_nonempty_dynamic_data_for_every_training_sample():
    examples = load_examples(Path("data"))
    assert {task_label(example) for example in examples}
    for example in examples:
        value = extract_dynamic_input(example)
        assert value
        assert "你是一位" not in value


def test_dynamic_input_excludes_fixed_report_template_sections():
    example = Example(
        id="holistic-example",
        domain="holistic-tcm-report",
        task="report_parts_3_4",
        group_id="holistic-example",
        messages=[
            {"role": "system", "content": "系统"},
            {
                "role": "user",
                "content": "严格按以下格式输出。\n四诊信息：\n年龄：35岁\n主诉：睡眠欠佳",
            },
            {"role": "assistant", "content": "报告"},
        ],
    )
    value = extract_dynamic_input(example)
    assert value.startswith("年龄：")
    assert "严格按以下格式输出" not in value
