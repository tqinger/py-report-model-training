from __future__ import annotations

import re

from tcm_qwen_eval.dataset import Example

TASK_LABELS = {
    "deduplicate_polish": "病例润色｜去重润色",
    "chief_complaint": "病例润色｜主诉",
    "present_illness_polish": "病例润色｜现病史润色",
    "constitution_manifestation": "体质分析｜体质表现",
    "constitution_regulation": "体质分析｜调理方向",
    "constitution_summary": "体质分析｜综合概括",
    "report_parts_1_2": "四诊报告｜第一、二部分",
    "report_parts_3_4": "四诊报告｜第三、四部分",
    "report_part_5": "四诊报告｜日常调护建议",
    "tongue_daily_advice": "舌诊分析｜日常调养建议",
    "tongue_integrated_analysis": "舌诊分析｜综合分析",
}


def _last_section(user: str, marker: str, terminator: str | None = None) -> str:
    start = user.rfind(marker)
    if start < 0:
        raise ValueError(f"Prompt does not contain expected marker {marker!r}")
    value = user[start + len(marker) :]
    if terminator:
        end = value.rfind(terminator)
        if end >= 0:
            value = value[:end]
    value = value.strip()
    if not value:
        raise ValueError(f"Prompt section {marker!r} is empty")
    return value


def _patient_context(user: str) -> str:
    match = re.search(r"## 患者信息\s*\n(?P<context>.*?)\n[（(]润色时须据此", user, re.DOTALL)
    return match.group("context").strip() if match else ""


def _case_dynamic_input(task: str, user: str) -> str:
    patient = _patient_context(user)
    if task == "present_illness_polish":
        source = _last_section(user, "## 本次输入", "请通读全部条目")
    elif task == "deduplicate_polish":
        source = _last_section(user, "## 待去重文稿", "请仅输出去重合并后")
    elif task == "chief_complaint":
        source = _last_section(user, "## 主病条目", "主诉：")
    else:
        raise ValueError(f"Unsupported case-polish task: {task}")
    parts = [part for part in (patient, source) if part]
    return "\n\n".join(parts)


def _tongue_dynamic_input(user: str) -> str:
    features = _last_section(user, "舌象特征：")
    reference = re.search(r"参考信息[（(](?P<value>.*?)[）)]", user)
    parts = []
    if reference:
        parts.append(reference.group("value").strip())
    parts.append(features)
    return "\n".join(parts)


def extract_dynamic_input(example: Example) -> str:
    """Return only the values injected into an otherwise fixed prompt template."""
    if example.domain == "tongue-analysis":
        return _tongue_dynamic_input(example.user)
    if example.domain == "constitution-analysis":
        return _last_section(example.user, "体质辨识信息：")
    if example.domain == "holistic-tcm-report":
        return _last_section(example.user, "四诊信息：")
    if example.domain == "case-polish":
        return _case_dynamic_input(example.task, example.user)
    raise ValueError(f"Unsupported domain: {example.domain}")


def task_label(example: Example) -> str:
    return TASK_LABELS[example.task]
