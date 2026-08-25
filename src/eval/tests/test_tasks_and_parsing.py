from __future__ import annotations

from ciqi_eval.config import PromptConfig
from ciqi_eval.judge import extract_scores
from ciqi_eval.parsing import (
    extract_answer,
    extract_hermes_tool_calls,
    parse_tool_arguments,
)
from ciqi_eval.tasks import get_task


def test_selection_accepts_list_options_and_exact_label() -> None:
    task = get_task("selection")
    prompt = task.prompt(
        {"question": "请选择", "options": ["甲", "乙", "丙", "丁"]},
        PromptConfig(system="s", final_instruction="final"),
    )
    assert "A. 甲" in prompt
    assert "D. 丁" in prompt
    assert task.score("B", "B", "") == 1.0
    assert task.score("B", "C", "") == 0.0


def test_parses_answer_native_arguments_and_hermes_call() -> None:
    text = '分析<tool_call>{"name":"search_text","arguments":{"query":"康熙青花"}}</tool_call>'
    visible, calls = extract_hermes_tool_calls(text)
    assert visible == "分析"
    assert calls[0].name == "search_text"
    assert parse_tool_arguments(calls[0].arguments) == {"query": "康熙青花"}
    assert extract_answer("a <answer> B </answer>") == "B"


def test_judge_parser_rejects_incomplete_or_out_of_range_scores() -> None:
    valid = "".join(
        f"<{tag}>0.5</{tag}>"
        for tag in ("朝代", "皇帝", "窑口", "釉色", "纹饰", "器型")
    )
    assert extract_scores(valid) == {
        tag: 0.5 for tag in ("朝代", "皇帝", "窑口", "釉色", "纹饰", "器型")
    }
    assert extract_scores(valid.replace("<朝代>0.5", "<朝代>2.0")) is None
    assert extract_scores("<朝代>1</朝代>") is None
