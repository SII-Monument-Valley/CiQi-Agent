from __future__ import annotations

import ast
import json
import re
from typing import Any

from .types import NativeToolCall

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
TOOL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def extract_answer(text: str) -> str | None:
    answers = ANSWER_PATTERN.findall(text or "")
    return answers[-1].strip() if answers else None


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    value: Any = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(value) if isinstance(value, str) else value
            if isinstance(value, str):
                value = parser(value)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    raise ValueError("Tool arguments must be a JSON object")


def extract_hermes_tool_calls(text: str) -> tuple[str, tuple[NativeToolCall, ...]]:
    calls: list[NativeToolCall] = []
    for raw in TOOL_PATTERN.findall(text or ""):
        try:
            payload = json.loads(raw.strip())
            name = payload.get("name")
            if isinstance(name, str) and name:
                calls.append(
                    NativeToolCall(
                        name=name,
                        arguments=parse_tool_arguments(payload.get("arguments", {})),
                    )
                )
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return TOOL_PATTERN.sub("", text or "").strip(), tuple(calls)
