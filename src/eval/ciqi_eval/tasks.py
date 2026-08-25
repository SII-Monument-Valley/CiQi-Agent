from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import PromptConfig

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _format_options(options: Any) -> str:
    if isinstance(options, Mapping):
        return "\n".join(f"{label}. {value}" for label, value in options.items())
    if isinstance(options, list):
        return "\n".join(
            f"{OPTION_LABELS[index]}. {value}" for index, value in enumerate(options)
        )
    raise ValueError("Selection samples require options as a mapping or list")


def _selection_answer(value: Any) -> str:
    match = re.search(r"\b([A-Z])\b", str(value).strip().upper())
    return match.group(1) if match else str(value).strip().upper()


@dataclass(frozen=True)
class EvaluationTask:
    name: str

    def expected(self, record: Mapping[str, Any]) -> Any:
        if self.name == "selection":
            if "answer" not in record:
                raise KeyError("Selection sample is missing 'answer'")
            return record["answer"]
        if "ground_truth" in record:
            return record["ground_truth"]
        if "answer" in record:
            return record["answer"]
        raise KeyError("QA sample is missing 'ground_truth' or 'answer'")

    def prompt(self, record: Mapping[str, Any], prompt: PromptConfig) -> str:
        if "question" not in record:
            raise KeyError("Sample is missing 'question'")
        parts = [prompt.user_prefix.strip(), str(record["question"]).strip()]
        if self.name == "selection":
            parts.append(_format_options(record.get("options")))
        parts.append(prompt.final_instruction.strip())
        return "\n".join(part for part in parts if part)

    def score(self, expected: Any, extracted: str | None, prediction: str) -> float:
        candidate = extracted if extracted is not None else prediction
        if self.name == "selection":
            return float(_selection_answer(expected) == _selection_answer(candidate))
        expected_text = " ".join(str(expected).split()).casefold()
        candidate_text = " ".join(str(candidate).split()).casefold()
        return float(bool(expected_text) and expected_text in candidate_text)


def get_task(name: str) -> EvaluationTask:
    if name not in {"selection", "qa"}:
        raise ValueError(f"Unsupported task: {name}")
    return EvaluationTask(name=name)
