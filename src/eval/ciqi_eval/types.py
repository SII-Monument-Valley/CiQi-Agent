from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    category: str
    question: str
    image_paths: tuple[Path, ...]
    expected: Any
    options: Any = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeToolCall:
    name: str
    arguments: Any
    call_id: str | None = None


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_calls: tuple[NativeToolCall, ...] = ()


@dataclass
class ToolContext:
    images: list[Image.Image]
    ratios: list[tuple[float, float]]


@dataclass(frozen=True)
class ToolOutput:
    images: tuple[Image.Image, ...]
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    sample_id: str
    split: str
    category: str
    question: str
    image_paths: tuple[str, ...]
    expected: Any
    prediction: str
    extracted_answer: str | None
    status: str
    score: float
    trace: tuple[dict[str, Any], ...]
    conversation: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "id": self.sample_id,
            "split": self.split,
            "category": self.category,
            "question": self.question,
            "images": list(self.image_paths),
            "expected": self.expected,
            "prediction": self.prediction,
            "extracted_answer": self.extracted_answer,
            "status": self.status,
            "score": self.score,
            "trace": list(self.trace),
            "error": self.error,
        }
        if self.conversation:
            payload["conversation"] = list(self.conversation)
        return payload
