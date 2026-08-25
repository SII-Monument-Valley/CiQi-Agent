from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ciqi_eval.config import evaluation_config_from_mapping
from ciqi_eval.runner import EvaluationRunner
from ciqi_eval.tooling import SEARCH_TEXT_SCHEMA, ToolRegistry
from ciqi_eval.types import ModelTurn, ToolContext, ToolOutput


class FakeToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[list[dict[str, Any]]] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.calls += 1
        self.seen_tools.append(tools)
        if self.calls == 1:
            return ModelTurn(
                text='需要检索。<tool_call>{"name":"search_text","arguments":{"query":"康熙"}}</tool_call>'
            )
        assert "检索证据" in str(messages[-1]["content"])
        return ModelTurn(text="综合证据，答案为 B。<answer>B</answer>")


class ImmediateAnswerModel:
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        return ModelTurn(text="<answer>B</answer>")


@pytest.mark.asyncio
async def test_runner_executes_tool_loop_and_writes_versioned_outputs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {"id": "s1", "question": "请选择", "options": ["甲", "乙"], "answer": "B"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = evaluation_config_from_mapping(
        {
            "version": 1,
            "experiment": {"name": "run", "task": "selection"},
            "datasets": {"test": {"path": str(dataset), "image_root": str(tmp_path)}},
            "model": {"name": "fake", "api_key": "EMPTY"},
            "tools": {
                "enabled": True,
                "names": ["search_text"],
                "rag": {"base_url": "http://unused.test"},
            },
            "runtime": {"concurrency": 2, "max_tool_steps": 2},
            "output": {"directory": str(tmp_path / "outputs")},
            "prompt": {"system": "system"},
        }
    )
    registry = ToolRegistry()

    async def search(context: ToolContext, arguments: dict[str, Any]) -> ToolOutput:
        assert arguments == {"query": "康熙"}
        return ToolOutput(images=(), text="检索证据：康熙", metadata={"success": True})

    registry.register(SEARCH_TEXT_SCHEMA, search)
    model = FakeToolCallingModel()
    summary = await EvaluationRunner(config, model=model, tools=registry).run()
    assert summary.metrics["score"] == 1.0
    assert summary.metrics["category_filter"] is None
    assert summary.metrics["categories"]["unknown"]["samples"] == 1
    assert model.seen_tools[0][0]["function"]["name"] == "search_text"

    result_path = summary.output_directory / "results-test.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["status"] == "success"
    assert result["extracted_answer"] == "B"
    assert any(event["type"] == "tool_result" for event in result["trace"])
    resolved = (summary.output_directory / "resolved-config.yaml").read_text(
        encoding="utf-8"
    )
    assert "api_key: <redacted>" in resolved


@pytest.mark.asyncio
async def test_runner_records_missing_image_as_sample_error(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "bad",
                "question": "q",
                "images": ["missing.jpg"],
                "ground_truth": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = evaluation_config_from_mapping(
        {
            "version": 1,
            "experiment": {"name": "errors", "task": "qa"},
            "datasets": {"test": {"path": str(dataset), "image_root": str(tmp_path)}},
            "model": {"name": "fake", "api_key": "EMPTY"},
            "tools": {"enabled": False},
            "output": {"directory": str(tmp_path / "outputs")},
            "prompt": {"system": "system"},
        }
    )
    summary = await EvaluationRunner(
        config, model=FakeToolCallingModel(), tools=ToolRegistry()
    ).run()
    result = json.loads(
        (summary.output_directory / "results-test.jsonl").read_text(encoding="utf-8")
    )
    assert result["status"] == "error"
    assert "Image does not exist" in result["error"]


@pytest.mark.asyncio
async def test_runner_filters_and_reports_dataset_categories(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    rows = [
        {
            "id": "color-1",
            "type": "color",
            "question": "q",
            "options": ["A", "B"],
            "answer": "B",
        },
        {
            "id": "color-2",
            "type": "color",
            "question": "q",
            "options": ["A", "B"],
            "answer": "B",
        },
        {
            "id": "dynasty-1",
            "type": "dynasty",
            "question": "q",
            "options": ["A", "B"],
            "answer": "B",
        },
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = evaluation_config_from_mapping(
        {
            "version": 1,
            "experiment": {"name": "categories", "task": "selection"},
            "datasets": {"test": {"path": str(dataset)}},
            "model": {"name": "fake", "api_key": "EMPTY"},
            "tools": {"enabled": False},
            "output": {"directory": str(tmp_path / "outputs")},
            "prompt": {"system": "system"},
        }
    )

    summary = await EvaluationRunner(
        config, model=ImmediateAnswerModel(), tools=ToolRegistry()
    ).run(category="COLOR")
    assert summary.metrics["category_filter"] == "color"
    assert summary.metrics["available_categories"] == ["color", "dynasty"]
    assert summary.metrics["samples"] == 2
    assert summary.metrics["successes"] == 2
    assert summary.metrics["categories"] == {
        "color": {"samples": 2, "successes": 2, "errors": 0, "score": 1.0}
    }
    result_rows = [
        json.loads(line)
        for line in (summary.output_directory / "results-test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["category"] for row in result_rows} == {"color"}

    with pytest.raises(ValueError, match="available categories: color, dynasty"):
        await EvaluationRunner(
            config, model=ImmediateAnswerModel(), tools=ToolRegistry()
        ).run(category="kiln")
