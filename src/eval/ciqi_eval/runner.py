from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import EvaluationConfig
from .conversation import ToolConversation
from .io import read_records, resolve_image_paths, write_json, write_results
from .model import ChatModel, build_chat_model
from .parsing import extract_answer, extract_hermes_tool_calls, parse_tool_arguments
from .tasks import EvaluationTask, get_task
from .tooling import ToolRegistry, build_tool_registry
from .types import EvaluationResult, EvaluationSample, NativeToolCall, ToolContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    output_directory: Path
    metrics: dict[str, Any]


class EvaluationRunner:
    """Orchestrates datasets, model turns, tools, traces, and deterministic outputs."""

    def __init__(
        self,
        config: EvaluationConfig,
        *,
        model: ChatModel | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.task: EvaluationTask = get_task(config.task)
        self.model = model or build_chat_model(config.model, config.generation)
        self.tools = tools or build_tool_registry(
            config.tools, min_dimension=config.images.factor
        )
        self.run_directory = config.output.directory / config.name

    def _sample(
        self, split: str, index: int, record: dict[str, Any]
    ) -> EvaluationSample:
        dataset = self.config.datasets[split]
        return EvaluationSample(
            sample_id=str(record.get("id") or f"{split}-{index:06d}"),
            category=str(record.get("type") or "unknown"),
            question=self.task.prompt(record, self.config.prompt),
            image_paths=resolve_image_paths(record, dataset),
            expected=self.task.expected(record),
            options=record.get("options"),
            raw=record,
        )

    async def _evaluate_sample(
        self, split: str, sample: EvaluationSample
    ) -> EvaluationResult:
        started_at = time.perf_counter()
        trace: list[dict[str, Any]] = []
        prediction = ""
        extracted: str | None = None
        status = "error"
        error: str | None = None
        conversation: ToolConversation | None = None

        try:
            conversation = ToolConversation(
                system_prompt=self.config.prompt.system,
                user_prompt=sample.question,
                image_paths=sample.image_paths,
                images=self.config.images,
            )
            context = ToolContext(
                images=conversation.images_raw, ratios=conversation.ratios
            )
            tool_steps = 0
            empty_responses = 0
            while True:
                model_started = time.perf_counter()
                turn = await self.model.complete(
                    conversation.messages, self.tools.schemas
                )
                prediction = turn.text
                extracted = extract_answer(turn.text)
                trace.append(
                    {
                        "type": "model_response",
                        "step": tool_steps + 1,
                        "text": turn.text,
                        "native_tool_calls": [call.name for call in turn.tool_calls],
                        "duration_ms": round(
                            (time.perf_counter() - model_started) * 1000
                        ),
                    }
                )
                if extracted is not None:
                    conversation.append_assistant(turn.text)
                    status = "success"
                    break

                visible_text, hermes_calls = extract_hermes_tool_calls(turn.text)
                call: NativeToolCall | None = (
                    turn.tool_calls[0] if turn.tool_calls else None
                )
                call = call or (hermes_calls[0] if hermes_calls else None)
                if call is None:
                    if turn.text.strip():
                        conversation.append_assistant(turn.text)
                        status = "completed_without_answer"
                        break
                    empty_responses += 1
                    if empty_responses > self.config.runtime.max_empty_responses:
                        error = "Model returned too many empty responses"
                        status = "empty_response"
                        break
                    continue

                if tool_steps >= self.config.runtime.max_tool_steps:
                    error = f"Tool call limit exceeded ({self.config.runtime.max_tool_steps})"
                    status = "tool_limit_exceeded"
                    break
                tool_steps += 1
                tool_call_id = call.call_id or f"ciqi-tool-{tool_steps}"
                tool_started = time.perf_counter()
                try:
                    arguments = parse_tool_arguments(call.arguments)
                    output = await self.tools.execute(call.name, context, arguments)
                    trace.append(
                        {
                            "type": "tool_result",
                            "step": tool_steps,
                            "tool": call.name,
                            "arguments": arguments,
                            "status": "success",
                            "metadata": output.metadata,
                            "returned_images": len(output.images),
                            "duration_ms": round(
                                (time.perf_counter() - tool_started) * 1000
                            ),
                        }
                    )
                    conversation.append_assistant(
                        visible_text,
                        tool_name=call.name,
                        arguments=arguments,
                        tool_call_id=tool_call_id,
                    )
                    conversation.append_tool_result(
                        output.text,
                        output.images,
                        tool_name=call.name,
                        tool_call_id=tool_call_id,
                    )
                except Exception as exc:  # noqa: BLE001 - plugin boundary must isolate sample failures
                    arguments = {}
                    try:
                        arguments = parse_tool_arguments(call.arguments)
                    except (TypeError, ValueError) as parse_error:
                        logger.debug(
                            "Invalid arguments from failed tool call: %s", parse_error
                        )
                    message = (
                        f"工具执行失败：{exc}。请勿重复无效调用；请依据已有证据继续分析，"
                        "并将最终答案放在 <answer>...</answer> 中。"
                    )
                    trace.append(
                        {
                            "type": "tool_result",
                            "step": tool_steps,
                            "tool": call.name,
                            "arguments": arguments,
                            "status": "error",
                            "error": str(exc),
                            "duration_ms": round(
                                (time.perf_counter() - tool_started) * 1000
                            ),
                        }
                    )
                    conversation.append_assistant(
                        visible_text,
                        tool_name=call.name,
                        arguments=arguments,
                        tool_call_id=tool_call_id,
                    )
                    conversation.append_tool_result(
                        message,
                        tool_name=call.name,
                        tool_call_id=tool_call_id,
                    )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            status = "error"
            if not self.config.runtime.continue_on_error:
                raise

        score = (
            self.task.score(sample.expected, extracted, prediction)
            if status != "error"
            else 0.0
        )
        trace.append(
            {
                "type": "sample_completed",
                "status": status,
                "duration_ms": round((time.perf_counter() - started_at) * 1000),
            }
        )
        exported = (
            conversation.export()
            if conversation and self.config.output.save_conversations
            else ()
        )
        return EvaluationResult(
            sample_id=sample.sample_id,
            split=split,
            category=sample.category,
            question=sample.question,
            image_paths=tuple(str(path) for path in sample.image_paths),
            expected=sample.expected,
            prediction=prediction,
            extracted_answer=extracted,
            status=status,
            score=score,
            trace=tuple(trace),
            conversation=exported,
            error=error,
        )

    async def _safe_evaluate_record(
        self,
        split: str,
        index: int,
        record: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, EvaluationResult]:
        async with semaphore:
            try:
                sample = self._sample(split, index, record)
                return index, await self._evaluate_sample(split, sample)
            except Exception as exc:
                if not self.config.runtime.continue_on_error:
                    raise
                sample_id = str(record.get("id") or f"{split}-{index:06d}")
                result = EvaluationResult(
                    sample_id=sample_id,
                    split=split,
                    category=str(record.get("type") or "unknown"),
                    question=str(record.get("question") or ""),
                    image_paths=(),
                    expected=record.get("answer", record.get("ground_truth")),
                    prediction="",
                    extracted_answer=None,
                    status="error",
                    score=0.0,
                    trace=(
                        {
                            "type": "sample_completed",
                            "status": "error",
                            "duration_ms": 0,
                        },
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
                return index, result

    async def _run_split(
        self, split: str, records: list[dict[str, Any]]
    ) -> list[EvaluationResult]:
        semaphore = asyncio.Semaphore(self.config.runtime.concurrency)
        pending = [
            asyncio.create_task(
                self._safe_evaluate_record(split, index, record, semaphore)
            )
            for index, record in enumerate(records)
        ]
        ordered: list[EvaluationResult | None] = [None] * len(pending)
        for completed, task in enumerate(asyncio.as_completed(pending), start=1):
            index, result = await task
            ordered[index] = result
            logger.info("%s: %d/%d complete", split, completed, len(pending))
        return [item for item in ordered if item is not None]

    @staticmethod
    def _metrics(results: list[EvaluationResult]) -> dict[str, Any]:
        score = (
            sum(result.score for result in results) / len(results) if results else 0.0
        )
        return {
            "samples": len(results),
            "successes": sum(result.status == "success" for result in results),
            "errors": sum(result.status == "error" for result in results),
            "score": round(score, 6),
        }

    @classmethod
    def _category_metrics(
        cls, results: list[EvaluationResult]
    ) -> dict[str, dict[str, Any]]:
        categories = sorted({result.category for result in results})
        return {
            category: cls._metrics(
                [result for result in results if result.category == category]
            )
            for category in categories
        }

    async def run(
        self, *, limit: int | None = None, category: str | None = None
    ) -> RunSummary:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        selected_category = category.strip() if category else None
        loaded = {
            split: read_records(dataset.path)
            for split, dataset in self.config.datasets.items()
        }
        available_categories = sorted(
            {
                str(record.get("type") or "unknown")
                for records in loaded.values()
                for record in records
            }
        )
        if selected_category:
            matches = {item.casefold(): item for item in available_categories}
            canonical = matches.get(selected_category.casefold())
            if canonical is None:
                raise ValueError(
                    f"Unknown category {selected_category!r}; available categories: "
                    + ", ".join(available_categories)
                )
            selected_category = canonical

        self.run_directory.mkdir(parents=True, exist_ok=True)
        if self.config.output.copy_config:
            public_yaml = yaml.safe_dump(
                self.config.public_dict(), allow_unicode=True, sort_keys=False
            )
            (self.run_directory / "resolved-config.yaml").write_text(
                public_yaml, encoding="utf-8"
            )

        split_metrics: dict[str, Any] = {}
        all_results: list[EvaluationResult] = []
        for split, records in loaded.items():
            if selected_category:
                records = [
                    record
                    for record in records
                    if str(record.get("type") or "unknown") == selected_category
                ]
            if limit is not None:
                records = records[:limit]
            logger.info("Evaluating split %s (%d samples)", split, len(records))
            results = await self._run_split(split, records)
            write_results(self.run_directory / f"results-{split}.jsonl", results)
            split_metrics[split] = {
                **self._metrics(results),
                "categories": self._category_metrics(results),
            }
            all_results.extend(results)

        metrics = {
            "schema_version": 1,
            "experiment": self.config.name,
            "task": self.config.task,
            "model": self.config.model.name,
            "tools": list(self.tools.names),
            "category_filter": selected_category,
            "available_categories": available_categories,
            **self._metrics(all_results),
            "categories": self._category_metrics(all_results),
            "splits": split_metrics,
        }
        write_json(self.run_directory / "summary.json", metrics)
        return RunSummary(output_directory=self.run_directory, metrics=metrics)
