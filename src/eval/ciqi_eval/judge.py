from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from openai import AsyncOpenAI

from .io import read_records, write_json

SCORE_TAGS = ("朝代", "皇帝", "窑口", "釉色", "纹饰", "器型")


JUDGE_PROMPT = """你是一位严谨的中国古陶瓷评审员。请比较参考答案与模型输出，并从以下六个维度保守评分：
朝代、皇帝、窑口、釉色、纹饰、器型。每项取 0 到 1；如果参考答案没有该维度，取 -1。
先简要说明理由，最后严格输出以下标签：
<朝代>分数</朝代>
<皇帝>分数</皇帝>
<窑口>分数</窑口>
<釉色>分数</釉色>
<纹饰>分数</纹饰>
<器型>分数</器型>

参考答案：{expected}
模型输出：{prediction}
"""


def extract_scores(text: str) -> dict[str, float] | None:
    result: dict[str, float] = {}
    for tag in SCORE_TAGS:
        match = re.search(rf"<{tag}>\s*(-?\d+(?:\.\d+)?)\s*</{tag}>", text)
        if not match:
            return None
        value = float(match.group(1))
        if value != -1 and not 0 <= value <= 1:
            return None
        result[tag] = value
    return result


async def _score_one(
    record: dict[str, Any],
    *,
    client: Any,
    model: str,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    max_tokens: int,
) -> dict[str, Any]:
    expected = record.get("expected", record.get("answer"))
    prediction = record.get("prediction", record.get("pred_ans", ""))
    prompt = JUDGE_PROMPT.format(expected=expected, prediction=prediction)
    last_text = ""
    async with semaphore:
        for attempt in range(max_attempts):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位严谨的瓷器鉴定评分专家。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                last_text = response.choices[0].message.content or ""
                scores = extract_scores(last_text)
                if scores is not None:
                    return {**record, "judge_scores": scores, "judge_status": "success"}
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose heterogeneous errors
                last_text = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < max_attempts:
                await asyncio.sleep(min(2**attempt, 8))
    return {
        **record,
        "judge_scores": None,
        "judge_status": "error",
        "judge_error": last_text[:500],
    }


async def score_file(
    input_path: Path,
    output_path: Path,
    judge_config: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    model = str(judge_config.get("model") or judge_config.get("model_name") or "judge")
    base_url = judge_config.get("base_url") or judge_config.get("api_url")
    api_key = str(judge_config.get("api_key") or "EMPTY")
    concurrency = int(
        judge_config.get("concurrency", judge_config.get("max_concurrency", 8))
    )
    max_attempts = int(judge_config.get("max_attempts", 3))
    max_tokens = int(judge_config.get("max_tokens", 2048))
    judge_client = client or AsyncOpenAI(
        api_key=api_key, base_url=base_url, max_retries=2
    )
    records = read_records(input_path)
    semaphore = asyncio.Semaphore(concurrency)
    scored = await asyncio.gather(
        *[
            _score_one(
                record,
                client=judge_client,
                model=model,
                semaphore=semaphore,
                max_attempts=max_attempts,
                max_tokens=max_tokens,
            )
            for record in records
        ]
    )
    if output_path.suffix == ".jsonl":
        text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in scored
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        write_json(output_path, scored)

    summary: dict[str, Any] = {}
    for tag in SCORE_TAGS:
        values = [
            float(record["judge_scores"][tag])
            for record in scored
            if record.get("judge_scores") and float(record["judge_scores"][tag]) >= 0
        ]
        summary[tag] = (
            {"mean": round(mean(values), 6), "count": len(values)}
            if values
            else {"mean": None, "count": 0}
        )
    summary["scored"] = sum(
        record.get("judge_status") == "success" for record in scored
    )
    summary["failed"] = sum(
        record.get("judge_status") != "success" for record in scored
    )
    write_json(output_path.parent / "judge-summary.json", summary)
    return summary
