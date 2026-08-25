from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from ciqi_eval.config import load_evaluation_config
from ciqi_eval.runner import EvaluationRunner


async def _run(config_path: Path, limit: int, category: str | None) -> dict[str, Any]:
    summary = await EvaluationRunner(load_evaluation_config(config_path)).run(
        limit=limit, category=category
    )
    payload = {"output": str(summary.output_directory), **summary.metrics}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test project backend and GPT")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="CiQi-VQA root; otherwise use CIQI_VQA_ROOT",
    )
    return parser


async def main() -> int:
    eval_dir = Path(__file__).resolve().parents[1]
    args = _parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.dataset_root is not None:
        os.environ["CIQI_VQA_ROOT"] = str(args.dataset_root.resolve())
    os.environ.setdefault("CIQI_EVAL_BACKEND_MODEL", "ciqi-agent")
    os.environ.setdefault("CIQI_EVAL_GPT_MODEL", "gpt-5")
    os.environ.setdefault("CIQI_AGENT_RAG_API_KEY", "")

    required = (
        "CIQI_VQA_ROOT",
        "CIQI_EVAL_BACKEND_BASE_URL",
        "CIQI_EVAL_BACKEND_API_KEY",
        "CIQI_EVAL_GPT_BASE_URL",
        "CIQI_EVAL_GPT_API_KEY",
        "CIQI_AGENT_RAG_API_URL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing required configuration: " + ", ".join(missing))

    backend_result = await _run(
        eval_dir / "examples/ciqi-vqa/backend.yaml", args.limit, args.category
    )
    gpt_result = await _run(
        eval_dir / "examples/ciqi-vqa/gpt.yaml", args.limit, args.category
    )
    return int(bool(backend_result["errors"] or gpt_result["errors"]))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
