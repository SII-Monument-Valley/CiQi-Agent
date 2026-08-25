from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import load_evaluation_config
from .dataset import validate_ciqi_vqa
from .judge import score_file
from .runner import EvaluationRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ciqi-eval", description="Evaluate multimodal tool-using agents"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run model inference and task metrics")
    run.add_argument("--config", required=True, help="YAML configuration path")
    run.add_argument(
        "--task", choices=("selection", "qa"), help="Override experiment.task"
    )
    run.add_argument(
        "--limit", type=int, help="Evaluate only the first N rows per split"
    )
    run.add_argument(
        "--category",
        help="Evaluate only one dataset type/category; omit to evaluate all categories",
    )
    run.add_argument("--output", help="Override output.directory")

    validate = subparsers.add_parser(
        "validate", help="Validate YAML and local dataset paths without calling APIs"
    )
    validate.add_argument("--config", required=True, help="YAML configuration path")
    validate.add_argument(
        "--task", choices=("selection", "qa"), help="Override experiment.task"
    )

    score = subparsers.add_parser(
        "score", help="Apply the optional LLM judge to an existing result file"
    )
    score.add_argument(
        "--config", required=True, help="YAML configuration containing judge settings"
    )
    score.add_argument("--input", required=True, help="Result JSON/JSONL file")
    score.add_argument("--output", required=True, help="Scored JSON/JSONL file")

    dataset = subparsers.add_parser(
        "dataset", help="Inspect or validate a downloaded evaluation dataset"
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_validate = dataset_commands.add_parser(
        "validate", help="Validate the CiQi-VQA test.jsonl layout and image paths"
    )
    dataset_validate.add_argument("--root", required=True, help="CiQi-VQA root")
    dataset_validate.add_argument(
        "--split", default="test", choices=("test",), help="Only test is evaluated"
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    config = load_evaluation_config(
        args.config,
        task_override=args.task,
        output_override=args.output,
    )
    try:
        summary = await EvaluationRunner(config).run(
            limit=args.limit, category=args.category
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {"output": str(summary.output_directory), **summary.metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate(args: argparse.Namespace) -> None:
    config = load_evaluation_config(args.config, task_override=args.task)
    missing = [
        str(dataset.path)
        for dataset in config.datasets.values()
        if not dataset.path.is_file()
    ]
    if missing:
        raise SystemExit("Missing dataset files:\n- " + "\n- ".join(missing))
    print(json.dumps(config.public_dict(), ensure_ascii=False, indent=2))


async def _score(args: argparse.Namespace) -> None:
    config = load_evaluation_config(args.config)
    if not config.judge:
        raise SystemExit("The configuration has no judge section")
    summary = await score_file(
        Path(args.input).resolve(), Path(args.output).resolve(), config.judge
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_dataset(args: argparse.Namespace) -> None:
    report = validate_ciqi_vqa(Path(args.root), split=args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "validate":
        _validate(args)
    elif args.command == "score":
        asyncio.run(_score(args))
    elif args.command == "dataset" and args.dataset_command == "validate":
        _validate_dataset(args)


if __name__ == "__main__":
    main()
