from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .io import DatasetError, read_records, resolve_image_paths


def validate_ciqi_vqa(root: Path, *, split: str = "test") -> dict[str, Any]:
    """Validate a downloaded CiQi-VQA split and return reproducible statistics."""
    root = root.expanduser().resolve()
    dataset_path = root / f"{split}.jsonl"
    records = read_records(dataset_path)
    config = DatasetConfig(path=dataset_path, image_root=root)
    errors: list[str] = []
    ids: set[str] = set()
    image_references = 0
    task_types: Counter[str] = Counter()
    answers: Counter[str] = Counter()

    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("id") or "")
        if not sample_id:
            errors.append(f"line {index}: missing id")
        elif sample_id in ids:
            errors.append(f"line {index}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        if not str(record.get("question") or "").strip():
            errors.append(f"line {index}: missing question")
        options = record.get("options")
        if not isinstance(options, (list, dict)) or not options:
            errors.append(f"line {index}: options must be a non-empty list or mapping")
        if "answer" not in record:
            errors.append(f"line {index}: missing answer")
        else:
            answers[str(record["answer"])] += 1
        task_types[str(record.get("type") or "unknown")] += 1
        try:
            image_references += len(resolve_image_paths(record, config))
        except DatasetError as exc:
            errors.append(f"line {index}: {exc}")

    return {
        "dataset": "SII-Monument-Valley/CiQi-VQA",
        "root": str(root),
        "split": split,
        "file": str(dataset_path),
        "samples": len(records),
        "unique_ids": len(ids),
        "image_references": image_references,
        "types": dict(sorted(task_types.items())),
        "answers": dict(sorted(answers.items())),
        "valid": not errors,
        "errors": errors[:20],
        "error_count": len(errors),
    }
