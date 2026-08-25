from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .types import EvaluationResult


class DatasetError(ValueError):
    """Raised when an evaluation dataset does not follow the documented schema."""


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"Dataset does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"Each JSONL row must be an object: {path}:{line_number}"
                )
            records.append(record)
        return records

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(
        isinstance(item, dict) for item in loaded
    ):
        raise DatasetError(f"JSON dataset must contain a list of objects: {path}")
    return loaded


def resolve_image_paths(
    record: dict[str, Any], dataset: DatasetConfig
) -> tuple[Path, ...]:
    raw_images = record.get("images", record.get("image", []))
    if raw_images in (None, ""):
        return ()
    values = raw_images if isinstance(raw_images, list) else [raw_images]
    resolved = []
    for raw in values:
        text = str(raw)
        if text.startswith(("http://", "https://", "data:image")):
            raise DatasetError(
                f"Remote/data-URL dataset images are not accepted by the reproducible loader: {text[:80]}"
            )
        path = Path(text[7:]) if text.startswith("file://") else Path(text)
        path = path.expanduser()
        path = (
            path.resolve()
            if path.is_absolute()
            else (dataset.image_root / path).resolve()
        )
        if not path.is_file():
            raise DatasetError(f"Image does not exist: {path}")
        resolved.append(path)
    return tuple(resolved)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_results(path: Path, results: Iterable[EvaluationResult]) -> None:
    content = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in results
    )
    _atomic_text(path, content)
