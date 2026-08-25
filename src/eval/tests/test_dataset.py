from __future__ import annotations

import json
from pathlib import Path

from ciqi_eval.dataset import validate_ciqi_vqa
from PIL import Image


def test_validate_ciqi_vqa_test_layout(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "part-000"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "one.jpg")
    row = {
        "id": "test-one",
        "question": "question",
        "images": ["images/part-000/one.jpg"],
        "ground_truth": "name",
        "options": ["A", "B"],
        "answer": "B",
        "type": "overall",
    }
    (tmp_path / "test.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = validate_ciqi_vqa(tmp_path)
    assert report["valid"] is True
    assert report["samples"] == 1
    assert report["image_references"] == 1


def test_validate_ciqi_vqa_reports_missing_image(tmp_path: Path) -> None:
    row = {
        "id": "test-one",
        "question": "question",
        "images": ["images/missing.jpg"],
        "options": ["A"],
        "answer": "A",
    }
    (tmp_path / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = validate_ciqi_vqa(tmp_path)
    assert report["valid"] is False
    assert report["error_count"] == 1
    assert "Image does not exist" in report["errors"][0]
