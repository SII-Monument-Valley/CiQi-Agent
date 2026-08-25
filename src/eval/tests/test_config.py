from __future__ import annotations

from pathlib import Path

import pytest
from ciqi_eval.config import (
    ConfigError,
    evaluation_config_from_mapping,
    load_evaluation_config,
)


def test_example_config_resolves_paths_relative_to_yaml() -> None:
    source = Path(__file__).parents[1] / "examples" / "config.yaml"
    config = load_evaluation_config(source)
    assert config.task == "selection"
    assert config.datasets["test"].path == source.parent / "data" / "sample.jsonl"
    assert config.output.directory == source.parent / "outputs"
    assert config.tools.names == ("image_zoom_in_tool", "search_text", "search_image")


def test_public_config_redacts_credentials_but_not_max_tokens(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        '{"question":"q","answer":"A","options":["x"]}\n', encoding="utf-8"
    )
    config = evaluation_config_from_mapping(
        {
            "version": 1,
            "experiment": {"name": "test", "task": "selection"},
            "datasets": {"test": {"path": str(dataset)}},
            "model": {"name": "ckpt", "api_key": "secret-value"},
            "generation": {"max_tokens": 1234},
            "tools": {"enabled": False},
            "output": {"directory": str(tmp_path / "out")},
            "prompt": {"system": "system"},
        }
    )
    public = config.public_dict()
    assert public["model"]["api_key"] == "<redacted>"
    assert public["generation"]["max_tokens"] == 1234


def test_required_environment_variable_has_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MISSING_CIQI_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="MISSING_CIQI_TEST_KEY"):
        evaluation_config_from_mapping(
            {
                "version": 1,
                "experiment": {"task": "qa"},
                "datasets": {"test": str(tmp_path / "data.jsonl")},
                "model": {"name": "model", "api_key": "${MISSING_CIQI_TEST_KEY}"},
                "tools": {"enabled": False},
                "output": {"directory": "out"},
                "prompt": {"system": "system"},
            }
        )


def test_legacy_flat_config_is_supported(tmp_path: Path) -> None:
    dataset = tmp_path / "legacy.json"
    dataset.write_text("[]", encoding="utf-8")
    config = evaluation_config_from_mapping(
        {
            "model_name": "legacy-run",
            "infer_model_name": "ckpt",
            "api_url": "http://model.test/v1",
            "file_path": {"overall": str(dataset)},
            "img_path": str(tmp_path),
            "save_path": str(tmp_path / "out"),
            "is_use_tool": False,
            "num_workers": 3,
        },
        task_override="qa",
    )
    assert config.name == "legacy-run"
    assert config.task == "qa"
    assert config.runtime.concurrency == 3
    assert config.datasets["overall"].path == dataset


def test_claude_and_google_provider_aliases_are_supported(tmp_path: Path) -> None:
    base = {
        "version": 1,
        "experiment": {"task": "qa"},
        "datasets": {"test": str(tmp_path / "data.jsonl")},
        "tools": {"enabled": False},
        "output": {"directory": str(tmp_path / "out")},
        "prompt": {"system": "system"},
    }
    claude = evaluation_config_from_mapping(
        {**base, "model": {"provider": "claude", "name": "claude", "api_key": "x"}}
    )
    google = evaluation_config_from_mapping(
        {**base, "model": {"provider": "google", "name": "gemini", "api_key": "x"}}
    )
    assert claude.model.provider == "anthropic"
    assert google.model.provider == "gemini"
