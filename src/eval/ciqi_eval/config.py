from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?}")
SUPPORTED_TASKS = {"selection", "qa"}
SUPPORTED_TOOLS = {"image_zoom_in_tool", "search_text", "search_image"}
SUPPORTED_PROVIDERS = {"openai", "anthropic", "gemini"}
PROVIDER_ALIASES = {"claude": "anthropic", "google": "gemini"}


class ConfigError(ValueError):
    """Raised when an evaluation configuration is invalid."""


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = os.getenv(name)
        if resolved is not None:
            return resolved
        if default is not None:
            return default
        raise ConfigError(
            f"Environment variable {name!r} is required by the configuration"
        )

    return ENV_PATTERN.sub(replace, value)


def _path(value: str | os.PathLike[str], base_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (base_dir / candidate).resolve()
    )


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    image_root: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str | None
    api_key: str
    provider: str = "openai"
    api_version: str | None = None
    timeout: float = 300.0
    retries: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class ImageConfig:
    resize: bool = True
    factor: int = 28
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 400 * 28 * 28
    max_ratio: int = 200


@dataclass(frozen=True)
class RagConfig:
    base_url: str = ""
    api_key: str = ""
    image_roots: tuple[Path, ...] = ()
    top_k: int = 3
    timeout: float = 120.0
    retries: int = 3


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = False
    names: tuple[str, ...] = ()
    rag: RagConfig = field(default_factory=RagConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int = 1
    max_tool_steps: int = 8
    max_empty_responses: int = 2
    continue_on_error: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    save_conversations: bool = True
    copy_config: bool = True


@dataclass(frozen=True)
class PromptConfig:
    system: str
    user_prefix: str = ""
    final_instruction: str = (
        "请在分析后选择调用工具或者回答，最终答案请置于 <answer></answer> 标签中。"
    )


@dataclass(frozen=True)
class EvaluationConfig:
    version: int
    name: str
    task: str
    datasets: dict[str, DatasetConfig]
    model: ModelConfig
    generation: GenerationConfig
    images: ImageConfig
    tools: ToolsConfig
    runtime: RuntimeConfig
    output: OutputConfig
    prompt: PromptConfig
    judge: dict[str, Any]
    source_path: Path | None = None

    def public_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe configuration with credentials redacted."""

        def clean(value: Any, key: str = "") -> Any:
            sensitive_keys = {
                "api_key",
                "secret",
                "password",
                "access_token",
                "auth_token",
            }
            if key.lower() in sensitive_keys:
                return "<redacted>" if value else ""
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {name: clean(item, str(name)) for name, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item, key) for item in value]
            return value

        payload = asdict(self)
        payload.pop("source_path", None)
        return clean(payload)


def _dataset_configs(
    raw: Any, base_dir: Path, default_image_root: str | None = None
) -> dict[str, DatasetConfig]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError("datasets must be a non-empty mapping")
    datasets: dict[str, DatasetConfig] = {}
    for split, item in raw.items():
        if isinstance(item, (str, os.PathLike)):
            path = _path(item, base_dir)
            image_root = _path(default_image_root or path.parent, base_dir)
        elif isinstance(item, Mapping):
            if not item.get("path"):
                raise ConfigError(f"datasets.{split}.path is required")
            path = _path(str(item["path"]), base_dir)
            image_root = _path(
                str(item.get("image_root") or default_image_root or path.parent),
                base_dir,
            )
        else:
            raise ConfigError(f"datasets.{split} must be a path or mapping")
        datasets[str(split)] = DatasetConfig(path=path, image_root=image_root)
    return datasets


def _tool_names(raw: Any) -> tuple[str, ...]:
    names: list[str] = []
    for item in raw or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Mapping):
            function = item.get("function") or {}
            name = function.get("name") if isinstance(function, Mapping) else None
            if name:
                names.append(str(name))
    unknown = sorted(set(names) - SUPPORTED_TOOLS)
    if unknown:
        raise ConfigError(f"Unsupported tools: {', '.join(unknown)}")
    return tuple(dict.fromkeys(names))


def _normalise_mapping(
    raw: Mapping[str, Any],
    *,
    task_override: str | None,
    output_override: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Translate the historical flat Gugong YAML schema to schema version 1."""
    if "datasets" in raw and "model" in raw:
        nested = copy.deepcopy(dict(raw))
        if task_override:
            nested.setdefault("experiment", {})["task"] = task_override
        if output_override:
            nested.setdefault("output", {})["directory"] = str(output_override)
        return nested

    api_config = dict(raw.get("api_config") or {})
    tool_names = _tool_names(raw.get("tool_info") or [])
    model_name = str(raw.get("infer_model_name") or raw.get("model_name") or "model")
    experiment_name = str(raw.get("model_name") or model_name)
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("CIQI_AGENT_API_KEY")
        or str(raw.get("api_key") or "EMPTY")
    )
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("CIQI_AGENT_API_URL")
        or raw.get("api_url")
    )
    output_dir = output_override or raw.get("save_path") or "outputs/eval"
    enabled = bool(raw.get("is_use_tool", raw.get("use_tool", False)))
    if enabled and not tool_names:
        tool_names = ("image_zoom_in_tool", "search_text", "search_image")
    return {
        "version": 1,
        "experiment": {
            "name": experiment_name,
            "task": task_override or raw.get("task") or "selection",
        },
        "datasets": raw.get("file_path") or {},
        "dataset_defaults": {"image_root": raw.get("img_path") or ""},
        "model": {
            "name": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "provider": "openai",
            "timeout": api_config.get("timeout", 300),
            "retries": raw.get("api_retries", 2),
        },
        "generation": {
            "temperature": api_config.get("temperature", 0),
            "max_tokens": api_config.get("max_tokens", 2048),
            "top_p": api_config.get("top_p"),
            "stop": api_config.get("stop") or [],
            "seed": api_config.get("seed"),
        },
        "images": {
            "resize": raw.get("is_resize_image", True),
            "factor": raw.get("image_factor", 28),
            "min_pixels": raw.get("min_pixels", 4 * 28 * 28),
            "max_pixels": raw.get("max_pixels", 400 * 28 * 28),
            "max_ratio": raw.get("max_ratio", 200),
        },
        "tools": {
            "enabled": enabled,
            "names": list(tool_names),
            "rag": {
                "base_url": os.getenv("CIQI_AGENT_RAG_API_URL", ""),
                "api_key": os.getenv("CIQI_AGENT_RAG_API_KEY", ""),
                "image_roots": [
                    part
                    for part in os.getenv("CIQI_AGENT_RAG_IMAGE_ROOTS", "").split(
                        os.pathsep
                    )
                    if part
                ],
            },
        },
        "runtime": {
            "concurrency": raw.get("num_workers", 1),
            "max_tool_steps": raw.get("max_turn_count", raw.get("max_try_count", 8)),
            "max_empty_responses": 2,
            "continue_on_error": True,
        },
        "output": {
            "directory": str(output_dir),
            "save_conversations": True,
            "copy_config": raw.get("copy_config", True),
        },
        "prompt": {
            "system": raw.get("system_prompt", "你是一个乐于助人的助手。"),
            "user_prefix": raw.get("user_prompt", ""),
            "final_instruction": raw.get(
                "final_instruction",
                "请在分析后选择调用工具或者回答，最终答案请置于 <answer></answer> 标签中。",
            ),
        },
        "judge": raw.get("judge_config") or raw.get("qa_score_config") or {},
    }


def evaluation_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    source_path: str | os.PathLike[str] | None = None,
    task_override: str | None = None,
    output_override: str | os.PathLike[str] | None = None,
) -> EvaluationConfig:
    source = Path(source_path).resolve() if source_path else None
    base_dir = source.parent if source else Path.cwd()
    data = _expand_env(
        _normalise_mapping(
            raw, task_override=task_override, output_override=output_override
        )
    )

    experiment = data.get("experiment") or {}
    task = str(experiment.get("task") or "selection").lower()
    if task not in SUPPORTED_TASKS:
        raise ConfigError(
            f"experiment.task must be one of {sorted(SUPPORTED_TASKS)}, got {task!r}"
        )
    name = str(experiment.get("name") or f"ciqi-{task}")

    defaults = data.get("dataset_defaults") or {}
    datasets = _dataset_configs(
        data.get("datasets"), base_dir, defaults.get("image_root")
    )
    model_raw = data.get("model") or {}
    generation_raw = data.get("generation") or {}
    images_raw = data.get("images") or {}
    tools_raw = data.get("tools") or {}
    rag_raw = tools_raw.get("rag") or {}
    runtime_raw = data.get("runtime") or {}
    output_raw = data.get("output") or {}
    prompt_raw = data.get("prompt") or {}

    names = _tool_names(tools_raw.get("names") or [])
    tools_enabled = bool(tools_raw.get("enabled", bool(names)))
    if not tools_enabled:
        names = ()
    if tools_enabled and not names:
        raise ConfigError("tools.enabled is true but tools.names is empty")
    if any(name.startswith("search_") for name in names) and not str(
        rag_raw.get("base_url") or ""
    ):
        raise ConfigError(
            "tools.rag.base_url is required when a search tool is enabled"
        )

    stop_raw = generation_raw.get("stop") or []
    stop = (
        (str(stop_raw),)
        if isinstance(stop_raw, str)
        else tuple(str(item) for item in stop_raw)
    )
    image_roots_raw = rag_raw.get("image_roots") or []
    if isinstance(image_roots_raw, (str, os.PathLike)):
        image_roots_raw = [image_roots_raw]

    output_dir = _path(str(output_raw.get("directory") or "outputs/eval"), base_dir)
    provider = str(model_raw.get("provider") or "openai").strip().lower()
    provider = PROVIDER_ALIASES.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            "model.provider must be one of "
            f"{sorted(SUPPORTED_PROVIDERS)}, got {provider!r}"
        )
    return EvaluationConfig(
        version=int(data.get("version", 1)),
        name=name,
        task=task,
        datasets=datasets,
        model=ModelConfig(
            name=str(model_raw.get("name") or "model"),
            base_url=str(model_raw["base_url"]).rstrip("/")
            if model_raw.get("base_url")
            else None,
            api_key=str(model_raw.get("api_key") or "EMPTY"),
            provider=provider,
            api_version=str(model_raw["api_version"])
            if model_raw.get("api_version")
            else None,
            timeout=float(model_raw.get("timeout", 300)),
            retries=_positive_int(
                model_raw.get("retries", 2), "model.retries", allow_zero=True
            ),
        ),
        generation=GenerationConfig(
            temperature=float(generation_raw.get("temperature", 0)),
            max_tokens=_positive_int(
                generation_raw.get("max_tokens", 2048), "generation.max_tokens"
            ),
            top_p=float(generation_raw["top_p"])
            if generation_raw.get("top_p") is not None
            else None,
            stop=stop,
            seed=int(generation_raw["seed"])
            if generation_raw.get("seed") is not None
            else None,
        ),
        images=ImageConfig(
            resize=bool(images_raw.get("resize", True)),
            factor=_positive_int(images_raw.get("factor", 28), "images.factor"),
            min_pixels=_positive_int(
                images_raw.get("min_pixels", 4 * 28 * 28), "images.min_pixels"
            ),
            max_pixels=_positive_int(
                images_raw.get("max_pixels", 400 * 28 * 28), "images.max_pixels"
            ),
            max_ratio=_positive_int(
                images_raw.get("max_ratio", 200), "images.max_ratio"
            ),
        ),
        tools=ToolsConfig(
            enabled=tools_enabled,
            names=names,
            rag=RagConfig(
                base_url=str(rag_raw.get("base_url") or "").rstrip("/"),
                api_key=str(rag_raw.get("api_key") or ""),
                image_roots=tuple(
                    _path(str(item), base_dir) for item in image_roots_raw
                ),
                top_k=_positive_int(rag_raw.get("top_k", 3), "tools.rag.top_k"),
                timeout=float(rag_raw.get("timeout", 120)),
                retries=_positive_int(rag_raw.get("retries", 3), "tools.rag.retries"),
            ),
        ),
        runtime=RuntimeConfig(
            concurrency=_positive_int(
                runtime_raw.get("concurrency", 1), "runtime.concurrency"
            ),
            max_tool_steps=_positive_int(
                runtime_raw.get("max_tool_steps", 8),
                "runtime.max_tool_steps",
                allow_zero=True,
            ),
            max_empty_responses=_positive_int(
                runtime_raw.get("max_empty_responses", 2),
                "runtime.max_empty_responses",
                allow_zero=True,
            ),
            continue_on_error=bool(runtime_raw.get("continue_on_error", True)),
        ),
        output=OutputConfig(
            directory=output_dir,
            save_conversations=bool(output_raw.get("save_conversations", True)),
            copy_config=bool(output_raw.get("copy_config", True)),
        ),
        prompt=PromptConfig(
            system=str(prompt_raw.get("system") or "你是一个乐于助人的助手。"),
            user_prefix=str(prompt_raw.get("user_prefix") or ""),
            final_instruction=str(
                prompt_raw.get("final_instruction")
                or "请在分析后选择调用工具或者回答，最终答案请置于 <answer></answer> 标签中。"
            ),
        ),
        judge=dict(data.get("judge") or {}),
        source_path=source,
    )


def load_raw_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration must be a mapping: {source}")
    return loaded


def load_evaluation_config(
    path: str | os.PathLike[str],
    *,
    task_override: str | None = None,
    output_override: str | os.PathLike[str] | None = None,
) -> EvaluationConfig:
    source = Path(path).expanduser().resolve()
    return evaluation_config_from_mapping(
        load_raw_config(source),
        source_path=source,
        task_override=task_override,
        output_override=output_override,
    )
