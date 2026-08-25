from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from math import ceil, floor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from .config import ToolsConfig
from .types import ToolContext, ToolOutput

ZOOM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": "裁剪并放大指定图片的局部区域，以观察纹饰、款识、釉面等细节。一次仅处理一张图。",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "图片序号，从 1 开始。"},
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "模型所见图片坐标中的 [x1, y1, x2, y2]。",
                },
                "label": {"type": "string", "description": "区域名称，可选。"},
            },
            "required": ["index", "bbox_2d"],
            "additionalProperties": False,
        },
    },
}

SEARCH_TEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_text",
        "description": "从瓷器知识库检索与文本查询相关的器物资料和参考图像。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "简洁、具体的检索词。"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

SEARCH_IMAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_image",
        "description": "以指定输入图片为查询，在瓷器知识库中查找视觉相似器物。",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "图片序号，从 1 开始。"}
            },
            "required": ["index"],
            "additionalProperties": False,
        },
    },
}


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolOutput]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[dict[str, Any], ToolHandler]] = {}

    def register(self, schema: dict[str, Any], handler: ToolHandler) -> None:
        name = str(schema.get("function", {}).get("name") or "")
        if not name:
            raise ValueError("Tool schema is missing function.name")
        if name in self._tools:
            raise ValueError(f"Tool is already registered: {name}")
        self._tools[name] = (schema, handler)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [schema for schema, _ in self._tools.values()]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def execute(
        self, name: str, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolOutput:
        if name not in self._tools:
            raise ValueError(f"Unsupported tool: {name}")
        return await self._tools[name][1](context, arguments)


def _map_bbox(bbox: Sequence[float], ratio: tuple[float, float]) -> list[int]:
    if len(bbox) != 4:
        raise ValueError("bbox_2d must contain four coordinates")
    x_ratio, y_ratio = ratio
    left, top, right, bottom = (float(value) for value in bbox)
    return [
        floor(left * x_ratio),
        floor(top * y_ratio),
        ceil(right * x_ratio),
        ceil(bottom * y_ratio),
    ]


def _fit_bbox(
    bbox: Sequence[float], width: int, height: int, min_dimension: int
) -> list[int]:
    left, top, right, bottom = (float(value) for value in bbox)
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(width), right), min(float(height), bottom)
    if left >= right or top >= bottom:
        raise ValueError("bbox_2d is outside the image or has zero area")
    box_width, box_height = right - left, bottom - top
    if max(box_width, box_height) / min(box_width, box_height) > 100:
        raise ValueError("bbox_2d aspect ratio is too large")
    target_width = min(float(width), max(box_width, float(min_dimension)))
    target_height = min(float(height), max(box_height, float(min_dimension)))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    left = min(max(0.0, center_x - target_width / 2), width - target_width)
    top = min(max(0.0, center_y - target_height / 2), height - target_height)
    return [
        floor(left),
        floor(top),
        ceil(left + target_width),
        ceil(top + target_height),
    ]


def zoom_image(
    images: Sequence[Image.Image],
    ratios: Sequence[tuple[float, float]],
    *,
    index: int,
    bbox_2d: Sequence[float],
    label: str = "",
    min_dimension: int = 28,
) -> ToolOutput:
    if index < 1 or index > len(images):
        raise ValueError(f"index must be between 1 and {len(images)}")
    image = images[index - 1]
    mapped = _map_bbox(bbox_2d, ratios[index - 1] if ratios else (1.0, 1.0))
    final = _fit_bbox(mapped, image.width, image.height, min_dimension)
    crop = image.crop(final)
    label_text = f"“{label}”" if label else "目标"
    return ToolOutput(
        images=(crop,),
        text=f"已放大第 {index} 张图中的{label_text}区域，可据此继续观察细节。",
        metadata={
            "success": True,
            "image_index": index,
            "requested_bbox": list(bbox_2d),
            "resolved_bbox": final,
            "crop_size": [crop.width, crop.height],
        },
    )


class RagSearchClient:
    def __init__(self, config: ToolsConfig) -> None:
        self.config = config.rag
        self.image_roots = tuple(root.resolve() for root in self.config.image_roots)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _request(
        self, *, text: str | None = None, image: Image.Image | None = None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                if image is not None:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, format="JPEG", quality=92)
                    response = requests.post(
                        f"{self.config.base_url}/search/image",
                        headers=self._headers(),
                        files={"files": ("query.jpg", buffer.getvalue(), "image/jpeg")},
                        data={"topk": self.config.top_k},
                        timeout=self.config.timeout,
                    )
                else:
                    response = requests.post(
                        f"{self.config.base_url}/search/text",
                        headers=self._headers(),
                        json={
                            "texts": [text or ""],
                            "topk": self.config.top_k,
                            "alpha": 0.8,
                        },
                        timeout=self.config.timeout,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("RAG response must be a JSON object")
                return payload
            except (requests.RequestException, TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < self.config.retries:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"RAG request failed: {last_error}")

    def _safe_local_path(self, payload: dict[str, Any]) -> Path | None:
        if not self.image_roots:
            return None
        raw_path = str(payload.get("image_path") or "")
        image_file = str(payload.get("image_file") or "")
        candidates: list[Path] = []
        if raw_path:
            path = Path(raw_path)
            candidates.append(
                path / image_file if path.is_dir() and image_file else path
            )
        for root in self.image_roots:
            if image_file:
                candidates.append(root / image_file)
                candidates.extend(root.rglob(image_file))
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not any(resolved.is_relative_to(root) for root in self.image_roots):
                    continue
                if resolved.is_file():
                    return resolved
            except (OSError, RuntimeError):
                continue
        return None

    def _load_image(self, payload: dict[str, Any]) -> Image.Image | None:
        encoded = payload.get("image_base64") or payload.get("base64")
        if isinstance(encoded, str) and encoded:
            try:
                return Image.open(
                    io.BytesIO(base64.b64decode(encoded.split(",", 1)[-1]))
                ).convert("RGB")
            except (binascii.Error, OSError, UnidentifiedImageError, ValueError):
                return None
        url = payload.get("image_url")
        if isinstance(url, str) and urlparse(url).scheme in {"http", "https"}:
            try:
                response = requests.get(
                    url, timeout=min(self.config.timeout, 30), stream=True
                )
                response.raise_for_status()
                content = response.raw.read(12 * 1024 * 1024 + 1)
                if len(content) > 12 * 1024 * 1024:
                    return None
                return Image.open(io.BytesIO(content)).convert("RGB")
            except (
                requests.RequestException,
                OSError,
                UnidentifiedImageError,
                ValueError,
            ):
                return None
        local_path = self._safe_local_path(payload)
        if local_path:
            try:
                with Image.open(local_path) as source:
                    return source.convert("RGB")
            except OSError:
                return None
        return None

    @staticmethod
    def _report(payload: dict[str, Any], score: Any) -> str:
        if payload.get("type") == "long_text":
            values = [
                ("文本", payload.get("text")),
                ("标题", payload.get("caption")),
                ("来源", payload.get("source")),
            ]
        else:
            values = [
                ("名称", payload.get("name")),
                ("朝代", payload.get("dynasty")),
                ("年号", payload.get("reign")),
                ("釉色", payload.get("color")),
                ("纹饰", payload.get("decoration")),
                ("器型", payload.get("object_type")),
                ("描述", payload.get("description")),
                ("来源", payload.get("source")),
            ]
        lines = [
            f"- {label}：{value}" for label, value in values if value not in (None, "")
        ]
        if score not in (None, ""):
            lines.append(f"- 匹配度：{score}")
        return "\n".join(lines)

    def _normalise(self, response: dict[str, Any], query: str) -> ToolOutput:
        reports: list[str] = []
        images: list[Image.Image] = []
        hits: list[dict[str, Any]] = []
        for result in response.get("results", []) or []:
            for hit in result.get("hits", []) or []:
                payload = hit.get("payload", {}) or {}
                report = self._report(payload, hit.get("score"))
                if report:
                    reports.append(report)
                image = self._load_image(payload)
                if image is not None:
                    images.append(image)
                hits.append(
                    {
                        "id": str(hit.get("id", "")),
                        "score": hit.get("score"),
                        "type": payload.get("type", "text"),
                        "name": payload.get("name") or payload.get("caption") or "",
                        "source": payload.get("source") or "",
                    }
                )
        message = (
            "成功搜索到以下内容：\n" + "\n---\n".join(reports)
            if reports
            else "没有搜索到相关结果。"
        )
        return ToolOutput(
            images=tuple(images),
            text=json.dumps({"搜索结果": message}, ensure_ascii=False),
            metadata={
                "success": True,
                "status": "success" if reports else "no_results",
                "query": query,
                "total_results": len(hits),
                "hits": hits,
            },
        )

    def search_text(self, query: str) -> ToolOutput:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return self._normalise(self._request(text=query), query)

    def search_image(self, images: Sequence[Image.Image], index: int) -> ToolOutput:
        if index < 1 or index > len(images):
            raise ValueError(f"index must be between 1 and {len(images)}")
        return self._normalise(self._request(image=images[index - 1]), f"image:{index}")


def build_tool_registry(
    config: ToolsConfig, *, min_dimension: int = 28
) -> ToolRegistry:
    registry = ToolRegistry()
    if not config.enabled:
        return registry
    rag = (
        RagSearchClient(config)
        if any(name.startswith("search_") for name in config.names)
        else None
    )

    async def handle_zoom(
        context: ToolContext, arguments: dict[str, Any]
    ) -> ToolOutput:
        return zoom_image(
            context.images,
            context.ratios,
            index=int(arguments["index"]),
            bbox_2d=arguments["bbox_2d"],
            label=str(arguments.get("label", "")),
            min_dimension=min_dimension,
        )

    async def handle_search_text(
        context: ToolContext, arguments: dict[str, Any]
    ) -> ToolOutput:
        assert rag is not None
        return await asyncio.to_thread(rag.search_text, str(arguments["query"]))

    async def handle_search_image(
        context: ToolContext, arguments: dict[str, Any]
    ) -> ToolOutput:
        assert rag is not None
        return await asyncio.to_thread(
            rag.search_image, context.images, int(arguments["index"])
        )

    definitions = {
        "image_zoom_in_tool": (ZOOM_SCHEMA, handle_zoom),
        "search_text": (SEARCH_TEXT_SCHEMA, handle_search_text),
        "search_image": (SEARCH_IMAGE_SCHEMA, handle_search_image),
    }
    for name in config.names:
        schema, handler = definitions[name]
        registry.register(schema, handler)
    return registry
