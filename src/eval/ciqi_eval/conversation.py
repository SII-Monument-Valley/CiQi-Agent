from __future__ import annotations

import base64
import copy
import json
import math
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ImageConfig


def _round_factor(value: float, factor: int, mode: str = "round") -> int:
    operation = {"ceil": math.ceil, "floor": math.floor, "round": round}[mode]
    return max(factor, int(operation(value / factor) * factor))


def smart_resize(width: int, height: int, config: ImageConfig) -> tuple[int, int]:
    if (
        min(width, height) <= 0
        or max(width, height) / min(width, height) > config.max_ratio
    ):
        raise ValueError("Invalid image dimensions or aspect ratio")
    resized_width = _round_factor(width, config.factor)
    resized_height = _round_factor(height, config.factor)
    area = resized_width * resized_height
    if area > config.max_pixels:
        scale = math.sqrt((width * height) / config.max_pixels)
        resized_width = _round_factor(width / scale, config.factor, "floor")
        resized_height = _round_factor(height / scale, config.factor, "floor")
    elif area < config.min_pixels:
        scale = math.sqrt(config.min_pixels / (width * height))
        resized_width = _round_factor(width * scale, config.factor, "ceil")
        resized_height = _round_factor(height * scale, config.factor, "ceil")
    return resized_width, resized_height


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


class ToolConversation:
    """Backend-compatible multimodal conversation state used by the evaluator."""

    def __init__(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[Path],
        images: ImageConfig,
    ) -> None:
        self.images_raw: list[Image.Image] = []
        self.ratios: list[tuple[float, float]] = []
        model_images: list[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as source:
                raw = source.convert("RGB")
            model_image = raw
            if images.resize:
                width, height = smart_resize(raw.width, raw.height, images)
                model_image = raw.resize((width, height), Image.Resampling.LANCZOS)
            self.images_raw.append(raw)
            model_images.append(model_image)
            self.ratios.append(
                (raw.width / model_image.width, raw.height / model_image.height)
            )

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        if model_images:
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
                for image in model_images
            ]
            content.append({"type": "text", "text": user_prompt})
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "user", "content": user_prompt})

    def append_assistant(
        self,
        content: str,
        *,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_name:
            message["_native_text"] = content
            message["_tool_call"] = {
                "id": tool_call_id,
                "name": tool_name,
                "arguments": arguments or {},
            }
            payload = json.dumps(
                {"name": tool_name, "arguments": arguments or {}},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            content = f"{content}\n<tool_call>\n{payload}\n</tool_call>".strip()
            message["content"] = content
        self.messages.append(message)

    def append_tool_result(
        self,
        text: str,
        images: Iterable[Image.Image] = (),
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        image_list = list(images)
        if not image_list:
            self.messages.append(
                {
                    "role": "user",
                    "content": text,
                    "_tool_result": {
                        "id": tool_call_id,
                        "name": tool_name,
                        "text": text,
                    },
                }
            )
            return
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
            for image in image_list
        ]
        content.append({"type": "text", "text": text})
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "_tool_result": {
                    "id": tool_call_id,
                    "name": tool_name,
                    "text": text,
                },
            }
        )

    def export(self) -> tuple[dict[str, Any], ...]:
        exported = copy.deepcopy(self.messages)
        for message in exported:
            for key in tuple(message):
                if key.startswith("_"):
                    message.pop(key)
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "image_url", "image_url": {"url": "[image omitted]"}}
                    if item.get("type") == "image_url"
                    else item
                    for item in message["content"]
                ]
        return tuple(exported)
