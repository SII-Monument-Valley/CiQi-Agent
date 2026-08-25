from __future__ import annotations

import asyncio
import base64
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

from .config import GenerationConfig, ModelConfig
from .types import ModelTurn, NativeToolCall


class ChatModel(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn: ...


class OpenAIChatModel:
    """OpenAI-compatible chat-completions adapter with no provider-specific globals."""

    def __init__(
        self,
        model: ModelConfig,
        generation: GenerationConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.generation = generation
        self.client = client or AsyncOpenAI(
            api_key=model.api_key,
            base_url=model.base_url,
            timeout=model.timeout,
            max_retries=model.retries,
        )

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.model.name,
            "messages": [
                {key: value for key, value in item.items() if not key.startswith("_")}
                for item in messages
            ],
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
        }
        if self.generation.top_p is not None:
            kwargs["top_p"] = self.generation.top_p
        if self.generation.stop:
            kwargs["stop"] = list(self.generation.stop)
        if self.generation.seed is not None:
            kwargs["seed"] = self.generation.seed
        if tools:
            kwargs["tools"] = tools

        completion = await self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        calls = []
        for item in getattr(message, "tool_calls", None) or []:
            calls.append(
                NativeToolCall(
                    name=item.function.name,
                    arguments=item.function.arguments or "{}",
                    call_id=getattr(item, "id", None),
                )
            )
        return ModelTurn(text=(message.content or "").strip(), tool_calls=tuple(calls))

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            await close()


def _data_url(value: str) -> tuple[str, str]:
    prefix, separator, encoded = value.partition(",")
    if not separator or not prefix.startswith("data:") or ";base64" not in prefix:
        raise ValueError("Provider adapters require base64 data URLs for images")
    media_type = prefix[5:].split(";", 1)[0] or "image/jpeg"
    base64.b64decode(encoded, validate=True)
    return media_type, encoded


def _openai_content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    return [item for item in content if isinstance(item, dict)]


def _anthropic_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    converted: list[dict[str, Any]] = []
    for part in _openai_content_parts(content):
        if part.get("type") == "text":
            converted.append({"type": "text", "text": str(part.get("text") or "")})
        elif part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            media_type, data = _data_url(str(image_url.get("url") or ""))
            converted.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
    return converted


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for part in _openai_content_parts(content):
        if part.get("type") == "text":
            converted.append({"text": str(part.get("text") or "")})
        elif part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            media_type, data = _data_url(str(image_url.get("url") or ""))
            converted.append({"inlineData": {"mimeType": media_type, "data": data}})
    return converted


class _NativeHttpModel:
    def __init__(
        self,
        model: ModelConfig,
        generation: GenerationConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.generation = generation
        self.client = client or httpx.AsyncClient(timeout=model.timeout)
        self._owns_client = client is None

    async def _post(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.model.retries + 1):
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise TypeError("Provider response must be a JSON object")
                return data
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.model.retries:
                    await asyncio.sleep(min(attempt + 1, 3))
        raise RuntimeError(f"Provider request failed: {last_error}") from last_error

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class AnthropicChatModel(_NativeHttpModel):
    """Native Anthropic Messages API adapter, including images and tools."""

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        system = "\n\n".join(
            str(item.get("content") or "")
            for item in messages
            if item.get("role") == "system"
        )
        request_messages: list[dict[str, Any]] = []
        for item in messages:
            if item.get("role") == "system":
                continue
            native_call = item.get("_tool_call")
            native_result = item.get("_tool_result")
            if isinstance(native_call, dict):
                content: list[dict[str, Any]] = []
                native_text = str(item.get("_native_text") or "").strip()
                if native_text:
                    content.append({"type": "text", "text": native_text})
                content.append(
                    {
                        "type": "tool_use",
                        "id": str(native_call.get("id") or ""),
                        "name": str(native_call.get("name") or ""),
                        "input": native_call.get("arguments") or {},
                    }
                )
                request_messages.append({"role": "assistant", "content": content})
            elif isinstance(native_result, dict):
                result_content = _anthropic_content(item.get("content", ""))
                request_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(native_result.get("id") or ""),
                                "content": result_content,
                            }
                        ],
                    }
                )
            else:
                request_messages.append(
                    {
                        "role": "assistant"
                        if item.get("role") == "assistant"
                        else "user",
                        "content": _anthropic_content(item.get("content", "")),
                    }
                )
        payload: dict[str, Any] = {
            "model": self.model.name,
            "messages": request_messages,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
        }
        if system:
            payload["system"] = system
        if self.generation.top_p is not None:
            payload["top_p"] = self.generation.top_p
        if self.generation.stop:
            payload["stop_sequences"] = list(self.generation.stop)
        if tools:
            payload["tools"] = [
                {
                    "name": item["function"]["name"],
                    "description": item["function"].get("description", ""),
                    "input_schema": item["function"].get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
                for item in tools
            ]

        base_url = self.model.base_url or "https://api.anthropic.com"
        url = (
            f"{base_url}/messages"
            if base_url.rstrip("/").endswith("/v1")
            else f"{base_url}/v1/messages"
        )
        data = await self._post(
            url,
            headers={
                "x-api-key": self.model.api_key,
                "anthropic-version": self.model.api_version or "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
        )
        texts: list[str] = []
        calls: list[NativeToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    NativeToolCall(
                        name=str(block.get("name") or ""),
                        arguments=block.get("input") or {},
                        call_id=str(block.get("id") or "") or None,
                    )
                )
        return ModelTurn(text="\n".join(texts).strip(), tool_calls=tuple(calls))


class GeminiChatModel(_NativeHttpModel):
    """Native Google Gemini generateContent adapter, including images and tools."""

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        system = "\n\n".join(
            str(item.get("content") or "")
            for item in messages
            if item.get("role") == "system"
        )
        contents: list[dict[str, Any]] = []
        for item in messages:
            if item.get("role") == "system":
                continue
            native_call = item.get("_tool_call")
            native_result = item.get("_tool_result")
            if isinstance(native_call, dict):
                parts: list[dict[str, Any]] = []
                native_text = str(item.get("_native_text") or "").strip()
                if native_text:
                    parts.append({"text": native_text})
                parts.append(
                    {
                        "functionCall": {
                            "name": str(native_call.get("name") or ""),
                            "args": native_call.get("arguments") or {},
                        }
                    }
                )
                contents.append({"role": "model", "parts": parts})
            elif isinstance(native_result, dict):
                parts = [
                    {
                        "functionResponse": {
                            "name": str(native_result.get("name") or ""),
                            "response": {
                                "result": str(native_result.get("text") or "")
                            },
                        }
                    }
                ]
                parts.extend(
                    part
                    for part in _gemini_parts(item.get("content", ""))
                    if "inlineData" in part
                )
                contents.append({"role": "user", "parts": parts})
            else:
                contents.append(
                    {
                        "role": "model" if item.get("role") == "assistant" else "user",
                        "parts": _gemini_parts(item.get("content", "")),
                    }
                )
        generation: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "maxOutputTokens": self.generation.max_tokens,
        }
        if self.generation.top_p is not None:
            generation["topP"] = self.generation.top_p
        if self.generation.stop:
            generation["stopSequences"] = list(self.generation.stop)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": item["function"]["name"],
                            "description": item["function"].get("description", ""),
                            "parameters": item["function"].get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        }
                        for item in tools
                    ]
                }
            ]

        base_url = (
            self.model.base_url or "https://generativelanguage.googleapis.com/v1beta"
        )
        url = f"{base_url}/models/{self.model.name}:generateContent"
        data = await self._post(
            url,
            headers={
                "x-goog-api-key": self.model.api_key,
                "content-type": "application/json",
            },
            payload=payload,
        )
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        texts: list[str] = []
        calls: list[NativeToolCall] = []
        for part in parts:
            if "text" in part:
                texts.append(str(part.get("text") or ""))
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                calls.append(
                    NativeToolCall(
                        name=str(function_call.get("name") or ""),
                        arguments=function_call.get("args") or {},
                    )
                )
        return ModelTurn(text="\n".join(texts).strip(), tool_calls=tuple(calls))


def build_chat_model(model: ModelConfig, generation: GenerationConfig) -> ChatModel:
    if model.provider == "openai":
        return OpenAIChatModel(model, generation)
    if model.provider == "anthropic":
        return AnthropicChatModel(model, generation)
    if model.provider == "gemini":
        return GeminiChatModel(model, generation)
    raise ValueError(f"Unsupported model provider: {model.provider}")
