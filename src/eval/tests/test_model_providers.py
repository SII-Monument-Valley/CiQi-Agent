from __future__ import annotations

import base64
import json

import httpx
import pytest
from ciqi_eval.config import GenerationConfig, ModelConfig
from ciqi_eval.model import AnthropicChatModel, GeminiChatModel

DATA_URL = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode("ascii")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "system"},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": "question"},
        ],
    },
    {
        "role": "assistant",
        "content": '检索\n<tool_call>{"name":"search_text"}</tool_call>',
        "_native_text": "检索",
        "_tool_call": {
            "id": "tool-previous",
            "name": "search_text",
            "arguments": {"query": "康熙"},
        },
    },
    {
        "role": "user",
        "content": "检索证据",
        "_tool_result": {
            "id": "tool-previous",
            "name": "search_text",
            "text": "检索证据",
        },
    },
]


@pytest.mark.asyncio
async def test_anthropic_native_images_tools_and_response() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "检索"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "search_text",
                        "input": {"query": "康熙"},
                    },
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = AnthropicChatModel(
        ModelConfig(
            provider="anthropic",
            name="claude-test",
            base_url="https://api.anthropic.test",
            api_key="secret",
        ),
        GenerationConfig(),
        client=client,
    )
    turn = await model.complete(MESSAGES, TOOLS)
    await client.aclose()

    assert captured["url"] == "https://api.anthropic.test/v1/messages"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["system"] == "system"
    assert body["messages"][0]["content"][0]["type"] == "image"
    assert body["messages"][1]["content"][1]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["messages"][2]["content"][0]["tool_use_id"] == "tool-previous"
    assert body["tools"][0]["input_schema"]["required"] == ["query"]
    assert turn.text == "检索"
    assert turn.tool_calls[0].arguments == {"query": "康熙"}
    assert turn.tool_calls[0].call_id == "tool-1"


@pytest.mark.asyncio
async def test_gemini_native_images_tools_and_response() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "检索"},
                                {
                                    "functionCall": {
                                        "name": "search_text",
                                        "args": {"query": "雍正"},
                                    }
                                },
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = GeminiChatModel(
        ModelConfig(
            provider="gemini",
            name="gemini-test",
            base_url="https://generativelanguage.test/v1beta",
            api_key="secret",
        ),
        GenerationConfig(),
        client=client,
    )
    turn = await model.complete(MESSAGES, TOOLS)
    await client.aclose()

    assert captured["url"] == (
        "https://generativelanguage.test/v1beta/models/gemini-test:generateContent"
    )
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["x-goog-api-key"] == "secret"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["systemInstruction"]["parts"][0]["text"] == "system"
    assert "inlineData" in body["contents"][0]["parts"][0]
    assert "functionCall" in body["contents"][1]["parts"][1]
    assert "functionResponse" in body["contents"][2]["parts"][0]
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "search_text"
    assert turn.tool_calls[0].arguments == {"query": "雍正"}
