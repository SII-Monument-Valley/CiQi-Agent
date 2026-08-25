#!/usr/bin/env python3
"""Minimal OpenAI-compatible conversation check for a deployed model."""

from __future__ import annotations

import json
import os
import urllib.request


base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:18901/v1").rstrip("/")
model = os.getenv("SERVED_MODEL_NAME", "ciqi-agent")
api_key = os.getenv("MODEL_API_KEY", "")
payload = json.dumps(
    {
        "model": model,
        "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
        "temperature": 0,
        "max_tokens": 128,
    },
    ensure_ascii=False,
).encode("utf-8")
headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
request = urllib.request.Request(
    f"{base_url}/chat/completions", data=payload, headers=headers, method="POST"
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.loads(response.read().decode("utf-8"))
message = result["choices"][0]["message"]["content"]
print(message)
