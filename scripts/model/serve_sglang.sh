#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVER_PROJECT_DIR="${PROJECT_ROOT}/src/model_server"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
[[ -n "${MODEL_PATH}" ]] || {
  echo "Set MODEL_PATH to a local model directory or Hugging Face model ID." >&2
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

if [[ "${MODEL_SERVER_SKIP_SYNC:-0}" != "1" ]]; then
  uv sync --project "${SERVER_PROJECT_DIR}"
fi

SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --host "${HOST:-127.0.0.1}"
  --port "${PORT:-18901}"
  --served-model-name "${SERVED_MODEL_NAME:-ciqi-agent}"
  --tp "${TP_SIZE:-1}"
  --dp "${DP_SIZE:-1}"
  --dtype "${DTYPE:-bfloat16}"
  --context-length "${CONTEXT_LENGTH:-32768}"
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen25}"
  --trust-remote-code
  --enable-multimodal
)
if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  SERVER_ARGS+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi
if [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]]; then
  SERVER_ARGS+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
fi
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  SERVER_ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
  SERVER_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  SERVER_ARGS+=(--disable-cuda-graph)
fi
if [[ -n "${API_KEY:-}" ]]; then
  SERVER_ARGS+=(--api-key "${API_KEY}")
fi

exec uv run --project "${SERVER_PROJECT_DIR}" \
  python -m sglang.launch_server "${SERVER_ARGS[@]}"
