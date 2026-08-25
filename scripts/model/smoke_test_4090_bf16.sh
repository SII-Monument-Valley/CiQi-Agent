#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT="${PORT:-18901}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
SERVER_LOG="${SERVER_LOG:-/tmp/ciqi-agent-4090-bf16.log}"

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Mirror startup logs to the job console while retaining a local copy for the
# failure tail below.
bash "${SCRIPT_DIR}/serve_sglang_4090_bf16.sh" > >(tee "${SERVER_LOG}") 2>&1 &
server_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Model server exited before becoming ready." >&2
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Model server did not become ready within ${STARTUP_TIMEOUT}s." >&2
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done

MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ciqi-agent}" \
python3 "${PROJECT_ROOT}/scripts/model/test_chat.py"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader
fi
echo "ciqi-agent 4090 BF16 smoke test passed"
