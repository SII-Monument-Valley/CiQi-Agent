#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

LOCALHOST="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
export LOCALHOST
export QDRANT_URL="http://${LOCALHOST:-127.0.0.1}:6333"

"${SCRIPT_DIR}/run_qdrant.sh" "${PROJECT_ROOT}/configs/rag/qdrant_distributed.yaml"

for _ in {1..30}; do
  curl -fsS "${QDRANT_URL}/" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "${QDRANT_URL}/" >/dev/null 2>&1 || {
  echo "[x] Qdrant 未能在 30 秒内就绪: ${QDRANT_URL}" >&2
  exit 1
}

RAG_ENV_FILE="${RAG_ENV_FILE:-${PROJECT_ROOT}/configs/rag/.env}"
[[ -f "$RAG_ENV_FILE" ]] || cp "${PROJECT_ROOT}/configs/rag/.env.example" "$RAG_ENV_FILE"
sed -i 's/\r$//' "$RAG_ENV_FILE" 2>/dev/null || true
set -a
# shellcheck disable=SC1091
. "$RAG_ENV_FILE"
set +a

RAG_PROJECT_DIR="${PROJECT_ROOT}/src/rag"
if [[ -n "${UV_BIN:-}" ]]; then
  :
elif command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  UV_BIN="${RAG_PROJECT_DIR}/bin/uv"
fi
[[ -x "$UV_BIN" ]] || { echo "[x] 未找到可执行的 uv: $UV_BIN" >&2; exit 1; }

if [[ "${RAG_SKIP_SYNC:-0}" != "1" ]]; then
  "$UV_BIN" sync --project "$RAG_PROJECT_DIR"
fi

"$UV_BIN" run --project "$RAG_PROJECT_DIR" python -m src.rag.api.ingest
exec "$UV_BIN" run --project "$RAG_PROJECT_DIR" uvicorn src.rag.api.main:app \
  --host 0.0.0.0 \
  --port "${UVICORN_PORT:-8000}" \
  --workers "${UVICORN_WORKERS:-4}"
