#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${EVAL_DIR}/../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "${EVAL_DIR}"
uv sync --extra dev --extra hf

if [[ "${1:-}" == "--check" ]]; then
  cd "${REPO_DIR}"
  uv run --project "${EVAL_DIR}" ruff check src/eval/ciqi_eval src/eval/tests
  uv run --project "${EVAL_DIR}" pytest -q src/eval/tests
  uv run --project "${EVAL_DIR}" ciqi-eval validate \
    --config src/eval/examples/config.yaml
fi

echo "Evaluation environment is ready: ${EVAL_DIR}/.venv"
