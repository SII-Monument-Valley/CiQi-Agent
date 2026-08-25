#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LIMIT="${1:-1}"
CATEGORY="${2:-}"

cd "${EVAL_DIR}"
if [[ -n "${CATEGORY}" ]]; then
  exec uv run python scripts/run_project_smoke.py --limit "${LIMIT}" --category "${CATEGORY}"
fi
exec uv run python scripts/run_project_smoke.py --limit "${LIMIT}"
