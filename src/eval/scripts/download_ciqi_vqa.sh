#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd -- "${EVAL_DIR}/../.." && pwd)"
REPO_ID="${1:-${CIQI_VQA_REPO_ID:-SII-Monument-Valley/CiQi-VQA}}"
DESTINATION="${2:-${CIQI_VQA_ROOT:-${PROJECT_DIR}/data/hf/ciqi-vqa}}"
REVISION="${3:-${CIQI_VQA_REVISION:-main}}"

cd "${EVAL_DIR}"
uv run --extra hf hf download "${REPO_ID}" test.jsonl \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DESTINATION}"

mapfile -t TEST_IMAGES < <(
  uv run python - "${DESTINATION}/test.jsonl" <<'PY'
import json
import sys
from pathlib import Path

images = set()
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    raw = json.loads(line).get("images", [])
    images.update(str(item) for item in (raw if isinstance(raw, list) else [raw]) if item)
print("\n".join(sorted(images)))
PY
)

if (( ${#TEST_IMAGES[@]} )); then
  uv run --extra hf hf download "${REPO_ID}" "${TEST_IMAGES[@]}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DESTINATION}"
fi

uv run ciqi-eval dataset validate --root "${DESTINATION}" --split test
