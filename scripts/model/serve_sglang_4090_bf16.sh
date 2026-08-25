#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The published checkpoint is FP32 on disk. SGLang converts each tensor to
# BF16 while loading, reducing resident weight memory from about 33.2 GB to
# about 16.6 GB. These conservative defaults target one 24 GB RTX 4090.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DTYPE="${DTYPE:-bfloat16}"
export TP_SIZE="${TP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"

exec bash "${SCRIPT_DIR}/serve_sglang.sh" "$@"
