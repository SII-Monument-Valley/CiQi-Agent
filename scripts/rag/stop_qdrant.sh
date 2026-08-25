#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

usage(){ echo "用法: $0 /path/to/qdrant_config.yaml"; exit 1; }
[[ $# -eq 1 ]] || usage

CONF="$1"
[[ -f "$CONF" ]] || { echo "[x] 配置文件不存在: $CONF"; exit 1; }

# 解析成绝对路径（兼容无 realpath 的系统）
if command -v realpath >/dev/null 2>&1; then
  CONF_ABS="$(realpath "$CONF")"
else
  CONF_ABS="$(cd "$(dirname "$CONF")" && pwd)/$(basename "$CONF")"
fi

# 可选：解析端口用于探活（没有 yq 就用默认 6333）
if command -v yq >/dev/null 2>&1; then
  PORT="$(yq -r '.service.http_port // 6333' "$CONF_ABS" 2>/dev/null || echo 6333)"
else
  PORT="${QDRANT_HTTP_PORT:-6333}"
fi

PID_DIR="${PID_DIR:-${PROJECT_ROOT}/outputs/rag/run}"
PID_FILE="${PID_DIR}/qdrant-$(basename "$CONF_ABS").pid"

http_alive() { curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; }

# 从 /proc/$pid/cmdline 中取 --config-path 并做真实路径比较
pid_matches_conf() {
  local pid="$1" cmdline cfg_arg cfg_real
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < /proc/"$pid"/cmdline)"
  [[ "$cmdline" == *"/qdrant"* && "$cmdline" == *"--config-path"* ]] || return 1
  cfg_arg="$(sed -n 's/.*--config-path[= ]\([^ ]*\).*/\1/p' <<< "$cmdline" | tail -n1)"
  cfg_real="$(readlink -f "$cfg_arg" 2>/dev/null || echo "$cfg_arg")"
  [[ "$cfg_real" == "$CONF_ABS" ]]
}

# 找到所有匹配该配置的 qdrant PIDs
collect_pids() {
  local pids=""
  # 1) PID 文件优先
  if [[ -f "$PID_FILE" ]]; then
    local fpid; fpid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${fpid:-}" && -d "/proc/$fpid" ]] && pid_matches_conf "$fpid"; then
      pids="$fpid"
    else
      # 失效 PID 文件清理
      rm -f "$PID_FILE"
    fi
  fi
  # 2) 枚举系统中所有 qdrant 进程
  local all; all="$(pgrep -x qdrant || true)"
  for pid in $all; do
    pid_matches_conf "$pid" || continue
    if [[ " $pids " != *" $pid "* ]]; then
      pids="$pids $pid"
    fi
  done
  echo "$pids" | xargs -r echo
}

PIDS="$(collect_pids)"

if [[ -z "$PIDS" ]]; then
  echo "[i] 未发现 Qdrant 进程（配置：$CONF_ABS）"
  exit 0
fi

echo "[*] 即将停止进程: $PIDS （配置：$CONF_ABS）"

# 优雅停止：SIGTERM
kill -TERM $PIDS 2>/dev/null || true

# 等待最多 10 次（5 秒）
for _ in {1..10}; do
  alive=()
  for pid in $PIDS; do
    if [[ -d "/proc/$pid" ]]; then alive+=("$pid"); fi
  done
  if [[ ${#alive[@]} -eq 0 ]]; then
    break
  fi
  sleep 0.5
done

# 仍存活则强制杀
still=()
for pid in $PIDS; do
  [[ -d "/proc/$pid" ]] && still+=("$pid")
done
if [[ ${#still[@]} -gt 0 ]]; then
  echo "[!] 发送 SIGKILL 给仍存活的进程: ${still[*]}"
  kill -KILL "${still[@]}" 2>/dev/null || true
fi

# 清理 PID 文件
if [[ -f "$PID_FILE" ]]; then
  rm -f "$PID_FILE"
fi

# 端口探活（可选）：确认已停止
if http_alive; then
  echo "[!] 端口 ${PORT} 仍有响应，请确认是否有其他实例/反向代理。"
else
  echo "[✓] 已停止 Qdrant（配置：$CONF_ABS）"
fi
