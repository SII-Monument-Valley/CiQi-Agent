#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# ===== 参数与用法 =====
usage() {
  echo "用法: $0 /path/to/qdrant_config.yaml"
  exit 1
}
[[ $# -eq 1 ]] || usage

CONF="$1"
[[ -f "$CONF" ]] || { echo "[x] 配置文件不存在: $CONF"; exit 1; }

# 兼容无 realpath 的系统
if command -v realpath >/dev/null 2>&1; then
  CONF_ABS="$(realpath "$CONF")"
else
  CONF_ABS="$(cd "$(dirname "$CONF")" && pwd)/$(basename "$CONF")"
fi

# ===== 常量 =====
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
QDRANT_VER="${QDRANT_VER:-v1.12.4}"
BIN_DIR="${BIN_DIR:-${PROJECT_ROOT}/src/rag/bin}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/rag/logs}"
URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-gnu.tar.gz"

# ===== 依赖与目录 =====
need(){ command -v "$1" >/dev/null 2>&1 || { echo "[x] 缺少依赖: $1"; exit 1; }; }
mkdir -p "$LOG_DIR"
[[ -d "$BIN_DIR" ]] || mkdir -p "$BIN_DIR"
[[ -w "$BIN_DIR" ]] || { echo "[x] ${BIN_DIR} 不可写（需要 sudo?）"; exit 1; }

# ===== 安装 qdrant（二进制不存在才装） =====
if [[ ! -x "$BIN_DIR/qdrant" ]]; then
  if [[ -x "${PROJECT_ROOT}/src/rag/bin/qdrant" ]]; then
    if [[ "$BIN_DIR/qdrant" != "${PROJECT_ROOT}/src/rag/bin/qdrant" ]]; then
      cp "${PROJECT_ROOT}/src/rag/bin/qdrant" "$BIN_DIR/qdrant"
      chmod +x "$BIN_DIR/qdrant"
    fi
  else
    need curl; need tar
    echo "[*] 下载 Qdrant ${QDRANT_VER} ..."
    TMP_DIR="$(mktemp -d)"; trap 'rm -rf "$TMP_DIR"' EXIT
    curl -fsSL "$URL" -o "$TMP_DIR/qdrant.tgz"
    tar -xzf "$TMP_DIR/qdrant.tgz" -C "$TMP_DIR"
    QBIN="$(find "$TMP_DIR" -type f -name qdrant -perm -u+x | head -n1)"
    [[ -n "$QBIN" ]] || { echo "[x] 未找到 qdrant 可执行文件"; exit 1; }
    install -m 0755 "$QBIN" "$BIN_DIR/qdrant"
    echo "[✓] 已安装到 $BIN_DIR/qdrant"
  fi
fi

# ===== 端口解析（可选）=====
if command -v yq >/dev/null 2>&1; then
  PORT="$(yq -r '.service.http_port // 6333' "$CONF_ABS" 2>/dev/null || echo 6333)"
else
  PORT="${QDRANT_HTTP_PORT:-6333}"
fi

http_alive() { curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; }

# ===== 精确检测：遍历 qdrant 进程并比对 --config-path 的 realpath =====
is_running() {
  local pids cmdline cfg_arg cfg_real
  pids="$(pgrep -x qdrant || true)"
  [[ -n "$pids" ]] || return 1
  for pid in $pids; do
    # 有些系统 /proc 不可读；做保护
    [[ -r "/proc/$pid/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < /proc/"$pid"/cmdline)"
    [[ "$cmdline" == *"--config-path"* ]] || continue
    # 兼容 "--config-path /path" 与 "--config-path=/path"
    cfg_arg="$(sed -n 's/.*--config-path[= ]\([^ ]*\).*/\1/p' <<< "$cmdline" | tail -n1)"
    # 统一成真实路径比较
    cfg_real="$(readlink -f "$cfg_arg" 2>/dev/null || echo "$cfg_arg")"
    if [[ "$cfg_real" == "$CONF_ABS" ]]; then
      # 若能探活，再确认一次服务可用性
      http_alive || true
      return 0
    fi
  done
  return 1
}


# ===== 已运行检测 =====
if is_running; then
  echo "[i] Qdrant 已在运行（配置：${CONF_ABS}）"
  exit 0
fi

# ===== 启动 =====
echo "[*] 启动 Qdrant（配置：${CONF_ABS}）..."
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
nohup "$BIN_DIR/qdrant" --config-path "$CONF_ABS" > "$LOG_DIR/qdrant-${CURRENT_TIME}.out" 2>&1 &

# 等待探活（最多 10 次，每次 0.5s）
for _ in {1..10}; do
  if is_running; then
    echo "[✓] Qdrant 已启动 → http://127.0.0.1:${PORT}"
    echo "日志: $LOG_DIR/qdrant-${CURRENT_TIME}.out"
    exit 0
  fi
  sleep 0.5
done

echo "[x] 启动失败，查看日志: $LOG_DIR/qdrant-${CURRENT_TIME}.out"
exit 1
