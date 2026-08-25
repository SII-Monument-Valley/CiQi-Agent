# app/worker_index.py
import os, json, fcntl, time
from typing import Optional

SEQ_PATH = "/tmp/uvicorn_worker_seq.json"

def _read_state():
    if not os.path.exists(SEQ_PATH):
        return {"next": 0, "pid2idx": {}, "mtime": time.time(), "namespace": {}}
    try:
        with open(SEQ_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"next": 0, "pid2idx": {}, "mtime": time.time(), "namespace": {}}

def _write_state(state):
    tmp = SEQ_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, SEQ_PATH)

def assign_worker_index(slots: int, namespace: Optional[str] = None) -> int:
    """
    返回当前进程在指定命名空间下的 worker 序号（0..slots-1），跨进程唯一且近似稳定。
    namespace: 建议用 "port=8000" 或 "app=xxx,port=8000" 之类，避免不同服务冲突。
    """
    pid = os.getpid()
    ns = namespace or "default"
    os.makedirs(os.path.dirname(SEQ_PATH), exist_ok=True)

    with open(SEQ_PATH, "a+") as f:
        f.seek(0)
        # 原子加锁
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            state = _read_state()
            ns_state = state.setdefault("namespace", {}).setdefault(ns, {"next": 0, "pid2idx": {}})

            # 清理已经不存在的进程（粗略清理，不要求完美）
            dead = []
            for p in list(ns_state["pid2idx"].keys()):
                try:
                    os.kill(int(p), 0)
                except Exception:
                    dead.append(p)
            for p in dead:
                ns_state["pid2idx"].pop(p, None)

            spid = str(pid)
            if spid in ns_state["pid2idx"]:
                idx = ns_state["pid2idx"][spid]
            else:
                idx = ns_state["next"] % max(1, slots)
                ns_state["pid2idx"][spid] = idx
                ns_state["next"] += 1

            _write_state(state)
            return idx
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
