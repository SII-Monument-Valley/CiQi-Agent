"""Pure point-ID helpers shared by ingestion and tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
import uuid


POINT_NAMESPACE = uuid.UUID("72006c3d-6c56-5a17-b83f-e8552cc89573")


def stable_point_id(meta: Mapping[str, Any], vector_name: str) -> str:
    if meta.get("id"):
        return str(meta["id"])
    canonical = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(POINT_NAMESPACE, f"{vector_name}:legacy:{digest}"))
