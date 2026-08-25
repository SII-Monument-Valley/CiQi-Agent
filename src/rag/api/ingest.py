import os, json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from qdrant_client.http.models.models import Distance
from qdrant_client.http import models as qm

from .ids import stable_point_id

load_dotenv()

URL = os.getenv("QDRANT_URL") or "http://127.0.0.1:6333"
KEY = os.getenv("QDRANT_API_KEY") or ""
COL = os.getenv("QDRANT_COLLECTION") or "multimodal_zh"
print(f"URL: {URL}")

DistanceType = {
    "COSINE": 'Cosine',
    "Dot": 'Dot',
    "EUCLID": 'Euclidean',
    "MANHATTAN": 'Manhattan'
}

CLIP_VEC = os.getenv("CLIP_VEC")
CLIP_META = os.getenv("CLIP_META")
TEXT_VEC = os.getenv("TEXT_VEC")
TEXT_META = os.getenv("TEXT_META")

CLIP_DIST = Distance[os.getenv("CLIP_DISTANCE", "COSINE")]
TEXT_DIST = Distance[os.getenv("TEXT_DISTANCE", "COSINE")]

client = QdrantClient(url=URL, api_key=KEY)

def load_pair(vec_path, meta_path):
    if not (vec_path and meta_path and os.path.exists(vec_path) and os.path.exists(meta_path)):
        return None, None
    vecs = np.load(vec_path)

    # 处理JSONL格式的元数据文件
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    meta.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, 行内容: {line[:100]}...")
                    continue

    assert len(vecs) == len(meta), f"Length mismatch: {vec_path} ({len(vecs)}) vs {meta_path} ({len(meta)})"
    return vecs, meta

def ensure_collection(clip_dim: int | None, text_dim: int | None):
    """
    仅用 clip_dim / text_dim 两个参数；其余（client, COL, CLIP_DIST, TEXT_DIST）沿用你的全局变量。
    策略：
      - 不存在 -> create_collection
      - 仅需新增命名向量 -> update_collection(VectorsConfigDiff)
      - 维度/距离不兼容，或历史为单向量形态/有多余向量 -> recreate_collection（会清空集合）
    """
    # 期望配置（命名向量）
    desired = {}
    if clip_dim:
        desired["clip"] = qm.VectorParams(size=int(clip_dim), distance=CLIP_DIST)
    if text_dim:
        desired["text"] = qm.VectorParams(size=int(text_dim),  distance=TEXT_DIST)

    if COL in {c.name for c in client.get_collections().collections}:
        client.recreate_collection(collection_name=COL, vectors_config=desired)
    else:
        client.create_collection(collection_name=COL, vectors_config=desired)


def gen_points(vecs, metas, vec_name):
    for v, m in zip(vecs, metas):
        pid = stable_point_id(m, vec_name)
        payload = dict(m)
        payload.setdefault("type", "clip" if vec_name=="clip" else "text")
        yield qm.PointStruct(
            id=pid,
            vector={vec_name: v.tolist()},
            payload=payload
        )

def batched(iterable, n=1024):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    clip_vecs, clip_meta = load_pair(CLIP_VEC, CLIP_META)
    text_vecs, text_meta = load_pair(TEXT_VEC, TEXT_META)
    clip_dim = clip_vecs.shape[1] if clip_vecs is not None else None
    text_dim = text_vecs.shape[1] if text_vecs is not None else None
    # assert clip_dim or text_dim, "No vectors found."

    ensure_collection(clip_dim, text_dim)

    if clip_vecs is not None:
        for chunk in batched(gen_points(clip_vecs, clip_meta, "clip")):
            client.upsert(collection_name=COL, points=chunk)

    if text_vecs is not None:
        for chunk in batched(gen_points(text_vecs, text_meta, "text")):
            client.upsert(collection_name=COL, points=chunk)

    count = client.count(COL, exact=True).count
    print(f"Upsert completed. Total points in `{COL}`: {count}")

if __name__ == "__main__":
    main()
