import asyncio, io, hashlib
import os
from typing import List, Optional, Literal, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from collections import OrderedDict
import numpy as np
from .encode_async_hf import encode_text_bge, encode_text_clip, encode_image_clip

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COL = os.getenv("QDRANT_COLLECTION", "multimodal_zh")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "10"))
IMG_EMB_CACHE_CAP = int(os.getenv("IMG_EMB_CACHE_CAP", "10000"))  # 最大缓存条数，可按需调
_IMG_EMB_CACHE = OrderedDict()  # key: sha256(hex) -> np.ndarray(float32, shape=(D,))
_IMG_EMB_LOCK = asyncio.Lock()

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI(title="Multimodal RAG (Qdrant)")

async def _cache_get(key: str):
    async with _IMG_EMB_LOCK:
        v = _IMG_EMB_CACHE.get(key)
        if v is not None:
            _IMG_EMB_CACHE.move_to_end(key)  # LRU: 最近使用放到尾部
        return v

async def _cache_set(key: str, value: np.ndarray):
    async with _IMG_EMB_LOCK:
        _IMG_EMB_CACHE[key] = value
        _IMG_EMB_CACHE.move_to_end(key)
        # 超容量淘汰最久未用
        while len(_IMG_EMB_CACHE) > IMG_EMB_CACHE_CAP:
            _IMG_EMB_CACHE.popitem(last=False)

class RawTextReq(BaseModel):
    texts: List[str]
    topk: int = 10
    alpha: float = 0.6

class TextSearchReq(BaseModel):
    # bge-m3 文本向量（文本→文本）
    text_vec: Optional[List[float]] = None
    # chinese-clip 文本塔向量（文本→图像）
    clip_text_vec: Optional[List[float]] = None
    topk: int = DEFAULT_TOPK
    alpha: float = 0.6  # 文本通道权重
    type_filter: Optional[Literal["clip","text"]] = None

class ImageSearchReq(BaseModel):
    # chinese-clip 图像塔向量（以图搜图）
    clip_vec: List[float]
    topk: int = DEFAULT_TOPK
    type_filter: Optional[Literal["clip"]] = None

def mk_filter(t: Optional[str]) -> Optional[Filter]:
    if not t: return None
    return Filter(must=[FieldCondition(key="type", match=MatchValue(value=t))])


@app.get("/healthz")
def healthz():
    _ = client.get_collections()
    return {"ok": True}

@app.post("/search/text/embed")
def search_text_embed(req: TextSearchReq):
    results = []
    if req.text_vec:
        r_text = client.search(
            collection_name=COL,
            query_vector=("text", req.text_vec),
            limit=req.topk,
            query_filter=mk_filter(req.type_filter)
        ); results.append(("text", r_text))
    if req.clip_text_vec:
        r_clip = client.search(
            collection_name=COL,
            query_vector=("clip", req.clip_text_vec),
            limit=req.topk,
            query_filter=mk_filter(req.type_filter)
        ); results.append(("clip", r_clip))

    fused: Dict[str, float] = {}
    for name, hits in results:
        if not hits: continue
        scores = [h.score for h in hits]
        smin, smax = min(scores), max(scores)
        for h in hits:
            # 将得分粗略归一为“相似度”并加权融合
            norm = 1.0 - (h.score - smin)/(smax - smin + 1e-9)
            w = req.alpha if name=="text" else (1-req.alpha)
            fused[str(h.id)] = max(fused.get(str(h.id), 0.0), norm*w)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:req.topk]
    out = []
    if ranked:
        ids = [rid for rid,_ in ranked]
        recs = {str(p.id): p for p in client.retrieve(COL, ids=ids)}
        for rid, sc in ranked:
            p = recs.get(rid)
            out.append({"id": rid, "score": sc, "payload": p.payload if p else {}})
    return {"hits": out}

@app.post("/search/image/embed")
def search_image_embed(req: ImageSearchReq):
    r = client.search(
        collection_name=COL,
        query_vector=("clip", req.clip_vec),
        limit=req.topk,
        query_filter=mk_filter(req.type_filter)
    )
    return {"hits": [{"id": str(h.id), "score": h.score, "payload": h.payload} for h in r]}

@app.post("/search/text")
async def search_text(req: RawTextReq):
    # 并发计算两路向量
    bge_vecs, clip_txt_vecs = await asyncio.gather(
        encode_text_bge(req.texts), encode_text_clip(req.texts)
    )
    n = len(req.texts)
    async def _one(i: int):
        payload = TextSearchReq(
            text_vec=bge_vecs[i].tolist(),
            clip_text_vec=clip_txt_vecs[i].tolist(),
            topk=req.topk,
            alpha=req.alpha,
        )
        # 如果 search_text_embed 是同步函数，用 asyncio.to_thread 包一层
        return await asyncio.to_thread(search_text_embed, payload)

    results = await asyncio.gather(*[_one(i) for i in range(n)])
    return {"results": results}

@app.post("/search/image")
async def search_image(files: List[UploadFile] = File(...), topk: int = Form(10)):
    # 1) 读取 bytes，并计算内容哈希（SHA-256）
    bufs: List[bytes] = [await f.read() for f in files]
    keys: List[str] = [hashlib.sha256(b).hexdigest() for b in bufs]

    # 2) 命中缓存的直接取向量；未命中的收集起来做一次性编码
    vecs: List[np.ndarray | None] = [None] * len(bufs)
    miss_imgs, miss_idx = [], []

    for i, (key, b) in enumerate(zip(keys, bufs)):
        v = await _cache_get(key)
        if v is not None:
            vecs[i] = v
        else:
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                miss_imgs.append(img)
                miss_idx.append(i)
            except Exception:
                # 该图片坏了或格式不支持：让它返回空结果
                vecs[i] = None

    # 3) 对未命中的图片批量编码，并写入缓存
    if miss_imgs:
        miss_mat = await encode_image_clip(miss_imgs)
        for i, row in zip(miss_idx, miss_mat):
            vecs[i] = row
            await _cache_set(keys[i], row)

    # 4) 逐个查询；对解码/编码失败的条目返回空结果
    async def _one(i: int):
        if vecs[i] is None:
            return []  # 或者返回 {"error":"bad_image"} 看你的需求
        req = ImageSearchReq(clip_vec=vecs[i].tolist(), topk=topk)
        # 如果 search_image_embed 是同步函数，用线程池包一层
        return await asyncio.to_thread(search_image_embed, req)

    results = await asyncio.gather(*[_one(i) for i in range(len(vecs))])
    return {"results": results}
