# app/encode_sync_hf.py
from typing import List
import numpy as np
import torch
from PIL import Image

from .models_hf import bge, clip_model, clip_proc, DEVICE, DTYPE, l2_normalize_np

def encode_text_bge_sync(texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """
    FlagEmbedding.BGEM3FlagModel
    返回 dense 向量；如需稀疏可设置 return_sparse=True 并合并策略。
    """
    # 官方接口：model.encode(corpus, return_dense=True/False, return_sparse=True/False)
    out = bge.encode(texts, batch_size=batch_size, max_length=8192,
                     return_dense=True)
    emb = np.asarray(out["dense_vecs"], dtype=np.float32)
    return l2_normalize_np(emb) if normalize else emb

def encode_text_clip_sync(texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = clip_proc(text=batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feat = clip_model.get_text_features(**inputs)
        feats.append(feat.float().cpu().numpy())
    emb = np.concatenate(feats, axis=0)
    return l2_normalize_np(emb) if normalize else emb

def encode_image_clip_sync(images: List[Image.Image], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    feats = []
    # 处理成像素张量
    pixel_batches = []
    buf = []
    for img in images:
        # 确保 RGB
        if img.mode != "RGB": img = img.convert("RGB")
        buf.append(clip_proc(images=img, return_tensors="pt")["pixel_values"])
        if len(buf) >= batch_size:
            pixel_batches.append(torch.cat(buf, dim=0)); buf = []
    if buf: pixel_batches.append(torch.cat(buf, dim=0))

    for pb in pixel_batches:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            feat = clip_model.get_image_features(pixel_values=pb.to(DEVICE, dtype=DTYPE))
        feats.append(feat.float().cpu().numpy())
    emb = np.concatenate(feats, axis=0)
    return l2_normalize_np(emb) if normalize else emb
