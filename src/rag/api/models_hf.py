# app/models_hf.py
import torch
import numpy as np
from typing import Tuple, Any
from FlagEmbedding import BGEM3FlagModel
from transformers import CLIPModel, CLIPProcessor
import os
from .worker_index import assign_worker_index

# 可选：如果环境支持 ChineseCLIPModel/Processor，优先 import
try:
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # type: ignore
    HAS_CHINESE = True
except Exception:
    ChineseCLIPModel = None  # type: ignore
    ChineseCLIPProcessor = None  # type: ignore
    HAS_CHINESE = False

def bind_gpu():
    num_gpus = torch.cuda.device_count()
    slots = max(1, num_gpus)  # 没有 GPU 时也给个槽位，避免除零
    ns = f"port={os.getenv('UVICORN_PORT', '8000')}"
    idx = assign_worker_index(slots=slots, namespace=ns)

    if num_gpus > 0:
        torch.cuda.set_device(idx % num_gpus)
        print(f"[worker pid={os.getpid()}] assigned idx={idx}, bind cuda:{idx % num_gpus}")
        return f"cuda:{idx % num_gpus}"
    else:
        print(f"[worker pid={os.getpid()}] assigned idx={idx}, CPU mode")
        return "cpu"

DEVICE = bind_gpu()
DTYPE  = torch.float16 if "cuda" in DEVICE else torch.float32
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "OFA-Sys/chinese-clip-vit-large-patch14")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "BAAI/bge-m3")

# ------------ BGE-M3 ------------
# 你给的方式
bge = BGEM3FlagModel(TEXT_MODEL_NAME, use_fp16=("cuda" in DEVICE))

# ------------ (Chinese) CLIP ------------
def _load_clip_model(model_name: str) -> Tuple[Any, Any, int]:
    """
    使用 Hugging Face 的 CLIP/Chinese-CLIP。
    返回: (model, processor, embedding_dim)
    """
    try:
        use_chinese = ("chinese" in model_name.lower() or "OFA-Sys/chinese-clip" in model_name) and HAS_CHINESE
        model_class = ChineseCLIPModel if use_chinese else CLIPModel
        processor_class = ChineseCLIPProcessor if use_chinese else CLIPProcessor

        model = model_class.from_pretrained(model_name).to(DEVICE)
        processor = processor_class.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        raise ValueError(f"加载CLIP模型失败: {e}")

# 按你的模型名来，这里给一个常见默认
clip_model, clip_proc = _load_clip_model(CLIP_MODEL_NAME)

# --------- 常用工具 ---------
@torch.no_grad()
def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n
