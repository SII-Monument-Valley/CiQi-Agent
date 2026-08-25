# app/encode_async_hf.py
import asyncio
from typing import List
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch

from .encode_sync_hf import (
    encode_text_bge_sync, encode_text_clip_sync, encode_image_clip_sync
)

POOL = ThreadPoolExecutor(max_workers=8)  # CPU 预处理/调度线程
GPU_SEM = asyncio.Semaphore(1)            # GPU 并发限流（视显存调大到2）

async def encode_text_bge(texts: List[str], batch_size: int = 64, normalize: bool = True):
    loop = asyncio.get_running_loop()
    async with GPU_SEM:
        return await loop.run_in_executor(POOL, encode_text_bge_sync, texts, batch_size, normalize)

async def encode_text_clip(texts: List[str], batch_size: int = 64, normalize: bool = True):
    loop = asyncio.get_running_loop()
    async with GPU_SEM:
        return await loop.run_in_executor(POOL, encode_text_clip_sync, texts, batch_size, normalize)

async def encode_image_clip(images: List[Image.Image], batch_size: int = 32, normalize: bool = True):
    loop = asyncio.get_running_loop()
    async with GPU_SEM:
        return await loop.run_in_executor(POOL, encode_image_clip_sync, images, batch_size, normalize)
