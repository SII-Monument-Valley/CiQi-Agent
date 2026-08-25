"""Reference builder for the ciqi multimodal Qdrant artifacts.

Users are expected to download and clean a corpus themselves. This script only
shows how cleaned JSONL records are encoded into the NumPy + JSONL pairs read
by ``src.rag.api.ingest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence
import uuid

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm


ID_NAMESPACE = uuid.UUID("72006c3d-6c56-5a17-b83f-e8552cc89573")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def row_key(row: dict[str, Any]) -> str:
    supplied = str(row.get("uuid") or row.get("id") or "").strip()
    if supplied:
        return supplied
    content = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def point_id(channel: str, kind: str, identity: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{channel}:{kind}:{identity}"))


def image_names(row: dict[str, Any]) -> list[str]:
    value = row.get("images") or row.get("images_raw") or row.get("images_resized") or []
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value]


def resolve_image(name: str, roots: Sequence[Path]) -> tuple[Path, str] | None:
    candidate = Path(name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate, candidate.name
    for root in roots:
        path = root / candidate
        if path.is_file():
            return path, path.relative_to(root).as_posix()
    return None


def caption(row: dict[str, Any], max_chars: int = 120) -> str:
    parts = [
        row.get("name", ""),
        row.get("type", ""),
        f"{row.get('dynasty', '')}{row.get('reign', '')}".strip(),
        row.get("decoration", ""),
        row.get("color", ""),
        row.get("source", ""),
    ]
    description = str(row.get("description") or "").replace("\n", " ").strip()
    if max_chars and len(description) > max_chars:
        description = description[:max_chars] + "…"
    return "；".join(str(value) for value in [*parts, description] if value)


def image_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name", ""),
        "description": row.get("description", ""),
        "object_type": row.get("type", ""),
        "color": row.get("color", ""),
        "decoration": row.get("decoration", ""),
        "dynasty": row.get("dynasty", ""),
        "reign": row.get("reign", ""),
        "source": row.get("source", ""),
    }


class ClipEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import (
            AutoConfig,
            CLIPModel,
            CLIPProcessor,
            ChineseCLIPModel,
            ChineseCLIPProcessor,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(model_name)
        is_chinese = getattr(config, "model_type", "") == "chinese_clip"
        model_class = ChineseCLIPModel if is_chinese else CLIPModel
        processor_class = ChineseCLIPProcessor if is_chinese else CLIPProcessor
        self.model = model_class.from_pretrained(model_name, config=config).to(self.device).eval()
        self.processor = processor_class.from_pretrained(model_name)

    def images(self, values: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(values), return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vectors = self.model.get_image_features(**inputs)
        return normalize(vectors.float().cpu().numpy())

    def texts(self, values: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(values), return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vectors = self.model.get_text_features(**inputs)
        return normalize(vectors.float().cpu().numpy())


def write_pair(output_dir: Path, name: str, vectors: list[np.ndarray], metadata: list[dict]) -> None:
    if not vectors or not metadata:
        return
    matrix = np.concatenate(vectors).astype(np.float32, copy=False)
    if len(matrix) != len(metadata):
        raise ValueError(f"{name}: vector and metadata lengths differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}_vectors.npy", matrix, allow_pickle=False)
    with (output_dir / f"{name}_meta.jsonl").open("w", encoding="utf-8") as handle:
        for payload in metadata:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"{name}: {matrix.shape}")


def build_clip(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.images_jsonl)
    encoder = ClipEncoder(args.clip_model, args.device)
    image_items: list[tuple[Path, dict[str, Any]]] = []
    caption_items: list[tuple[str, dict[str, Any]]] = []

    for row in rows:
        key = row_key(row)
        names = image_names(row)
        for index, name in enumerate(names):
            resolved = resolve_image(name, args.image_root)
            if resolved is None:
                continue
            local_path, relative_path = resolved
            payload = {
                "id": point_id("clip", "image", f"{key}:{index}:{relative_path}"),
                "type": "image",
                "uuid": key,
                "image_file": Path(name).name,
                "image_path": relative_path,
                "image_index": index,
                **image_payload(row),
            }
            image_items.append((local_path, payload))

        text = caption(row, args.caption_max_chars)
        if text:
            first_name = names[0] if names else ""
            first_resolved = resolve_image(first_name, args.image_root) if first_name else None
            relative_path = first_resolved[1] if first_resolved else Path(first_name).name
            payload = {
                "id": point_id("clip", "clip_text", key),
                "type": "clip_text",
                "uuid": f"{key}#cap",
                "caption": text,
                "image_file": Path(first_name).name,
                "image_path": relative_path,
                **image_payload(row),
            }
            caption_items.append((text, payload))

    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    image_batches = (len(image_items) + args.image_batch_size - 1) // args.image_batch_size
    for chunk in tqdm(
        batched(image_items, args.image_batch_size), total=image_batches, desc="CLIP images"
    ):
        images = []
        for path, _ in chunk:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        vectors.append(encoder.images(images))
        metadata.extend(payload for _, payload in chunk)
        for image in images:
            image.close()
    caption_batches = (len(caption_items) + args.text_batch_size - 1) // args.text_batch_size
    for chunk in tqdm(
        batched(caption_items, args.text_batch_size),
        total=caption_batches,
        desc="CLIP captions",
    ):
        vectors.append(encoder.texts([text for text, _ in chunk]))
        metadata.extend(payload for _, payload in chunk)
    write_pair(args.output_dir, "clip", vectors, metadata)


def build_text(args: argparse.Namespace) -> None:
    import torch
    from FlagEmbedding import BGEM3FlagModel

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel(
        args.text_model, use_fp16=device.startswith("cuda"), devices=device
    )
    items: list[tuple[str, dict[str, Any]]] = []
    for row in read_jsonl(args.texts_jsonl):
        title = str(row.get("caption") or "").strip()
        body = str(row.get("text") or "").strip()
        text = f"{title}\n{body}".strip()
        if not text:
            continue
        identity = hashlib.sha256(
            json.dumps([row.get("source", ""), title, body], ensure_ascii=False).encode()
        ).hexdigest()
        payload = {
            "id": point_id("text", "long_text", identity),
            "type": "long_text",
            "source": row.get("source", ""),
            "caption": row.get("caption", ""),
            "text": row.get("text", ""),
            "perplexity": row.get("perplexity", 0.0),
        }
        items.append((text, payload))

    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    text_batches = (len(items) + args.text_batch_size - 1) // args.text_batch_size
    for chunk in tqdm(
        batched(items, args.text_batch_size), total=text_batches, desc="BGE text"
    ):
        result = model.encode(
            [text for text, _ in chunk],
            batch_size=args.text_batch_size,
            max_length=args.max_length,
            return_dense=True,
        )
        vectors.append(normalize(result["dense_vecs"]))
        metadata.extend(payload for _, payload in chunk)
    write_pair(args.output_dir, "text", vectors, metadata)


def export_faiss(args: argparse.Namespace) -> None:
    index = faiss.read_index(str(args.index))
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    if index.ntotal:
        index.reconstruct_n(0, index.ntotal, vectors)
    with args.metadata.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != index.ntotal:
        raise ValueError(f"metadata rows ({rows}) != vectors ({index.ntotal})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, vectors, allow_pickle=False)
    print(f"exported: {vectors.shape}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="encode a user-prepared corpus")
    build.add_argument("--images-jsonl", type=Path)
    build.add_argument("--texts-jsonl", type=Path)
    build.add_argument("--image-root", type=Path, action="append", default=[])
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--clip-model", default="OFA-Sys/chinese-clip-vit-large-patch14")
    build.add_argument("--text-model", default="BAAI/bge-m3")
    build.add_argument("--device", default="auto")
    build.add_argument("--image-batch-size", type=int, default=32)
    build.add_argument("--text-batch-size", type=int, default=64)
    build.add_argument("--caption-max-chars", type=int, default=120)
    build.add_argument("--max-length", type=int, default=8192)

    export = commands.add_parser("export-faiss", help="export a matching legacy index")
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--metadata", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "export-faiss":
        export_faiss(args)
        return
    if not args.images_jsonl and not args.texts_jsonl:
        raise SystemExit("provide --images-jsonl and/or --texts-jsonl")
    if args.images_jsonl:
        if not args.image_root:
            raise SystemExit("--image-root is required with --images-jsonl")
        build_clip(args)
    if args.texts_jsonl:
        build_text(args)


if __name__ == "__main__":
    main()
