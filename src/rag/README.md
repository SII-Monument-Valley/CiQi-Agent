# ciqi-agent RAG reference

本目录仅提供当前 RAG 实现的参考代码，不提供一键复刻数据处理流程，也不随
仓库发布原始图像、文本、metadata、向量文件、Qdrant 数据目录或模型权重。
使用者需要自行下载有权使用的图像和文字，并按自己的数据来源完成清洗。

## 参考文件

- `indexing/build_index.py`：合并后的参考脚本，包含 Chinese-CLIP 图像/短文本
  编码、BGE-M3 长文本编码，以及旧 FAISS 索引导出。
- `api/ingest.py`：将 `.npy + JSONL metadata` 写入 Qdrant。
- `api/main.py`：文本和图像检索 API。
- `client/`：调用检索 API 的示例客户端。

参考实现接受两类 JSONL。图像记录可包含：

```json
{"uuid":"...","images_raw":["image.jpg"],"name":"...","description":"...","type":"...","source":"..."}
```

文本记录可包含：

```json
{"source":"...","caption":"...","text":"...","perplexity":0.0}
```

字段只用于说明当前实现的连接方式，并不是要求外部数据集采用相同格式。使用者
可以修改 `indexing/build_index.py` 中的 JSONL 读取和 metadata 映射。

## 环境和调用

```bash
uv sync --project src/rag
```

公开仓库不提交 `uv.lock`。默认可从 PyPI 安装，也可以通过 `UV_INDEX_URL` 或
`UV_DEFAULT_INDEX` 指定其他公共源。

用自行准备的数据生成 Qdrant 导入产物：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project src/rag \
  python -m src.rag.indexing.build_index build \
  --images-jsonl /path/to/images.jsonl \
  --image-root /path/to/images \
  --texts-jsonl /path/to/texts.jsonl \
  --output-dir /path/to/output
```

两个 JSONL 参数均为可选，可以只生成图像或文本通道。旧 FAISS 索引如需转换：

```bash
uv run --project src/rag python -m src.rag.indexing.build_index export-faiss \
  --index /path/to/index \
  --metadata /path/to/matching_meta.jsonl \
  --output /path/to/vectors.npy
```

索引和 metadata 必须来自同一次构建，不能只按行数相同就混用。准备好自己的
四个产物后，可在 `configs/rag/.env` 中配置路径并启动：

```bash
bash scripts/rag/start_api_distributed.sh
```

本参考实现只说明技术连接关系。数据授权、下载、清洗、去重、脱敏和最终发布
范围由使用者自行负责。
