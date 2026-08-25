<h1 align="center">
  <img src="assets/ciqi-agent-logo.png" alt="CiQi-Agent logo" width="88" valign="middle">
  CiQi-Agent
</h1>

<p align="center">
  <a href="https://github.com/SII-Monument-Valley/CiQi-Agent">Github</a> |
  <a href="https://huggingface.co/SII-Monument-Valley/CiQi-Agent-7B">Model</a> |
  <a href="https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA">Dataset</a> |
  <a href="https://arxiv.org/abs/2603.28474">Paper</a>
</p>

<p align="center">
  <strong>CiQi-Agent: Aligning Vision, Tools and Aesthetics in Multimodal Agent for Cultural Reasoning on Chinese Porcelains</strong><br>
  <strong>Accepted to ECCV 2026</strong>
</p>

## 🎯 Overview

**CiQi-Agent has been accepted to ECCV 2026.**

We present **CiQi-Agent**, a domain-specific multimodal agent for **antique Chinese porcelain connoisseurship**. The project is designed to combine **fine-grained visual perception**, **tool-augmented reasoning**, and **cultural-heritage knowledge grounding** for explainable porcelain analysis.

CiQi-Agent is built for **tool-augmented multimodal reasoning** on antique Chinese porcelains. During inference, it can inspect local visual evidence with an **image zoom-in tool**, retrieve visually similar examples with **image retrieval**, and access relevant domain knowledge with **text retrieval**, enabling more grounded and interpretable connoisseurship analysis.

Alongside the model, we release a single Hugging Face dataset repository that contains both the training data and benchmark data:

- **CiQi-VQA**, covering **29,596 porcelain specimens**, **51,553 images**, and **557,943 VQA pairs**
- **CiQi-Bench**, the benchmark portion included in the CiQi-VQA repository, built from **775 porcelain specimens**, **878 images**, and **5,425 multiple-choice questions**

On CiQi-Bench, CiQi-Agent achieves **81.5%** average accuracy on multiple-choice evaluation and **66.7%** average score on free-form evaluation, outperforming strong open-source and closed-source multimodal baselines.

## 📁 Repository

This repository contains the public reference implementation for three parts of the CiQi-Agent project:

- **Evaluation** (`src/eval`): a reproducible CiQi-VQA test-set evaluator with category filtering, per-category reporting, tool use, and OpenAI-compatible, Claude, and Gemini model adapters.
- **Model deployment** (`src/model_server` and `scripts/model`): SGLang serving scripts for CiQi-Agent-7B, including a conservative single-GPU BF16 profile.
- **RAG data preparation** (`src/rag`): reference code for encoding user-provided images and text, importing vectors into Qdrant, and calling the retrieval API.

```text
CiQi-Agent/
├── assets/                  # Project logo
├── configs/rag/             # Public Qdrant and environment examples
├── scripts/model/           # SGLang deployment and smoke tests
├── scripts/rag/             # Local Qdrant and retrieval API launchers
└── src/
    ├── eval/                # Evaluation package, configs, and tests
    ├── model_server/        # Model-serving environment and documentation
    └── rag/                 # Reference indexing, API, and client code
```

The repository does not include training checkpoints, raw RAG corpora, vector databases, service credentials, or private deployment configuration. Python environments use [uv](https://docs.astral.sh/uv/); lock files are intentionally not committed, so dependencies may be installed from any public package index.

### Evaluation quick start

The evaluator uses the `test` data in the CiQi-VQA repository. Omitting `--category` evaluates all categories; supplying it evaluates only the requested category. Reports contain both aggregate and category-level results.

```bash
cd src/eval
./scripts/setup_uv.sh
./scripts/download_ciqi_vqa.sh

export CIQI_VQA_ROOT=/path/to/CiQi-VQA
export CIQI_EVAL_BACKEND_BASE_URL=http://127.0.0.1:18901/v1
export CIQI_EVAL_BACKEND_API_KEY=EMPTY
export CIQI_EVAL_BACKEND_MODEL=ciqi-agent

uv run ciqi-eval validate --config examples/ciqi-vqa/backend.yaml
uv run ciqi-eval run --config examples/ciqi-vqa/backend.yaml
uv run ciqi-eval run --config examples/ciqi-vqa/backend.yaml --category color
```

See [`src/eval/README.md`](src/eval/README.md) for the dataset layout, provider-specific Claude and Gemini examples, tool configuration, and output schema.

## 🤖 Model

- Released as **[SII-Monument-Valley/CiQi-Agent-7B](https://huggingface.co/SII-Monument-Valley/CiQi-Agent-7B)**
- Built on **[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** with **8.29B parameters**
- Architecture: `Qwen2_5_VLForConditionalGeneration`
- Inputs: text, one or more images, and optional tool definitions/results
- Outputs: text and optional Qwen-style tool calls
- Uses **tool-augmented reasoning** with zoom-in, image retrieval, and text retrieval
- Trained with a **two-phase supervised fine-tuning + reinforcement learning pipeline**
- Evaluated on **CiQi-Bench** with both **multiple-choice** and **free-form** protocols
- Achieves **81.5%** average accuracy on multiple-choice evaluation, exceeding **GPT-5 by 5.7 points** and the strongest listed open-source baseline **GLM-4.5V (72.6%)** by **8.9 points**
- Achieves **66.7%** average score on free-form evaluation, exceeding **GPT-5 by 18.7 points** and **Qwen2.5-VL-72B-Instruct by 23.7 points**

The released checkpoint is stored in FP32 (approximately 33.2 GB) and can be downcast to BF16 by SGLang while loading. A conservative 4090 profile with a 4K context, an 8192-token KV cache pool, one running request, and CUDA graph disabled completed an OpenAI-compatible chat request using **17,676 MiB**. The measurement was collected on a platform-provided 48 GB RTX 4090; it is below a 24 GB memory budget, but a standard 24 GB RTX 4090 was not directly tested. H200-class hardware remains recommended for longer contexts, higher concurrency, and production deployment.

## 🚀 Deployment

Install a recent SGLang release compatible with Qwen2.5-VL. The following command serves the model through an OpenAI-compatible API:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path SII-Monument-Valley/CiQi-Agent-7B \
  --host 127.0.0.1 \
  --port 18901 \
  --served-model-name ciqi-agent \
  --tp 1 \
  --dp 1 \
  --dtype bfloat16 \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --tool-call-parser qwen25 \
  --trust-remote-code \
  --enable-multimodal
```

For tensor parallel deployment, expose the required GPUs and set `--tp` to the same GPU count, for example `CUDA_VISIBLE_DEVICES=0,1` with `--tp 2`.

### Low-memory RTX 4090 profile

Use this configuration when targeting a 24 GB memory budget:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path SII-Monument-Valley/CiQi-Agent-7B \
  --host 127.0.0.1 \
  --port 18901 \
  --served-model-name ciqi-agent \
  --tp 1 \
  --dp 1 \
  --dtype bfloat16 \
  --context-length 4096 \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --chunked-prefill-size 1024 \
  --mem-fraction-static 0.88 \
  --disable-cuda-graph \
  --tool-call-parser qwen25 \
  --trust-remote-code \
  --enable-multimodal
```

This profile preserves BF16 model computation; `--max-total-tokens` limits KV-cache capacity rather than quantizing the model. A single conversation can use up to the configured 4096-token context, including text history, visual tokens, and output. Stop other GPU processes before startup and increase context, concurrency, or the KV-token pool only after checking available memory.

The low-memory configuration reduces context length and concurrency, not the precision of the loaded model. Further weight quantization may reduce memory use, but it can also affect multimodal reasoning and tool-call reliability and has not been validated for this release.

Do not expose the raw model server directly to the public internet. Put it behind TLS, authentication, rate limiting, request-size limits, logging, and key-revocation controls.

### OpenAI-compatible chat

After the server becomes ready:

```bash
curl http://127.0.0.1:18901/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ciqi-agent",
    "messages": [
      {"role": "user", "content": "Please introduce yourself in one sentence."}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

For an image reachable by the model server:

```json
{
  "model": "ciqi-agent",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {"url": "https://example.org/object.jpg"}
        },
        {"type": "text", "text": "Describe the object and explain the visual evidence."}
      ]
    }
  ]
}
```

### Tool calling

The model follows the Qwen2.5 tool-calling convention. When serving with SGLang, keep `--tool-call-parser qwen25` enabled and pass tools through the OpenAI-compatible `tools` field. Tool execution remains the responsibility of the client or agent runtime; return each tool result to the model before asking it to continue.

## 🔎 RAG Data Preparation Reference

The RAG module is intentionally a reference implementation rather than a packaged corpus. Users should download and clean images and text that they are authorized to use, then adapt the JSONL field mapping when necessary.

```bash
uv sync --project src/rag

CUDA_VISIBLE_DEVICES=0 uv run --project src/rag \
  python -m src.rag.indexing.build_index build \
  --images-jsonl /path/to/images.jsonl \
  --image-root /path/to/images \
  --texts-jsonl /path/to/texts.jsonl \
  --output-dir /path/to/rag-artifacts
```

The generated NumPy vectors and JSONL metadata can be imported into Qdrant with `src/rag/api/ingest.py`. Image and text channels may be built independently. See [`src/rag/README.md`](src/rag/README.md) for the input examples, old-FAISS export command, API startup, and data-safety notes.

Raw corpora, metadata, embedding files, and Qdrant storage are not distributed in this repository. Embeddings should not be treated as irreversible anonymization; private retrieval data should remain behind an authenticated service instead of being published as a vector snapshot.

## 📊 Model Performance

### Multiple-Choice Accuracy (%) on CiQi-Bench

| Model                       | Dynasty  | Reign    | Kiln     | Color    | Motif    | Shape    | Naming   | Average  |
| --------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| GPT-5                       | 65.7     | 61.4     | 79.6     | 86.5     | 69.3     | 83.8     | 84.3     | 75.8     |
| GPT-4.1                     | 59.3     | 68.3     | 71.1     | 85.0     | 62.2     | 81.8     | 77.9     | 72.2     |
| GPT-4o                      | 59.1     | 60.4     | 68.6     | 89.2     | 70.1     | 84.2     | 82.1     | 73.4     |
| o3                          | 57.6     | 57.4     | 72.2     | 82.6     | 62.4     | 76.8     | 76.6     | 69.4     |
| Qwen2.5-VL-72B-Instruct     | 57.6     | 34.7     | 69.2     | 86.7     | 71.7     | 84.1     | 80.3     | 69.2     |
| GLM-4.5V (106B)             | 58.3     | 59.4     | 75.8     | 82.3     | 70.4     | 81.8     | 80.6     | 72.6     |
| InternVL3.5-241B-A28B-Flash | 57.1     | 38.6     | 59.5     | 82.1     | 64.8     | 73.9     | 68.5     | 63.5     |
| Kimi-VL-A3B-Instruct (16B)  | 59.3     | 22.8     | 48.8     | 84.8     | 59.8     | 77.9     | 70.3     | 60.5     |
| **CiQi-Agent (Ours, 7B)**   | **77.6** | **70.3** | **81.8** | **91.4** | **75.7** | **88.1** | **85.2** | **81.5** |

### Free-Form Score (%) on CiQi-Bench

| Model                       | Dynasty  | Reign    | Kiln     | Color    | Motif    | Shape    | Average  |
| --------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| GPT-5                       | 39.4     | 32.8     | 42.6     | 74.4     | 35.3     | 63.9     | 48.0     |
| GPT-4.1                     | 36.7     | 27.2     | 29.0     | 67.5     | 27.6     | 60.1     | 41.3     |
| GPT-4o                      | 26.9     | 13.4     | 15.1     | 53.9     | 21.1     | 47.6     | 29.7     |
| o3                          | 42.7     | 36.6     | 44.4     | 74.2     | 33.1     | 62.1     | 48.8     |
| Qwen2.5-VL-72B-Instruct     | 29.5     | 31.2     | 27.7     | 75.8     | 31.0     | 62.6     | 43.0     |
| GLM-4.5V (106B)             | 31.0     | 14.3     | 32.8     | 65.4     | 31.1     | 65.2     | 39.9     |
| InternVL3.5-241B-A28B-Flash | 42.4     | 31.6     | 36.9     | 52.6     | 19.6     | 41.5     | 37.4     |
| Kimi-VL-A3B-Instruct (16B)  | 17.3     | 23.7     | 16.2     | 69.5     | 26.5     | 61.3     | 35.7     |
| **CiQi-Agent (Ours, 7B)**   | **71.3** | **49.1** | **69.8** | **85.4** | **49.7** | **75.0** | **66.7** |

## 📦 Dataset & Benchmark

The **[SII-Monument-Valley/CiQi-VQA](https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA)** repository contains both the CiQi-VQA training data and the CiQi-Bench evaluation data.

### 📊 CiQi-VQA Training Data

**CiQi-VQA** is a large-scale dataset for porcelain-centered multimodal training.

- 29,596 porcelain specimens
- 51,553 images
- 557,943 VQA pairs
- 38 dynasties
- 42 reign periods
- 246 glaze color categories
- 248 decorative motif categories
- 158 vessel shape categories

Link: [https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA](https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA)

### 🧪 CiQi-Bench Evaluation Data

**CiQi-Bench** is the expert-aligned benchmark portion of the same CiQi-VQA repository for evaluating porcelain connoisseurship ability.

- 775 porcelain specimens
- 878 images
- 5,425 multiple-choice questions
- Free-form evaluation with attribute-wise scoring

Repository: [https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA](https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA)

### 📈 Dataset and Benchmark Statistics

| Split / Resource      | Porcelains | Images | VQA Questions | Multiple-Choice Questions | Attributes                                |
| --------------------- | ---------- | ------ | ------------- | ------------------------- | ----------------------------------------- |
| CiQi-VQA SFT          | 28,821     | 50,675 | 557,168       | ---                       | dynasty, reign, kiln, color, motif, shape |
| CiQi-VQA RL subset    | 10,275     | 10,275 | 10,275        | ---                       | dynasty, reign, kiln, color, motif, shape |
| CiQi-Bench Evaluation | 775        | 878    | 775           | 5,425                     | dynasty, reign, kiln, color, motif, shape |
| Total                 | 29,596     | 51,553 | 557,943       | 5,425                     | dynasty, reign, kiln, color, motif, shape |

## Intended Use and Limitations

The model is intended for research and non-commercial cultural-heritage visual question answering, multimodal reasoning, retrieval-augmented generation, and agent/tool-use experiments.

- Outputs may be factually incorrect and should not be treated as provenance, authentication, attribution, appraisal, conservation, or legal advice.
- The model may inherit biases and limitations from its base model and training data.
- Users should verify museum, collection, licensing, and attribution metadata against authoritative sources.
- Static repository scans found no plaintext credentials or private deployment paths in this release. This does not prove that a neural model cannot reproduce fragments of its training data; deployments handling sensitive inputs should apply independent memorization and extraction-risk testing.
- Do not send confidential images or text to a third-party deployment unless its retention and access policies are acceptable.

## 📜 License

- **Source code license**: [Apache License 2.0](LICENSE)
- **Model license**: [CC BY-NC 4.0](https://huggingface.co/SII-Monument-Valley/CiQi-Agent-7B)
- **Dataset license**: CC BY-NC 4.0

Apache-2.0 applies only to the source code in this repository. Model weights and dataset contents remain governed by their respective licenses.

Users are also responsible for complying with the licenses and terms of the base model, source images, and any external retrieval corpus they use.

## 🤝 Acknowledgement

We thank **[Verl](https://github.com/volcengine/verl)** for providing an open-source reinforcement learning framework that supports this line of research.

We also thank **[DeepEyes](https://github.com/Visual-Agent/DeepEyes)** for inspiring and informing our exploration of tool-augmented multimodal reasoning.

## 📜 Citation

```bibtex
@inproceedings{wang2026ciqiagent,
      title={CiQi-Agent: Aligning Vision, Tools and Aesthetics in Multimodal Agent for Cultural Reasoning on Chinese Porcelains},
      author={Wenhan Wang and Zhixiang Zhou and Zhongtian Ma and Yanzhu Chen and Ziyu Lin and Hao Sheng and Pengfei Liu and Honglin Ma and Wenqi Shao and Qiaosheng Zhang and Yu Qiao},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2026},
      eprint={2603.28474},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.28474},
}
```
