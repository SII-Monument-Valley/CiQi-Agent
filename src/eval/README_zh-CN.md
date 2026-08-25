# CiQi-Agent 评测

`ciqi-eval` 是独立的多模态智能体评测包，支持：

- OpenAI 及 OpenAI-compatible `/chat/completions`
- Anthropic Claude Messages API
- Google Gemini `generateContent`
- 图像放大、RAG 文本检索和 RAG 图像检索工具
- 按类别运行及分类汇总

完整配置字段与扩展接口见 [README.md](README.md)。

## 安装

需要 Python 3.11 和 uv：

```bash
cd src/eval
./scripts/setup_uv.sh
./scripts/setup_uv.sh --check
```

第二条命令会执行 lint、单元测试和示例配置校验。

## 下载测试集

数据集位于
[SII-Monument-Valley/CiQi-VQA](https://huggingface.co/datasets/SII-Monument-Valley/CiQi-VQA)。
下载脚本只获取 `test.jsonl` 及其中引用的图像：

```bash
./scripts/download_ciqi_vqa.sh
export CIQI_VQA_ROOT=/absolute/path/to/ciqi-agent/data/hf/ciqi-vqa
uv run ciqi-eval dataset validate --root "$CIQI_VQA_ROOT" --split test
```

需要固定数据版本时，把 Hugging Face commit SHA 作为第三个参数：

```bash
./scripts/download_ciqi_vqa.sh \
  SII-Monument-Valley/CiQi-VQA \
  /path/to/ciqi-vqa \
  COMMIT_SHA
```

## 配置模型

以本地 SGLang/OpenAI-compatible 服务为例：

```bash
export CIQI_EVAL_BACKEND_BASE_URL=http://127.0.0.1:18901/v1
export CIQI_EVAL_BACKEND_API_KEY=dummy
export CIQI_EVAL_BACKEND_MODEL=ciqi-agent
```

如果启用 RAG 工具，再设置：

```bash
export CIQI_AGENT_RAG_API_URL=http://127.0.0.1:8000
export CIQI_AGENT_RAG_API_KEY=''
```

Claude 和 Gemini 分别参考 `examples/ciqi-vqa/claude.yaml` 与
`examples/ciqi-vqa/gemini.yaml`。key 只能使用环境变量，不能写进 YAML。

## 运行

```bash
uv run ciqi-eval validate --config examples/ciqi-vqa/backend.yaml

# 默认评测全部类别
uv run ciqi-eval run --config examples/ciqi-vqa/backend.yaml

# 只评测某个类别
uv run ciqi-eval run \
  --config examples/ciqi-vqa/backend.yaml \
  --category color

# 小规模连通性测试
uv run ciqi-eval run \
  --config examples/ciqi-vqa/backend.yaml \
  --limit 10 \
  --output outputs/smoke
```

当前类别包括 `color`、`dynasty`、`emperor`、`kiln`、`overall`、`pattern`
和 `shape`。不传 `--category` 时全部运行；结果的 `summary.json` 同时包含总体分数
和 `categories` 分类分数。

每个 split 还会生成 `results-<split>.jsonl`，其中保存预测、解析答案、得分、
模型/工具调用轨迹和错误状态，便于复核。
