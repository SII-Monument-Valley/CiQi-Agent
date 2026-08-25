# ciqi-eval

`ciqi-eval` is a standalone, reproducible evaluation package for multimodal
agents that use OpenAI-compatible, Anthropic Claude, or Google Gemini chat
models and local or remote tools. It
contains the evaluation architecture used by CiQi-Agent without depending on
the web backend, private absolute paths, or credentials embedded in source.

## Design

```text
YAML config
   │
   ├── datasets ──> versioned sample adapter ──> task prompt/metric
   ├── model ─────> OpenAI / Anthropic / Gemini adapter
   ├── tools ─────> ToolRegistry ─┬─ image_zoom_in_tool
   │                              ├─ search_text
   │                              └─ search_image
   └── runtime ───> EvaluationRunner ──> JSONL traces + summary.json
```

The package is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `config.py` | Versioned YAML schema, environment expansion, path resolution, legacy conversion |
| `io.py` | JSON/JSONL datasets, safe relative image paths, atomic output |
| `tasks.py` | Selection and open QA prompt/metric behavior |
| `model.py` | OpenAI chat-completions, Anthropic Messages, and Gemini generateContent adapters |
| `dataset.py` | CiQi-VQA `test.jsonl` layout and referenced-image validation |
| `conversation.py` | Multimodal conversation state compatible with the backend protocol |
| `tooling.py` | Tool schemas, registry, zoom implementation, RAG client |
| `runner.py` | Bounded concurrency, agent loop, traces, split/overall metrics |
| `judge.py` | Optional six-dimension LLM judge for open QA |
| `cli.py` | `run`, `validate`, and `score` commands |

Both native OpenAI tool calls and Hermes `<tool_call>...</tool_call>` output are
accepted.  Tool results are returned to the model using the same multimodal
conversation convention as the current CiQi-Agent backend.  Tool failures are
recorded and returned to the model so an evaluation sample can continue with
the evidence already available.

## uv environment

Python 3.11 or newer is required.

```bash
cd src/eval
./scripts/setup_uv.sh
./scripts/setup_uv.sh --check  # also runs lint, tests, and config validation
```

The script executes `uv sync --extra dev --extra hf`. The public repository
does not commit a lock file, so users may install from PyPI or select another
public package index with uv's `UV_INDEX_URL` or `UV_DEFAULT_INDEX`. It installs
the evaluator, test tools, and the current `hf` CLI into an isolated `.venv`;
activating the environment is optional because all commands can be launched
with `uv run`.

No API credential belongs in YAML.  Set environment variables instead:

```bash
export OPENAI_BASE_URL='https://model.example.com/v1'
export OPENAI_API_KEY='...'
export CIQI_AGENT_MODEL='ciqi-agent'
export CIQI_AGENT_RAG_API_URL='https://rag.example.com'
export CIQI_AGENT_RAG_API_KEY='...'
```

The example YAML uses `${NAME:-default}` interpolation.  `${NAME}` without a
default is treated as required and produces a clear configuration error.

## Run

```bash
ciqi-eval validate --config examples/config.yaml
ciqi-eval run --config examples/config.yaml
ciqi-eval run --config examples/config.yaml --limit 10 --output ./outputs/debug

# Omit --category to evaluate every type; provide it to run only that type.
ciqi-eval run --config examples/ciqi-vqa/backend.yaml --category color
```

CiQi-VQA categories currently include `color`, `dynasty`, `emperor`, `kiln`,
`overall`, `pattern`, and `shape`. Category matching is case-insensitive. An
unknown value fails before model requests and lists the available categories.
Every `summary.json` reports both overall metrics and a `categories` mapping;
each split also contains its own categorized metrics. Result JSONL rows retain
their canonical `category`, and `category_filter` records whether a filtered
run was requested.

The same commands work without installation from this directory:

```bash
uv run python -m ciqi_eval run --config examples/config.yaml
```

## CiQi-VQA test-set connection

The supported evaluation input is the `test` split from
`SII-Monument-Valley/CiQi-VQA`. The downloader first obtains `test.jsonl`, then
downloads only the image paths referenced by those rows. It deliberately
excludes training metadata and keeps the Hub layout unchanged (`test.jsonl`
plus `images/...`):

```bash
cd src/eval
./scripts/download_ciqi_vqa.sh

# Explicit repository, local destination, and immutable revision are also accepted.
./scripts/download_ciqi_vqa.sh \
  SII-Monument-Valley/CiQi-VQA \
  /path/to/data/hf/ciqi-vqa \
  <commit-sha>
```

The `hf` CLI reads normal Hugging Face authentication, including `HF_TOKEN`,
when the repository requires it. Tokens are never command-line arguments. To
reuse an existing download:

```bash
export CIQI_VQA_ROOT=/path/to/data/hf/ciqi-vqa
uv run ciqi-eval dataset validate --root "$CIQI_VQA_ROOT" --split test
```

`examples/ciqi-vqa/*.yaml` connects `datasets.test.path` to
`${CIQI_VQA_ROOT}/test.jsonl` and resolves every `images` entry relative to that
root. No train split is configured or evaluated.

## Model providers

Set `model.provider` in YAML. All adapters accept the evaluator's same
OpenAI-shaped internal multimodal messages and tool schemas:

| Provider value | Native API | Typical environment |
| --- | --- | --- |
| `openai` | `/chat/completions` (also OpenAI-compatible gateways) | base URL, API key, model name |
| `anthropic` or `claude` | Anthropic `/v1/messages` | `ANTHROPIC_API_KEY`, optional base URL/version |
| `gemini` or `google` | Gemini `models/{model}:generateContent` | `GEMINI_API_KEY`, optional base URL |

Ready-to-edit examples are in `examples/ciqi-vqa/backend.yaml`, `gpt.yaml`,
`claude.yaml`, and `gemini.yaml`. Claude and Gemini examples start with tools
disabled for broad endpoint compatibility; setting `tools.enabled: true` and
adding the same RAG block used by `backend.yaml` enables native function calls.

The dual-model smoke script evaluates the same sample with a CiQi-Agent backend
and a GPT endpoint. Configure both endpoints exclusively through environment
variables; the script does not read project-private environment files or legacy
YAML:

```bash
cd src/eval
export CIQI_VQA_ROOT=/path/to/data/hf/ciqi-vqa
export CIQI_EVAL_BACKEND_BASE_URL=http://127.0.0.1:18901/v1
export CIQI_EVAL_BACKEND_API_KEY=EMPTY
export CIQI_EVAL_BACKEND_MODEL=ciqi-agent
export CIQI_EVAL_GPT_BASE_URL=https://api.openai.com/v1
export CIQI_EVAL_GPT_API_KEY=your-key
export CIQI_EVAL_GPT_MODEL=gpt-5
export CIQI_AGENT_RAG_API_URL=http://127.0.0.1:8001
export CIQI_AGENT_RAG_API_KEY=your-rag-key
./scripts/smoke_models.sh 1

# One sample from only the kiln category for each model:
./scripts/smoke_models.sh 1 kiln
```

It writes separate runs under `outputs/ciqi-vqa-backend` and
`outputs/ciqi-vqa-gpt`. Increase the numeric argument only after the one-sample
smoke test succeeds.

To apply the optional LLM judge to QA results:

```bash
ciqi-eval score \
  --config examples/config.yaml \
  --input outputs/ciqi-example/results-test.jsonl \
  --output outputs/ciqi-example/scored-test.jsonl
```

## Dataset schema

JSON and JSONL are supported.  Every sample is an object:

```json
{
  "id": "sample-001",
  "question": "请选择正确名称。",
  "images": ["images/object.jpg"],
  "options": ["选项 A", "选项 B", "选项 C", "选项 D"],
  "answer": "B",
  "ground_truth": "用于开放问答的参考答案"
}
```

- `images` may be a string or list. Paths are resolved relative to the split's
  `image_root`; missing images fail that sample with an explicit error.
- Selection accepts `options` as a list or as an `A`/`B`/... mapping and uses
  `answer` as the label.
- QA prefers `ground_truth` and falls back to `answer`.
- Remote image URLs are intentionally rejected by the dataset loader so a run
  remains reproducible and does not silently depend on mutable web content.

## Configuration

See [`examples/config.yaml`](examples/config.yaml).  Important properties:

- All relative dataset, image-root, output, and RAG image-root paths are
  resolved relative to the YAML file, not the shell working directory.
- `tools.names` is an allow-list. Unknown names fail validation before a model
  request is made.
- `runtime.max_tool_steps` bounds agent loops and
  `runtime.concurrency` bounds in-flight samples.
- A resolved, credential-redacted YAML snapshot is written with every run.
- Results preserve input order even though samples execute concurrently.
- `model.provider` selects a native provider adapter; provider aliases are
  normalized during validation and unsupported values fail before any API call.

Each split produces `results-<split>.jsonl`; `summary.json` contains split and
weighted overall metrics. Every row includes a version, status, extracted
answer, score, compact model/tool trace, and an image-redacted conversation.

## Extending tools

Create an async handler that accepts `ToolContext` and a JSON argument mapping,
returns `ToolOutput`, and register it with an OpenAI function schema:

```python
from ciqi_eval.tooling import ToolRegistry
from ciqi_eval.types import ToolOutput

registry = ToolRegistry()
registry.register(MY_SCHEMA, my_async_handler)
runner = EvaluationRunner(config, tools=registry)
```

This makes tool behavior injectable in unit tests and avoids dynamic `sys.path`
modification or imports from project-private utility directories.

## Legacy entry points

The historical commands remain available and convert the old flat YAML schema
at runtime:

```bash
python infer_gugong_selection_tools.py --config /path/to/legacy.yaml
python infer_gugong_qa_tools.py --config /path/to/legacy.yaml --score-qa
```

New experiments should use the versioned schema and `ciqi-eval` CLI.

## Release hygiene

The repository source code is released under Apache-2.0. Before publishing
changes, run the test suite and perform a repository-wide secret scan. The
package and example configuration contain no service credentials; deployment
secrets must remain in environment variables or the target platform's secret
manager.
