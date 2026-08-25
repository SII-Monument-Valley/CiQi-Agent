# 模型部署

本目录定义 CiQi-Agent 模型服务的独立 uv 环境，使用 SGLang 暴露
OpenAI-compatible 多模态对话接口。仓库不包含模型权重。

## 前置条件

- Linux、NVIDIA GPU 和可用的 CUDA 驱动
- Python 3.10
- [uv](https://docs.astral.sh/uv/)
- NUMA runtime library（Debian/Ubuntu 包名为 `libnuma1`）
- 本地 Hugging Face 模型目录，或已经公开/有权限访问的 Hugging Face 模型 ID

Debian/Ubuntu 基础镜像若缺少 NUMA runtime，可先安装：

```bash
apt-get update && apt-get install -y libnuma1
```

安装环境：

```bash
uv sync --project src/model_server
```

公开仓库不提交 `uv.lock`。默认可从 PyPI 安装，也可以通过 `UV_INDEX_URL` 或
`UV_DEFAULT_INDEX` 指定其他公共源。

## 单卡启动

```bash
MODEL_PATH=YOUR_ORG/YOUR_CIQI_MODEL \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/model/serve_sglang.sh
```

也可以使用本地权重：

```bash
MODEL_PATH=/absolute/path/to/huggingface-model \
bash scripts/model/serve_sglang.sh
```

默认只监听 `127.0.0.1:18901`，服务模型名为 `ciqi-agent`。常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `18901` | HTTP 端口 |
| `SERVED_MODEL_NAME` | `ciqi-agent` | API 中使用的模型名 |
| `TP_SIZE` | `1` | tensor parallel GPU 数 |
| `DP_SIZE` | `1` | data parallel 副本数 |
| `DTYPE` | `bfloat16` | 权重加载及计算精度；FP32 checkpoint 会在加载时转换 |
| `CONTEXT_LENGTH` | `32768` | 最大上下文长度 |
| `MEM_FRACTION_STATIC` | 空 | SGLang 静态显存比例；默认交由 SGLang 自动计算 |
| `CHUNKED_PREFILL_SIZE` | 空 | 分块 prefill 大小；默认交由 SGLang自动计算 |
| `MAX_RUNNING_REQUESTS` | 空 | 最大同时运行请求数 |
| `MAX_TOTAL_TOKENS` | 空 | KV cache token 池上限 |
| `DISABLE_CUDA_GRAPH` | `0` | 设为 `1` 可关闭 CUDA graph 以节省显存 |
| `TOOL_CALL_PARSER` | `qwen25` | 工具调用解析器 |
| `API_KEY` | 空 | 可选的 SGLang API key |

## 单卡 RTX 4090 BF16 尝试

公开 checkpoint 在磁盘中是 FP32（约 33.2 GB），但部署时可以转换为 BF16。实测
SGLang 加载后的权重显存为 15.77 GB。下面的低显存配置使用 4K 上下文、8192-token
KV cache 池、单并发、1K 分块 prefill，并关闭 CUDA graph：

```bash
MODEL_PATH=SII-Monument-Valley/CiQi-Agent-7B \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/model/serve_sglang_4090_bf16.sh
```

这组参数以 24 GB 显存预算为目标，不代表最佳吞吐量。确认服务稳定后，可以
依次尝试提高 `CONTEXT_LENGTH`、`MAX_RUNNING_REQUESTS`，最后再设置
`DISABLE_CUDA_GRAPH=0`。如果初始化时仍然显存不足，先确认卡上没有其他进程；再将
`MAX_TOTAL_TOKENS` 降到不低于 `CONTEXT_LENGTH` 的值。若出现 KV cache token 不足
而非 CUDA OOM，则需要提高 `MAX_TOTAL_TOKENS`，这也会增加显存占用。

该配置已在平台提供的 48 GB RTX 4090 上完成服务启动和普通对话，权重占用
15.77 GB，完成对话时整卡占用 17,676 MiB。该内存足迹低于 24 GB，但尚未在标准
24 GB RTX 4090 实卡上直接验证；不同驱动、CUDA、SGLang 版本及图像尺寸会影响
余量。

可用同一配置执行启动与普通对话冒烟测试；测试结束会自动停止服务：

```bash
MODEL_PATH=SII-Monument-Valley/CiQi-Agent-7B \
bash scripts/model/smoke_test_4090_bf16.sh
```

多卡部署示例：

```bash
MODEL_PATH=YOUR_ORG/YOUR_CIQI_MODEL \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
bash scripts/model/serve_sglang.sh
```

如果显存不足，可减小 `CONTEXT_LENGTH` 或 `MEM_FRACTION_STATIC`，或者提高
`TP_SIZE`。实际需要的 GPU 显存取决于权重精度、上下文长度和并发量。

## 验证服务

```bash
curl -fsS http://127.0.0.1:18901/v1/models
python scripts/model/test_chat.py
```

修改了地址、模型名或 key 时：

```bash
MODEL_BASE_URL=http://127.0.0.1:18901/v1 \
SERVED_MODEL_NAME=ciqi-agent \
MODEL_API_KEY=your-key \
python scripts/model/test_chat.py
```

## 公网部署

SGLang 本身不应直接裸露到公网。建议使用 API gateway 或反向代理实现 TLS、账号
绑定、API key、限流、请求体限制、审计和 key 撤销。真实 key 只能存放在部署平台
的 secret manager 或本地 `.env`，不能提交到 GitHub。
