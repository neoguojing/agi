# AGI：面向多模态与检索增强的 FastAPI 智能体工程

一个可落地的 AGI 工程化项目：以 **LangChain + LangGraph + FastAPI** 为核心，提供 OpenAI 兼容接口，并通过可插拔子服务实现文本对话、RAG、网页检索、文生图、语音识别、语音合成与多模态理解。

## ✨ 项目特性

- **OpenAI 兼容接口**：提供 `/v1/chat/completions`，可直接接入现有 OpenAI SDK 与 OpenWebUI。
- **多能力统一编排**：通过图工作流（Graph）与任务工厂（TaskFactory）动态决策，按请求路由到 LLM、RAG、Agent、图像、音频等链路。
- **多模态输入输出**：支持文本、图片、音频、视频输入；支持文本、图片、音频等输出形态。
- **RAG + Web 检索增强**：支持本地知识库检索、网页搜索与内容抽取，提升时效性与可解释性。
- **微服务化模型部署**：文本、图像、语音、多模态、向量服务可拆分独立部署，便于扩容与隔离。

## 🏗️ 架构概览

项目由一个主服务和多个能力子服务组成：

- `agi`（主编排服务，端口 8000）
- `image_gen`（文生图/图生图，端口 8001）
- `tts`（语音合成，端口 8002）
- `whisper`（语音识别，端口 8003）
- `huggingface`（多模态能力，端口 8005）
- `embd`（Embedding / Rerank，端口 8006）
- `yacy`（搜索引擎后端，端口 8090）
- `open-webui`（可视化交互，端口 3000）

> 详细服务编排与端口映射可见 `docker-compose.yaml`。

## 📁 目录结构（核心）

```text
agi/
├── fastapi_agi.py          # 统一 API 入口（OpenAI 兼容）
├── config.py               # 环境变量与运行配置
├── tasks/                  # 任务编排与图工作流（核心）
├── apps/                   # 各能力 FastAPI 子服务封装
├── llms/                   # 各类模型调用封装
└── utils/                  # 搜索/爬虫/NLP/存储等通用能力

tests/                      # 单元与集成测试
requirements/              # 分能力依赖清单
tools/                     # 模型下载、准备脚本
```

## 🚀 快速开始

### 1) 环境准备

- Python 3.10+
- 建议 Linux + NVIDIA GPU（CPU 也可运行部分能力）
- 已部署 Ollama（用于默认文本模型）

安装依赖：

```bash
pip install -r requirements.txt
```

### 2) 启动主服务

```bash
make run
```

默认启动：`http://0.0.0.0:8000`

### 3) 按需启动能力子服务

```bash
make run_image      # 8001
make run_tts        # 8002
make run_whisper    # 8003
make run_hugface    # 8005
make run_embd       # 8006
```

### 4) Docker Compose 一键部署（推荐生产/联调）

```bash
docker compose -f docker-compose.yaml up -d
```

## 🔧 关键配置

所有配置统一在环境变量中定义，重点包括：

- **鉴权与服务地址**：`API_KEY`、`BASE_URL`
- **模型/推理服务地址**：`OLLAMA_API_BASE_URL`、`IMAGE_GEN_BASE_URL`、`TTS_BASE_URL`、`WHISPER_BASE_URL`、`MULTI_MODEL_BASE_URL`、`EMBEDDING_BASE_URL`
- **模型选择**：`OLLAMA_DEFAULT_MODE`、`TEXT_TO_IMAGE_MODEL_NAME`、`MULTI_MODEL_NAME`
- **存储路径**：`CACHE_DIR`、`FILE_STORAGE_URL`
- **RAG 检索相关**：`RAG_EMBEDDING_MODEL`、`RAG_EMBEDDING_MODEL_PATH`

建议从 `docker-compose.env` 开始定制。

## 🧪 测试

```bash
make test
make test_llms
make test_tasks
make test_api
```

## 📚 核心子模块文档

- `agi/tasks/README.md`：任务工厂、Graph 编排、RAG 与 Agent 流程
- `agi/apps/README.md`：各 FastAPI 能力子服务说明
- `agi/llms/README.md`：模型接入层设计与扩展方式
- `agi/utils/README.md`：搜索、爬虫、NLP、文件存储等基础设施

## 🤝 适用场景

- 企业内部知识库问答（RAG）
- AI 助手平台（文本 + 语音 + 图片）
- 多模型路由与 Agent 自动化工作流
- 与 OpenWebUI 集成的私有化智能平台

## 📄 License

Apache License
