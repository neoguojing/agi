# AGI 项目

## 中文版本

### 项目概述

这是一个基于 LangChain 的 AGI（人工通用智能）项目，旨在通过构建一个高可用的 Agent，支持多个智能能力，包含以下功能：

- **LLM 聊天**：通过大型语言模型（LLM）与用户进行对话。
- **知识库检索**：通过自然语言查询知识库并获取相关信息。
- **Web 查询**：从互联网上获取实时信息，提供最新的查询结果。
- **文生图**：基于文本描述生成图片。
- **图生文**：根据图片内容生成描述性文本。
- **语音问答**：支持语音输入与语音输出，进行问答交互。

项目基于 **LangChain** 框架构建，提供高效的集成和开发支持，结合 **FastAPI** 提供高性能的接口支持。同时，项目兼容 **OpenWebUI**，为用户提供一个便捷的前端交互界面。

### 功能特性

- **高可用的智能 Agent**：通过整合多个智能模块，构建一个高可用的智能 Agent，能够执行多种任务，如对话、查询、生成、语音识别等。
- **LangChain 框架**：利用 LangChain 强大的链式构建能力，简化复杂的 AI 任务，提升开发效率。
- **FastAPI 集成**：使用 FastAPI 提供快速、可靠的 API 接口，确保高并发情况下的稳定性。
- **OpenWebUI 兼容**：与 OpenWebUI 完美集成，为用户提供易于使用的界面。
- **多种智能能力**：包括 LLM 聊天、知识库检索、Web 查询、文生图、图生文和语音问答，覆盖多种应用场景。

### 安装与部署

1. 克隆项目：
   ```bash
   git clone https://github.com/your-username/agi-project.git
   cd agi-project

## 性能参数

GPU显存： 初始：2550MB 峰值：20GB
内存： 初始：7GB 峰值：16GB

## RAG
- history_chain = RunnableLambda(self._enter_history, self._aenter_history).with_config(run_name="load_history")
- history_chain = RunnablePassthrough.assign(**{"chat_history": history_chain}).with_config(run_name="insert_history")
- retrieval_docs = (lambda x: x["input"]) | retriever
- retriever = (lambda x: x["input"]) | retriever  or  prompt | llm | StrOutputParser() | retriever, run_name="chat_retriever_chain"
- context=retrieval_docs.with_config(run_name="retrieve_documents")
- "context": format_docs run_name="format_inputs"
- answer =  "context" | prompt | llm | _output_parser     run_name="stuff_documents_chain"

## TODO
- 知识库支持多租户 DONE
- 探索graph将知识库和检索结果实时返回的场景：1.拆分流程；2.流程可以直接返回 DONE
- 支持多模态图片作为输入的ocr解读 Done
- 上传文件之后，其他的问答军基于该文档问答，会导致问题 DONE
- doc文件提取有问题 test
- 需要agent决策是否检索文档 DONE
- agi自动测试
- 支持使用sd 3.5作为图片生成 DONE
- 支持qwen omini作为多模态基模型 doing
- 支持lang-graph 人工介入确认
- 支持python代码执行器
- 其他常用工具支持
- 流式返回测试 DONE
- graph 节点重试特性

## OPENWEBUI 修改
- 新增: backend/open_webui/routers/agi.py
- 修改: backend/open_webui/utils/models.py
- 修改: backend/open_webui/utils/middleware.py
- 修改: backend/open_webui/main.py
- 修改: backend/open_webui/config.py
- 修改: backend/open_webui/audio.py
- 修改: .env.example