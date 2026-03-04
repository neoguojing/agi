# agi.tasks：任务编排与智能体核心

`agi.tasks` 是项目的“大脑层”，负责把用户请求转换为可执行的智能流程。

## 模块职责

- **任务注册与实例管理**：通过 `TaskFactory` 统一创建/缓存不同任务实例。
- **Graph 状态流转**：通过 `AgiGraph` 将意图识别、工具调用、RAG、生成等步骤串联成有状态流程。
- **链路构建**：封装 LLM 对话、带历史对话、文档问答、Web 检索、多模态、TTS/STT 等链。
- **知识库管理**：接入向量库、文档切分、检索与重排。

## 关键文件

- `task_factory.py`：任务类型常量、创建函数、单例缓存与依赖注入入口。
- `graph.py`：多步骤工作流编排与节点路由逻辑。
- `llm_app.py`：纯文本对话 / RAG / Web 检索相关链。
- `multi_model_app.py`：图像、语音、多模态任务链。
- `agent.py`：ReAct Agent 与工具调用逻辑。
- `retriever.py`、`db_builder.py`、`vectore_store.py`：知识库构建与检索底座。

## 典型执行路径

1. API 层构造 `State`（消息、输入类型、用户、功能标记等）。
2. Graph 根据输入类型与 feature 决定走 LLM / Agent / RAG / 多模态链路。
3. TaskFactory 交付所需 Runnable（或检索器/嵌入器）。
4. 结果统一回传到 API 层，转换为 OpenAI 兼容格式。

## 扩展指南

- 新增任务：
  1) 在 `task_factory.py` 定义任务常量；
  2) 实现创建函数；
  3) 注册到 `task_creators`。
- 新增图节点：在 `graph.py` 中增加节点与路由条件，保持输入输出 `State` 结构一致。
- 新增工具：在 `tools.py` 实现并在 Agent 构建时注册。
