# tasks 目录基于 deepagents 的重构方案（修订版）

## 1. 目标与约束

本方案面向 `agi/tasks` 的现有能力重构，覆盖：
- 知识库上传与检索（RAG）
- Web 检索/抓取
- 图片生成/多模态
- 语音处理（ASR/TTS/语音对话）
- 简单工具（如天气查询、股票查询等）

核心约束：**必须依赖并复用 `agi/deepagents` 代码，不再继续扩展“纯自定义状态机 + task 工厂”的耦合方式**。

本次修订明确实施原则：
- **复杂调用（多步、跨模型、跨系统、需规划）统一走 subagent**。
- **简单调用（单步、确定输入输出、低副作用）统一走 tool**。

---

## 2. 现状分析（为什么要重构）

### 2.1 TaskFactory 职责过重

`TaskFactory` 目前同时管理模型实例、embedding、KnowledgeManager、任务注册与实例缓存，导致：
- 生命周期（单例）与业务职责耦合；
- 新能力接入要改动中心工厂；
- 测试替换依赖成本高。

### 2.2 检索链路与编排链路耦合

当前检索能力分散在 `KnowledgeManager`、`create_rag`、`graph` 节点中：
- collection/tenant、文档加载、切分、检索策略混在同一层；
- 路由逻辑和执行逻辑相互依赖，不利于复用。

### 2.3 多模态路由依赖大型手写条件分支

`agi/tasks/graph.py` 中 feature 控制（文本/图片/音频/视频）使用大量条件分支，扩展新 feature 需要频繁改路由代码。

---

## 3. deepagents 可复用能力映射

`agi/deepagents/graph.py` 的 `create_deep_agent` 已经提供了适合替代当前编排层的“基础设施能力”：

- 统一 Agent 构建入口：`create_deep_agent(...)`
- 标准中间件栈：Todo、Filesystem、SubAgent、Summarization、PatchToolCalls
- 可插拔 Skills/Memory/HITL（`interrupt_on`）
- 子代理机制（`SubAgentMiddleware`）
- 统一 backend 抽象（默认 `StateBackend`）

因此，重构重点不是“重写能力”，而是：

1. 以 deepagents 作为总编排层。
2. 复杂域能力交给 subagent。
3. 简单函数能力保留为 tool。
4. 保留现有 RAG / 图像 / 语音算法实现，逐步迁移接口。

---

## 4. 核心设计原则：SubAgent 与 Tool 的边界

### 4.1 判定标准

归类为 **SubAgent**（复杂调用）当满足任一条件：
- 需要多步推理与任务拆解；
- 涉及多模型协同（例如检索 + 重排 + 生成）；
- 依赖外部系统链路（知识库、Web 检索、抓取、索引）；
- 需要上下文记忆、容错重试、策略选择。

归类为 **Tool**（简单调用）当满足全部条件：
- 单步函数式调用；
- 输入输出结构稳定、可强 schema 化；
- 无复杂子流程；
- 失败处理简单（可直接报错或重试一次）。

### 4.2 本项目映射（强约束）

#### 必须用 SubAgent 的能力
- `rag_specialist`：知识库上传、向量检索、重排、文档问答。
- `web_research_specialist`：搜索引擎检索、网页抓取、摘要与证据整理。
- `image_specialist`：图像生成/图像编辑/视觉理解（涉及不同模型路径）。
- `audio_specialist`：ASR/TTS/语音对话编排（含前后处理）。

#### 必须用 Tool 的能力（示例）
- `get_weather(city, date)`（天气查询）
- `get_stock(symbol)`（股票行情）
- `get_time(timezone)`（时区时间）
- `calc(expression)`（数学计算）

> 结论：**“知识库、Web 检索、跨模型调用”统一 subagent；“天气等轻量函数”统一 tool。**

---

## 5. 目标架构（分层）

### 5.1 新分层建议

- **Agent Orchestration 层（新增）**
  - `agi/tasks/orchestration/deepagent_builder.py`
  - 创建主 Deep Agent，注册 subagents 与 simple tools。

- **Complex Domain Subagents 层（新增）**
  - `agi/tasks/subagents/rag_specialist.py`
  - `agi/tasks/subagents/web_research_specialist.py`
  - `agi/tasks/subagents/image_specialist.py`
  - `agi/tasks/subagents/audio_specialist.py`

- **Simple Tools 层（新增）**
  - `agi/tasks/simple_tools.py`
  - 聚合天气、股票、时间、计算等轻量工具。

- **Domain Service 层（保留并轻量改造）**
  - `KnowledgeManager`（继续承担索引与检索）
  - `multi_model_app`（继续承担图像/语音底层能力）

- **API/兼容层（改造）**
  - `TaskFactory.create_task(...)` 逐步改为返回 Deep Agent runnable 或其子能力代理。

### 5.2 主 Agent + 子 Agent 划分

#### 动态注册机制（新增）

- tools 与 skills 统一通过注册中心动态维护，不在构建函数中写死。
- 分为两类：
  - **内建（builtin）**：项目默认可用能力（如天气、股票、时间等 simple tools，以及项目内置 skills）。
  - **外部（external）**：运行时按租户/场景注入的 tools 与 skills（如三方服务插件、业务域技能目录）。
- 构建主代理时可按开关选择是否加载 external 能力（例如生产环境只允许 builtin）。


建议最小 5 个 agent：

1. `coordinator`（主代理）
   - 负责意图理解、任务拆解、委派。
2. `rag_specialist`（子代理）
   - 负责上传、索引、检索、文档问答。
3. `web_research_specialist`（子代理）
   - 负责 web 搜索、抓取、证据聚合。
4. `image_specialist`（子代理）
   - 负责文生图/图生图/视觉问答。
5. `audio_specialist`（子代理）
   - 负责 ASR/TTS/语音对话预处理。

主代理直接挂 simple tools（天气/股票等），无需再派发 subagent。

---

## 6. 关键重构点（按能力）

### 6.1 知识库上传与检索（RAG）——SubAgent

#### 保留
- `KnowledgeManager.store/query_doc/get_retriever` 核心逻辑。

#### 改造
- 将 RAG 操作封装到 `rag_specialist` 内部工具集，不直接暴露给主代理：
  - `rag_upload_documents(collection, files|urls, tenant)`
  - `rag_query(collections, query, tenant, top_k)`
  - `rag_list_collections(tenant)`
- 主代理通过 `task(subagent="rag_specialist", ...)` 触发复杂检索流程。

### 6.2 Web 检索/抓取——SubAgent

- 新增 `web_research_specialist`，统一管理：
  - 搜索词改写
  - 并发检索
  - 网页抓取与清洗
  - 证据片段组织
- 避免主代理直接串联多个 web 工具导致提示词复杂度失控。

### 6.3 图片生成/多模态——SubAgent

- `image_specialist` 负责跨模型路径选择（文生图、图生图、视觉问答）。
- 主代理只做委派与结果整合。

### 6.4 语音处理（ASR/TTS）——SubAgent

- `audio_specialist` 统一管理音频预处理、识别、合成、对话回合。
- `need_speech` 这类状态移至子代理上下文或 tool 参数，不在全局 graph 里硬编码。

### 6.5 简单函数能力——Tool

- 将天气、股票、时间、计算等轻量函数直接注册到主代理 tools。
- 不创建专门 subagent，减少开销。


### 6.6 RAG 模块整合（实现补充）

为提升可复用性，RAG 相关代码建议（并已落地）统一收敛到 `agi/tasks/rag/`：

- `service.py`：统一封装向量库管理、知识上传、检索查询接口（对 `KnowledgeManager` 做领域层适配）。
- `tools.py`：暴露内置 RAG 工具（`rag_upload_documents` / `rag_query` / `rag_list_collections`）。
- `__init__.py`：统一导出 service 与工具清单（`rag_builtin_tools`）。

这样 `rag_specialist` 与 orchestration registry 只依赖 `rag_builtin_tools`，避免在多个模块重复拼装 RAG 逻辑。


---

## 7. 路由重构：从“手写状态机”到“主代理自动路由”

现有 `graph.py` 的 `text_feature_control/image_feature_control/audio_feature_control` 迁移为两阶段：

### 阶段 A（兼容）
- 保留原 graph 入口。
- feature 分支改为调用 `deepagent_router.invoke(...)`。
- 对复杂 feature（RAG/Web/Image/Audio）在 deepagent 内部转为 subagent 委派。
- 对简单 feature（天气等）在 deepagent 内部直接 tool 调用。

### 阶段 B（收敛）
- 移除大部分 feature 控制节点。
- 统一入口：用户输入 -> 主 Deep Agent -> 自动选择 subagent 或 tool -> 输出。

---

## 8. 代码组织建议（可直接落地）

建议新增目录：

```text
agi/tasks/orchestration/
  deepagent_builder.py

agi/tasks/subagents/
  rag_specialist.py
  web_research_specialist.py
  image_specialist.py
  audio_specialist.py

agi/tasks/
  simple_tools.py
```

`deepagent_builder.py` 示例（伪代码）：

```python
from agi.deepagents.graph import create_deep_agent

from agi.tasks.subagents.rag_specialist import rag_subagent
from agi.tasks.subagents.web_research_specialist import web_subagent
from agi.tasks.subagents.image_specialist import image_subagent
from agi.tasks.subagents.audio_specialist import audio_subagent
from agi.tasks.simple_tools import simple_tools


def build_main_agent(model, backend, skills=None, memory=None):
    return create_deep_agent(
        model=model,
        backend=backend,
        subagents=[rag_subagent, web_subagent, image_subagent, audio_subagent],
        tools=simple_tools,  # 天气/股票等轻量函数
        skills=skills,
        memory=memory,
    )
```

---


### 8.1 模块迁移与兼容层（实现补充）

为降低耦合、提升可读性，建议（并已落地）将历史“大而全”模块拆分为领域模块：

- `agi/tasks/runtime/task_factory.py`：任务工厂运行时模块
- `agi/tasks/chat/chains.py`：原 `llm_app` 的对话/文档链路
- `agi/tasks/rag/knowledge.py`：原 `retriever` 的知识检索实现

并保留薄兼容层文件：

- `agi/tasks/task_factory.py`
- `agi/tasks/llm_app.py`
- `agi/tasks/retriever.py`

兼容层仅做 re-export，避免一次性改动导致外部调用中断。

## 9. 分阶段迁移计划（建议 4 个里程碑）

### M1：抽象与兼容
- 新增 deepagent builder。
- 拆分“复杂 subagents”与“简单 tools”。
- 保持现有 API 入参/出参不变。

### M2：替换编排
- `TaskFactory` 默认返回 Deep Agent runnable。
- `graph.py` 改为薄路由，仅保留必要兼容逻辑。

### M3：清理耦合
- 下沉复杂逻辑到对应 subagent，主代理只做路由与汇总。
- 会话态（tenant/thread）统一通过 configurable/context_schema 管理。

### M4：能力增强
- 接入 `skills`（复杂任务规范）与 `memory`（长期偏好/项目规则）。
- 对高风险工具启用 `interrupt_on`。

---

## 10. 测试与验收

### 10.1 回归用例（必须）
- RAG：上传文档 -> 检索命中 -> 文档问答可引用（subagent 路径）。
- Web：搜索 -> 抓取 -> 证据回答（subagent 路径）。
- 图片：text2image + image2image（subagent 路径）。
- 语音：speech2text + text2speech（subagent 路径）。
- 简单工具：天气/股票查询直连 tool（非 subagent）。

### 10.2 架构验收（必须）
- 复杂能力是否都通过 subagent 执行。
- 简单函数是否都通过 tool 执行。
- 是否默认通过 `create_deep_agent` 统一创建运行图。

---


## 11. 编码任务中的上下文压缩与共享设计（补充）

### 11.1 问题

在编码任务中，主问题不是“能否调用工具”，而是：
- 文件多、上下文长，token 快速膨胀；
- 多 subagent 并行后，重复传递相同代码片段；
- 跨线程时缺乏可复用的项目记忆。

### 11.2 解决框架

采用“Working Set + Session Digest + Project Memory”三层：

1. Working Set（短期）
   - 当前任务涉及文件与最近消息，保留在短期状态。
2. Session Digest（中期）
   - 每轮阶段性总结：目标、已改文件、未完成事项、风险。
3. Project Memory（长期）
   - 跨线程共享的架构与规范，写入 `/memories/projects/*`。

### 11.3 SubAgent 共享协议

主代理给子代理不传全量历史，而传结构化 context bundle：

- 任务目标
- working set 文件列表
- 每个文件的摘要卡片
- 约束与验收条件

这样可减少重复 token，并避免不同子代理各自重读全仓。

### 11.4 压缩触发点（建议）

- context 使用率 > 70%：轻压缩（裁剪工具参数/中间输出）
- context 使用率 > 85%：强压缩（只保留 session digest + working set）
- 消息数 > 40：强制生成阶段摘要

### 11.5 持久化策略（与 session 文档对齐）

仅将可复用上下文写入长期记忆：
- 项目约束
- 架构决策
- 文件索引摘要

临时上下文不落长期：
- 中间思维过程
- 一次性命令输出
- 短期调试日志

### 11.6 验收标准补充

- 同一编码任务在 3 个线程中可复用项目记忆，不重复全量解释代码结构。
- 子代理平均上下文 token 占用较基线降低（建议目标 30%+）。
- 在超长会话下仍可保持规划-执行-验证闭环，不因上下文溢出中断。

---

## 12. 风险与对策

- **风险 1：SubAgent 边界定义不清导致重复实现**
  - 对策：先制定“复杂=SubAgent，简单=Tool”判定清单并固化到代码评审模板。

- **风险 2：工具 schema 不稳定导致 agent 调用失败**
  - 对策：每个 tool 使用 Pydantic 输入模型 + 单测。

- **风险 3：历史 API 兼容问题**
  - 对策：保留 `TaskFactory.create_task` 外观，内部切换实现。

---

## 13. 最小实施清单（可执行）

1. 新增 `orchestration/deepagent_builder.py`。
2. 新增 4 个复杂域 subagent（RAG/Web/Image/Audio）。
3. 新增 `simple_tools.py`，首批接入天气/股票/时间/计算。
4. `TaskFactory` 增加 `TASK_DEEPAGENT`，默认走 deepagent。
5. `graph.py` 把 `rag_search/web_search/tts/speech2text/image` 分支切到 subagent 路径。
6. 为“复杂走 subagent、简单走 tool”补充集成测试。

> 以上方案满足你的要求：以 deepagents 为编排核心，复杂能力走 subagent，简单能力走 tool。
