# 技术手册：对话摘要中间件 (Summarization Middleware)

## 1. 系统概述
对话摘要中间件是针对长对话上下文管理设计的核心组件。其核心逻辑是在模型请求前拦截并检查 Token 使用情况，当历史消息过长时，通过调用 LLM 将旧历史压缩为精炼的总结（Summary），并将原始详细内容持久化到外部存储中，以确保 Agent 在保持长期记忆的同时能够高效运行。

## 2. 技术架构与核心组件

### A. `SummarizationMiddleware` (自动触发)
负责监控对话 Token 消耗情况。它采用“预处理”模式，在请求到达 LLM 前判断是否需要压缩历史。
- **判定逻辑**：基于阈值计算（Token, Message Count, Fraction），并支持基于上一次 AI 响应的 `usage_metadata` 进行动态判定。
- **核心能力**：具备消息分区、摘要生成、结果重构和历史归档功能，且包含防止 ToolMessage 被孤立的保护机制。

### B. `SummarizationToolMiddleware` (工具驱动)
为智能体提供显式的上下文管理手段。它包装了 `compact_conversation` 工具，允许 Agent 根据自身逻辑（例如任务切换）主动触发压缩流程。
- **安全机制**：包含资格检查（Eligibility Gate），防止模型在对话刚开始时就浪费 Token 进行无效压缩。

---

## 3. 数据结构与协议定义

### A. 输入/输出上下文 (Context)
- **`_summarization_event` (PrivateStateAttr)**: 一个持久化的中间状态字典，记录最近一次摘要的元数据。
  - `cutoff_index`: 被压缩掉的消息在原始列表中的截断位置。
  - `summary_message`: 包含总结内容的 `HumanMessage` 对象。
  - `file_path`: 用于存放原始消息存档的虚拟或物理路径（如 `/history/{thread_id}.md`）。

### B. 模型契约 (Models)
- **`ContextSize`**: 定义触发条件的类型映射：`tuple[str, int | float]`（例如 `("tokens", 10000)`）。
- **`SummarizationEvent`**: 标准化的摘要事件协议，用于在多轮对话中跨请求传递压缩信息。

---

## 4. 函数详细职责描述

| 函数名称 | 功能详述 | 调用逻辑 / 处理细节 |
| :--- | :--- | :--- |
| `_should_summarize` | **阈值扫描** | 获取模型 Profile 中的 `max_input_tokens`，对比当前总 Token 数。支持三种触发模式：(1) 消息条数 $\ge$ 阈值；(2) 总 Token 数 $\ge$ 阈值；(3) 基于上一次 AI 响应中报告的实际消耗（reported tokens）是否超过阈值。(注：若使用 `fraction` 模式，模型 Profile 数据缺失将触发 ValueError)。 |
| `_determine_cutoff_index` | **切片边界计算** | 根据 `keep` 配置确定起始偏移量。对于基于 Token 或 Fraction 的限制，采用 **二分查找算法 (Binary Search)** 以精确锁定能保留最大信息的最小消息索引。若模型数据缺失，则降级为基于条数的简单截断。 |
| `_find_safe_cutoff_point` | **AI/Tool 对保护** | **核心安全机制**：为防止摘要截断导致 ToolMessage 失去对应的 AI 请求，若切片点落在 ToolMessage 上，系统会向上回溯查找其关联的 `AIMessage` 并将该对完整保留。如果找不到关联项，则向前推进跳过所有孤立的 ToolResponse。 |
| `_partition_messages` | **物理分块** | 调用内置函数将消息列表划分为 `to_summarize`（待压缩）和 `preserved_messages`（保留）。 |
| `_create_summary` | **摘要生成引擎** | 将 `to_summarize` 消息转换为 buffer 字符串，调用模型推理并获取总结文本。生成的摘要遵循特定结构：SESSION INTENT, SUMMARY, ARTIFACTS, NEXT STEPS。 |
| `_offload_to_backend` | **历史归档** | 获取或创建存储路径；读取现有内容 $\rightarrow$ 追加新章节（带时间戳）$\rightarrow$ 写回后端。这是保障“信息永不丢失”的核心函数。 |
| `_build_new_messages_with_path` | **上下文重构** | 拼接新的对话流：`[系统提示词 + 指引总结的 HumanMessage(含路径) + summary文本 + 保留的历史消息]`。 |
| `_compute_state_cutoff` | **状态对齐运算** | 在多轮压缩循环中，通过计算上一次摘要点与当前切片点的差值，纠正“相对索引”带来的偏差。 |
| `wrap_model_call` | **提示词注入钩子** | 在 System Prompt 中动态插入引导语，告知 Agent 当前存在可用压缩工具，并说明为什么要压缩。 |

---

## 5. 业务调用全流程 (Detailed Flow)

### 场景一：自动摘要触发流 (Automatic Workflow)
1.  **请求拦截** (`wrap_model_call`) $\rightarrow$ 系统计算 Token 总数。
2.  **阈值判定** (`_should_summarize` $\rightarrow$ `True`) $\rightarrow$ 进入压缩逻辑。
3.  **确定切片** (`_determine_cutoff_index`) $\rightarrow$ 获取索引（如 80）。系统使用二分查找确保 Token 配额精准满足，并调用 `_find_safe_cutoff_point` 防止工具消息断裂。
4.  **执行归档** (`_offload_to_backend`) $\rightarrow$ 将消息 [0-79] 追加至数据库记录或文件系统中。
5.  **生成总结** (`_create_summary`) $\rightarrow$ LLM 生成包含意图、历史回顾、产出物和后续步骤的总结。
6.  **上下文重构** (`_build_new_messages_with_path`) $\rightarrow$ 构建包含新总结和最后 20 条消息的新列表。
7.  **更新状态** $\rightarrow$ 将结果写入 `_summarization_event` 并发送给模型请求，完成当前轮次压缩。

### 场景二：Agent 主动工具调用流 (Tool-Invoked Workflow)
1.  **意图识别** $\rightarrow$ Agent 发起 `compact_conversation` 工具调用。
2.  **权限校验** (`_is_eligible_for_compaction`) $\rightarrow$ 检测历史消息是否足够长以支持压缩。
3.  **同步执行流程** $\rightarrow$ 调用上述与“自动触发”一致的切片、归档和生成总结逻辑。
4.  **结果反馈** $\rightarrow$ 返回带总结信息的 `ToolMessage` 给 Agent，Agent 可据此决定下一步动作。

## 6. 设计关键点 (Key Design Decisions)
- **防重复保护**：通过检查消息的 `lc_source="summarization"` 标识，确保摘要消息不会被再次作为“原始输入”去进行二次压缩（避免陷入无限循环）。
- **模型特化适配**：针对 Anthropic 模型采用专门的 Token 计算因子（如 3.3 chars/token），并强制要求模型 Profile 以支持精确的 `fraction` 模式。
- **容错机制**：如果后端归档失败（`offload_to_backend` 返回 `None`），系统依然允许当前轮次的总结流程继续执行，保证用户对话不中断。
- **状态一致性**：通过计算相对偏移量并配合持久化的摘要记录，确保即使在多轮循环压缩中，Agent 始终能获取到最新的“干净”历史。
