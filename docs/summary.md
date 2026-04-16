用户消息
   │
   ▼
Agent 调用模型
   │
   ▼
SummarizationMiddleware.wrap_model_call()
   │
   │ ① 还原有效消息
   │   (处理历史 summarization event)
   ▼
effective_messages

   │
   │ ② truncate tool args（可选）
   │   截断旧 tool call 参数
   ▼

   │
   │ ③ 计算 token
   │   判断是否达到 trigger
   ▼
should_summarize ?

   ├── 否
   │      │
   │      ▼
   │   正常调用 LLM
   │
   └── 是
          │
          │ ④ 计算 cutoff
          │   决定哪些消息需要总结
          ▼

    ┌───────────────┬───────────────┐
    │summarize部分  │保留最近消息    │
    └───────────────┴───────────────┘

          │
          │ ⑤ 保存原始历史
          │
          ▼
backend.write()

          │
          │ ⑥ LLM生成summary
          │
          ▼
summary_message

          │
          │ ⑦ 构建新上下文
          │
          ▼

[summary_message]
        +
[recent_messages]

          │
          │ ⑧ 更新state
          │
          ▼
_summarization_event


| 参数                       | 类型                  | 作用                   |
| ------------------------ | ------------------- | -------------------- |
| model                    | str / BaseChatModel | 用于生成 summary 的模型     |
| backend                  | BackendProtocol     | 保存历史对话               |
| trigger                  | ContextSize         | 触发 summarization 的阈值 |
| keep                     | ContextSize         | 保留多少最近上下文            |
| token_counter            | function            | token 统计函数           |
| summary_prompt           | str                 | summary 提示词          |
| trim_tokens_to_summarize | int                 | 最多总结多少 token         |
| history_path_prefix      | str                 | 历史文件路径               |
| truncate_args_settings   | dict                | 截断 tool 参数           |

| 类型       | 示例                 | 含义        |
| -------- | ------------------ | --------- |
| tokens   | ("tokens", 120000) | token数    |
| messages | ("messages", 20)   | message数量 |
| fraction | ("fraction", 0.85) | 占模型上下文比例  |


from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)

summ = SummarizationMiddleware(
    model="gpt-4o-mini",
    backend=backend
)

tool_mw = SummarizationToolMiddleware(summ)

agent = create_deep_agent(
    middleware=[summ, tool_mw]
)

request
  ↓
effective_messages（恢复历史压缩状态）
  ↓
truncate tool args
  ↓
token check
  ↓
[no need summarization]
    → direct LLM call
  ↓
[need summarization]
    → cutoff
    → split
    → offload backend
    → summarize
    → build summary message
    → reconstruct messages
    → LLM call
    → return + state update


| 方法                         | 输入                     | 输出                        | 作用                              |
| -------------------------- | ---------------------- | ------------------------- | ------------------------------- |
| `_get_effective_messages`  | ModelRequest           | list[AnyMessage]          | 恢复“摘要后视图”的对话                    |
| `_apply_event_to_messages` | messages, event        | list[AnyMessage]          | 根据 `_summarization_event` 重建上下文 |
| `_determine_cutoff_index`  | messages               | int                       | 决定“从哪里开始压缩历史”                   |
| `_partition_messages`      | messages, cutoff_index | (to_summarize, preserved) | 拆分历史 vs 保留消息                    |
| `_should_summarize`        | messages, tokens       | bool                      | 判断是否触发摘要                        |
| `_get_profile_limits`      | -                      | int | None                | 获取模型上下文窗口限制                     |

| 方法                                 | 输入                              | 输出               | 作用                           |
| ---------------------------------- | ------------------------------- | ---------------- | ---------------------------- |
| `_should_truncate_args`            | messages, total_tokens          | bool             | 是否需要裁剪 tool args             |
| `_determine_truncate_cutoff_index` | messages                        | int              | 哪些旧消息允许被裁剪                   |
| `_truncate_args`                   | messages, system_message, tools | (messages, bool) | 批量裁剪 AIMessage tool_calls 参数 |
| `_truncate_tool_call`              | tool_call dict                  | tool_call dict   | 单个 tool call 的 args 截断       |

| 方法                              | 输入                    | 输出                 | 作用                 |
| ------------------------------- | --------------------- | ------------------ | ------------------ |
| `_create_summary`               | messages_to_summarize | str                | 同步生成摘要             |
| `_acreate_summary`              | messages_to_summarize | str                | 异步生成摘要             |
| `_build_new_messages_with_path` | summary, file_path    | list[HumanMessage] | 构造“summary消息”插入上下文 |

| 方法                         | 输入              | 输出               | 作用                       |
| -------------------------- | --------------- | ---------------- | ------------------------ |
| `_get_effective_messages`  | ModelRequest    | list[AnyMessage] | 基于 event 构造当前输入          |
| `_apply_event_to_messages` | messages, event | list[AnyMessage] | 重建压缩后的消息流                |
| `_compute_state_cutoff`    | event, cutoff   | int              | 转换 state 层 cut-off index |

| 方法                  | 输入 | 输出  | 作用                       |
| ------------------- | -- | --- | ------------------------ |
| `_get_thread_id`    | -  | str | 获取会话 ID（或生成 session_xxx） |
| `_get_history_path` | -  | str | 生成 history 文件路径          |

| 方法                     | 输入                | 输出               | 作用                     |
| ---------------------- | ----------------- | ---------------- | ---------------------- |
| `_get_backend`         | state, runtime    | BackendProtocol  | 解析 backend（支持 factory） |
| `_offload_to_backend`  | backend, messages | file_path | None | 同步写入历史 markdown        |
| `_aoffload_to_backend` | backend, messages | file_path | None | 异步写入历史                 |

