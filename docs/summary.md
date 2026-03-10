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