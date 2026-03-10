| 能力       | 说明                   |
| -------- | -------------------- |
| 任务拆分     | 将复杂任务交给子 Agent       |
| 上下文隔离    | 子 Agent 使用独立 context |
| 并行执行     | 多个任务可以同时运行           |
| Token 节省 | 主 Agent 只接收最终结果      |
| 专家 Agent | 不同 Agent 不同能力        |


| 特性                    | 说明                |
| --------------------- | ----------------- |
| Agent Orchestration   | 主 Agent 调度子 Agent |
| Context Isolation     | 子 Agent 独立上下文     |
| Parallel Tasks        | 并行执行              |
| Tool-based delegation | 通过 tool 调用        |
| Ephemeral Agents      | 一次性 Agent         |
| Clean Output          | 只返回最终结果           |


User
  │
  ▼
Main Agent
  │
  │ (LLM decide)
  ▼
Call Tool: task
  │
  ▼
SubAgentMiddleware.task()
  │
  ├─ 1. 校验 subagent_type
  │
  ├─ 2. 构造 SubAgent State
  │       messages=[HumanMessage(description)]
  │
  ├─ 3. 选择 SubAgent Runnable
  │
  ▼
SubAgent.invoke()
  │
  │  (独立 Agent 推理)
  │
  ▼
SubAgent 完成任务
  │
  ▼
返回 state
{
  messages:[...]
}
  │
  ▼
转换为 ToolMessage
  │
  ▼
返回 Main Agent
  │
  ▼
Main Agent 汇总结果
  │
  ▼
Final Response


SubAgentMiddleware.__init__
      │
      ▼
_build_task_tool()
      │
      ▼
StructuredTool(task)
      │
      ▼
Agent Runtime
      │
      ▼
task()
      │
      ├─ _validate_and_prepare_state()
      │
      ├─ subagent.invoke()
      │
      └─ _return_command_with_state_update()



from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent

research_agent = {
    "name": "researcher",
    "description": "Research complex topics",
    "system_prompt": "You are a research expert.",
    "model": "openai:gpt-4o",
    "tools": [search_tool],
}

agent = create_agent(
    "openai:gpt-4o",
    tools=[search_tool],
    middleware=[
        SubAgentMiddleware(
            backend=my_backend,
            subagents=[research_agent]
        )
    ]
)