下面我把这段 create_deep_agent 代码的核心技术细节、组件作用、参数含义和使用方式做了系统整理，并给出 表格 + 使用示例，方便你快速理解 Deep Agent 的架构。

一、Deep Agent 架构概览

create_deep_agent() 本质是 LangChain + LangGraph 的高级 Agent 构建器。

核心能力：

能力	说明
工具调用	LLM可以调用 Python tool
文件系统	读写文件、grep、glob
任务规划	todo list 管理
子 Agent	可以派发任务给 subagent
记忆	加载 AGENTS.md
技能	加载技能脚本
执行环境	可执行 shell
中断审批	Human-in-the-loop
摘要	长上下文自动压缩

整体结构：

Deep Agent
   │
   ├── TodoListMiddleware
   ├── MemoryMiddleware
   ├── SkillsMiddleware
   ├── FilesystemMiddleware
   ├── SubAgentMiddleware
   │        │
   │        └── General Purpose SubAgent
   │               ├── tools
   │               ├── filesystem
   │               ├── shell execute
   │
   ├── SummarizationMiddleware
   ├── PromptCachingMiddleware
   ├── PatchToolCallsMiddleware
   │
   └── LLM (Claude / GPT)
二、函数参数说明
create_deep_agent 参数表
参数	类型	默认	作用
model	str / BaseChatModel	Claude Sonnet 4.6	LLM 模型
tools	list	None	用户自定义工具
system_prompt	str / SystemMessage	None	自定义系统提示
middleware	list	[]	额外 middleware
subagents	list	None	子 agent
skills	list[str]	None	技能目录
memory	list[str]	None	记忆文件
response_format	ResponseFormat	None	结构化输出
context_schema	type	None	上下文 schema
checkpointer	Checkpointer	None	状态持久化
store	BaseStore	None	持久存储
backend	Backend	StateBackend	执行环境
interrupt_on	dict	None	工具调用中断
debug	bool	False	调试模式
name	str	None	agent 名称
cache	BaseCache	None	缓存
三、Deep Agent 默认工具

DeepAgent 内置工具：

工具	作用
write_todos	管理任务计划
ls	列目录
read_file	读取文件
write_file	写文件
edit_file	编辑文件
glob	文件搜索
grep	文本搜索
execute	执行 shell
task	调用 subagent
四、Middleware 体系

DeepAgent 的核心是 Middleware pipeline。

默认 middleware 顺序
顺序	Middleware	作用
1	TodoListMiddleware	任务规划
2	MemoryMiddleware	加载长期记忆
3	SkillsMiddleware	加载技能
4	FilesystemMiddleware	文件操作
5	SubAgentMiddleware	子 agent
6	SummarizationMiddleware	上下文摘要
7	PromptCachingMiddleware	prompt cache
8	PatchToolCallsMiddleware	修复 tool call
9	HumanInTheLoopMiddleware	人工审批
五、SubAgent 结构

DeepAgent 默认创建 General Purpose SubAgent

GENERAL_PURPOSE_SUBAGENT

其结构：

字段	作用
name	agent 名字
description	调用条件
system_prompt	子agent提示
tools	可用工具
model	子模型
middleware	子agent中间件

调用方式：

task({
  "subagent": "xxx",
  "task": "..."
})
六、Backend 体系

Backend 控制：

文件系统

shell

sandbox

默认：

StateBackend

Backend 类型：

Backend	能力
StateBackend	内存文件系统
FilesystemBackend	真实磁盘
SandboxBackend	支持 shell
七、Memory 机制

Memory 通过加载 AGENTS.md。

例如：

memory=[
"/memory/AGENTS.md"
]

内容会被注入到：

system prompt

作用：

长期上下文

用户偏好

项目规范

八、Skills 机制

skills 是 agent 技能脚本

例如：

skills=[
"/skills/user",
"/skills/project"
]

技能内容：

skills/
   build.md
   test.md
   deploy.md

Agent 可以自动调用技能。

九、Human-in-the-loop

允许人工审批工具调用。

示例：

interrupt_on={
"edit_file": True,
"execute": True
}

执行流程：

agent -> tool call -> pause -> human approve
十、模型解析

resolve_model()

支持：

格式	示例
openai	openai:gpt-5
anthropic	anthropic:claude-sonnet
实例	ChatAnthropic()

OpenAI 默认：

Responses API
十一、完整使用示例
示例1：最简单 Deep Agent
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-5"
)

result = agent.invoke({
    "messages":[
        ("user","list files")
    ]
})

print(result)
示例2：带工具
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def add(a:int,b:int)->int:
    return a+b

agent = create_deep_agent(
    model="openai:gpt-5",
    tools=[add]
)
示例3：带子 Agent
subagents=[
{
"name":"researcher",
"description":"research information",
"system_prompt":"You are a research agent"
}
]

agent = create_deep_agent(
    model="openai:gpt-5",
    subagents=subagents
)

调用：

task("research something")
示例4：带 Memory
agent = create_deep_agent(
    memory=["/memory/AGENTS.md"]
)
示例5：带 Skills
agent = create_deep_agent(
    skills=["/skills/project"]
)
十二、Agent 执行流程

完整流程：

User Message
      │
      ▼
DeepAgent
      │
      ├─ Todo planning
      │
      ├─ read memory
      │
      ├─ load skills
      │
      ├─ filesystem ops
      │
      ├─ call subagent
      │
      ├─ tool execution
      │
      └─ summarization
      │
      ▼
LLM Response
十三、关键技术总结
技术	作用
LangGraph	agent state machine
Middleware	能力扩展
SubAgent	任务分解
Filesystem	代码操作
Skills	能力复用
Memory	长期知识
Prompt Caching	降低 token
Summarization	长上下文


一、DeepAgent 总体架构

DeepAgent 实际是 LangGraph StateGraph + 多层 Agent + Middleware Pipeline。

                        ┌────────────────────────────┐
                        │           USER             │
                        └─────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────┐
                       │        Deep Agent       │
                       │  (LangGraph StateGraph) │
                       └─────────────┬───────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
        TodoListMiddleware   MemoryMiddleware   SkillsMiddleware
            (planning)          (AGENTS.md)         (skills)
                 │                   │                   │
                 └──────────────┬────┴────┬──────────────┘
                                ▼         ▼
                        FilesystemMiddleware
                             │
                             ▼
                        SubAgentMiddleware
                             │
             ┌───────────────┼────────────────┐
             ▼                                ▼
    General Purpose Agent               Custom SubAgents
             │                                │
             ▼                                ▼
     Tools / Shell / FS                Domain Tools
             │                                │
             └───────────────┬────────────────┘
                             ▼
                     LLM Reasoning
                     (Claude / GPT)
                             │
                             ▼
                   SummarizationMiddleware
                             │
                             ▼
                   PromptCachingMiddleware
                             │
                             ▼
                      Final Response
二、LangGraph 执行状态机

DeepAgent 底层是 LangGraph DAG。

执行流程：

START
  │
  ▼
LLM Reason
  │
  ├── call tool ──────► Tool Node
  │                         │
  │                         ▼
  │                     Tool Result
  │                         │
  │                         ▼
  │                     LLM Reason
  │
  └── final answer ──► END

完整状态机：

           ┌──────────────┐
           │    START     │
           └──────┬───────┘
                  ▼
         ┌─────────────────┐
         │   LLM Reason    │
         └──────┬──────────┘
                │
     ┌──────────┴───────────┐
     ▼                      ▼
Tool Call             Final Answer
     │                      │
     ▼                      ▼
┌─────────────┐        ┌─────────┐
│  Tool Node  │        │   END   │
└──────┬──────┘        └─────────┘
       │
       ▼
 Tool Result
       │
       ▼
   LLM Reason
三、SubAgent 调度结构

DeepAgent 的核心是 SubAgent 调度系统。

Main Agent
   │
   ▼
SubAgentMiddleware
   │
   ├── General Purpose Agent
   │       │
   │       ├── Filesystem tools
   │       ├── Shell execute
   │       ├── Todo planning
   │       └── Skills
   │
   ├── Research Agent
   │
   ├── Coding Agent
   │
   └── Data Agent

调用方式：

task({
  "subagent": "research_agent",
  "task": "search latest papers about slam"
})
四、Middleware Pipeline

DeepAgent 的能力几乎全部来自 Middleware。

完整 Pipeline：

User Message
      │
      ▼
TodoListMiddleware
      │
      ▼
MemoryMiddleware
      │
      ▼
SkillsMiddleware
      │
      ▼
FilesystemMiddleware
      │
      ▼
SubAgentMiddleware
      │
      ▼
SummarizationMiddleware
      │
      ▼
PromptCachingMiddleware
      │
      ▼
PatchToolCallsMiddleware
      │
      ▼
HumanInTheLoopMiddleware
      │
      ▼
LLM
五、工具调用流程

Agent 调用工具的完整过程：

User Question
     │
     ▼
LLM decides tool
     │
     ▼
Tool Call JSON
     │
     ▼
Tool Execution
     │
     ▼
Tool Output
     │
     ▼
LLM continues reasoning
     │
     ▼
Final Answer

示例：

{
  "tool": "read_file",
  "args": {
    "path": "main.py"
  }
}
六、DeepAgent 默认能力

DeepAgent 默认提供：

能力	来源
任务规划	TodoListMiddleware
文件系统	FilesystemMiddleware
shell执行	Backend
代码搜索	grep / glob
代码编辑	edit_file
技能	SkillsMiddleware
长期记忆	MemoryMiddleware
子agent	SubAgentMiddleware
上下文压缩	SummarizationMiddleware
Prompt缓存	AnthropicPromptCachingMiddleware



静态注入 (System Prompt):

BASE_AGENT_PROMPT: 定义了 Agent 的核心行为准则（简洁、直接、专业客观、任务导向）。

system_prompt: 用户自定义的初始指令，会与 BASE_AGENT_PROMPT 拼接。

动态中间件注入 (Middleware):

TodoListMiddleware: 注入待办事项列表（Tasks/Todos）。

MemoryMiddleware: 读取指定的 AGENTS.md 文件并将其内容注入系统提示词。

SkillsMiddleware: 加载指定的技能文件（Skills）并注入。

FilesystemMiddleware: 注入文件系统相关的工具描述和当前目录状态。

SubAgentMiddleware: 注入子代理（Sub-agents）的描述和调用工具。

SummarizationMiddleware: 注入对话历史的摘要，以节省 Context Window。