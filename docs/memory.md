MemoryMiddleware = 文件级 Agent Memory Loader

类 / 方法	类型	参数	返回值	作用说明
MemoryMiddleware	class	backend: BACKEND_TYPES
sources: list[str]	-	Agent Middleware，用于从 AGENTS.md 文件加载记忆并注入到 system prompt
state_schema	class attribute	-	MemoryState	指定 middleware 使用的 Agent State Schema
__init__	constructor	backend：文件读写 backend 或 factory
sources：AGENTS.md 文件路径列表	-	初始化 memory middleware，定义 memory 文件来源
_get_backend	internal method	state
runtime
config	BackendProtocol	解析 backend。如果 backend 是 factory，则通过 ToolRuntime 创建
_format_agent_memory	internal method	contents: dict[path, content]	str	将加载的 memory 格式化为 <agent_memory> prompt
before_agent	hook	state
runtime
config	MemoryStateUpdate | None	Agent 执行前加载 memory（同步）
abefore_agent	async hook	state
runtime
config	MemoryStateUpdate | None	Agent 执行前加载 memory（异步）
modify_request	request modifier	request: ModelRequest	ModelRequest	将 memory 注入 system message
wrap_model_call	model wrapper	request
handler	ModelResponse	在模型调用前注入 memory
awrap_model_call	async wrapper	request
handler	ModelResponse	异步版本模型调用包装
State 结构
类	类型	字段	说明
MemoryState	AgentState	memory_contents: dict[str,str]	存储每个 AGENTS.md 文件加载的内容（private state）
MemoryStateUpdate	TypedDict	memory_contents	middleware 更新 state 时使用
执行流程

MemoryMiddleware 在 Agent 生命周期中的位置：

Agent Start
    │
    │ before_agent / abefore_agent
    │   ↓
    │ 读取 AGENTS.md
    │   ↓
    │ 写入 state.memory_contents
    │
Model Call
    │
    │ wrap_model_call
    │   ↓
    │ modify_request
    │   ↓
    │ 将 memory 注入 system prompt
    │
LLM 执行