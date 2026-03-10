FilesystemMiddleware 是一个 Agent 中间件，用于把文件读写、搜索和命令执行等能力封装为工具，并统一管理状态、后端路由和结果控制。

Agent.run()
   │
   ▼
Middleware.wrap_model_call()
   │
   │ ① 创建 filesystem tools
   │
   ▼
request.tools += filesystem_tools
   │
   ▼
LLM
   │
   │ ② LLM 选择调用 tool
   ▼
Middleware.wrap_tool_call()
   │
   ▼
backend.read/write/edit
   │
   ▼
ToolMessage → LLM

Agent
  │
  ▼
FilesystemMiddleware
  │
  ├─ Tools
  │   ├ ls
  │   ├ read_file
  │   ├ write_file
  │   ├ edit_file
  │   ├ glob
  │   ├ grep
  │   └ execute
  │
  ├─ Backend
  │   ├ StateBackend
  │   ├ StoreBackend
  │   ├ CompositeBackend
  │   └ SandboxBackend
  │
  ├─ State
  │   └ FilesystemState
  │
  └─ Advanced
      ├ Large Result Eviction
      ├ Dynamic System Prompt
      └ Tool Interception

Tool 名称	功能	关键参数	返回值	备注
ls	列出目录下的文件	path: str 绝对路径	List[str] 文件路径列表	返回结果可能被截断
read_file	读取文件内容	file_path: str 文件路径
offset: int 起始行(默认0)
limit: int 最大行数(默认100)	string（文本）或 ToolMessage（图片）	支持 .png/.jpg/.jpeg/.gif/.webp 图片；大文件建议分页读取
write_file	创建新文件并写入内容	file_path: str 文件路径
content: str 文件内容	"Updated file {path}" 或 Command	用于创建文件，不建议用于修改
edit_file	在文件中进行字符串替换	file_path: str 文件路径
old_string: str 原字符串
new_string: str 新字符串
replace_all: bool 是否全部替换	"Successfully replaced N instance(s)" 或 Command	使用前必须先 read_file
glob	按文件模式搜索文件	pattern: str glob模式
path: str 搜索目录(默认 /)	List[str] 文件路径	支持 *, **, ?
grep	搜索文件内容	pattern: str 搜索文本
path: str 搜索目录
glob: str 文件过滤
output_mode: enum 输出模式	string 搜索结果	output_mode: files_with_matches / content / count
execute	在 sandbox 中执行 shell 命令	command: str shell命令
timeout: int 超时秒	string 命令输出 + exit code	仅当 backend 支持 SandboxBackendProtocol


类 / 类型	类型类别	主要职责	关键字段 / 方法	备注
FilesystemMiddleware	Middleware	为 Agent 提供文件系统和可选的命令执行工具	wrap_model_call()
wrap_tool_call()
awrap_model_call()
awrap_tool_call()	核心入口，中间件负责注册 tools、处理 system prompt、拦截大结果
FilesystemState	AgentState	Agent 的文件系统状态	files: dict[str, FileData]	使用 reducer _file_data_reducer 合并文件更新
FileData	TypedDict	表示文件内容和元信息	content: list[str]
created_at: str
modified_at: str	存储在 Agent state 中
BackendProtocol	Protocol 接口	文件系统 backend 抽象接口	read() write() edit()
ls_info() glob_info()
grep_raw()	所有文件系统 backend 必须实现
SandboxBackendProtocol	Protocol 接口	支持命令执行的 backend	execute()
aexecute()	execute tool 依赖该接口
CompositeBackend	Backend 实现	组合多个 backend	default
routes	可按路径路由到不同 backend
StateBackend	Backend 实现	使用 Agent state 存储文件	read() write() edit()	默认 backend，非持久化
WriteResult	数据结构	write 操作返回结果	path
error
files_update	backend.write 返回
EditResult	数据结构	edit 操作返回结果	path
occurrences
error
files_update	backend.edit 返回
ToolRuntime	LangChain Runtime	Tool 调用运行时上下文	tool_call_id
state	用于获取 backend 和更新 state
ToolMessage	LangChain Message	Tool 调用结果消息	content
tool_call_id
name	Agent 工具输出消息
Command	LangGraph 控制对象	更新 agent state	update={messages, files}	Tool 返回状态更新时使用
ToolCallRequest	Tool 调用请求	表示一次 tool 调用	tool_call
runtime	由 middleware 拦截处理
AgentMiddleware	基类	LangChain Agent middleware 基类	wrap_model_call()
wrap_tool_call()