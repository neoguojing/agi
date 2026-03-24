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


from typing import Any, Callable, Awaitable, Annotated, Literal
from typing_extensions import NotRequired, TypedDict
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

# =========================
# 数据结构定义
# =========================

class FileData(TypedDict):
    """文件数据结构"""
    content: list[str]
    created_at: str
    modified_at: str


class FilesystemState(AgentState):
    """Middleware 状态"""
    files: Annotated[NotRequired[dict[str, FileData]], "_file_data_reducer"]


# =========================
# Middleware 定义
# =========================

class FilesystemMiddleware(AgentMiddleware[FilesystemState, Any, Any]):
    """
    文件系统中间件（抽象定义）

    提供能力：
    - 文件读写 / 编辑
    - 文件搜索（glob / grep）
    - 命令执行（可选）
    - 大结果自动落盘（eviction）
    """

    # -------------------------
    # 核心属性
    # -------------------------

    state_schema = FilesystemState

    backend: Any
    tools: list[BaseTool]

    _custom_system_prompt: str | None
    _custom_tool_descriptions: dict[str, str]
    _tool_token_limit_before_evict: int | None
    _max_execute_timeout: int

    # -------------------------
    # 初始化
    # -------------------------

    def __init__(
        self,
        *,
        backend: Any | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        max_execute_timeout: int = 3600,
    ) -> None:
        ...

    # -------------------------
    # Backend 解析
    # -------------------------

    def _get_backend(self, runtime: ToolRuntime) -> Any:
        """获取 backend 实例"""
        ...

    # -------------------------
    # Tool 构造方法
    # -------------------------

    def _create_ls_tool(self) -> BaseTool:
        ...

    def _create_read_file_tool(self) -> BaseTool:
        ...

    def _create_write_file_tool(self) -> BaseTool:
        ...

    def _create_edit_file_tool(self) -> BaseTool:
        ...

    def _create_glob_tool(self) -> BaseTool:
        ...

    def _create_grep_tool(self) -> BaseTool:
        ...

    def _create_execute_tool(self) -> BaseTool:
        ...

    # -------------------------
    # Model Call Hook
    # -------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        修改：
        - system prompt
        - tools（如移除 execute）
        """
        ...

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        ...

    # -------------------------
    # Tool Call Hook
    # -------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        拦截 tool 结果：
        - 判断是否过大
        - 是否需要 eviction
        """
        ...

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        ...

    # -------------------------
    # 大结果处理（核心机制）
    # -------------------------

    def _process_large_message(
        self,
        message: ToolMessage,
        backend: Any,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        ...

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        backend: Any,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        ...

    def _intercept_large_tool_result(
        self,
        tool_result: ToolMessage | Command,
        runtime: ToolRuntime,
    ) -> ToolMessage | Command:
        ...

    async def _aintercept_large_tool_result(
        self,
        tool_result: ToolMessage | Command,
        runtime: ToolRuntime,
    ) -> ToolMessage | Command:
        ...