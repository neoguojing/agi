# Deep Agents 代码接口输入/输出说明（中文）

本文档聚焦“可直接使用”的核心函数与类，按模块列出：

- 输入（参数）
- 输出（返回值）
- 功能说明
- 可用性示例

> 范围说明：优先覆盖包级公共 API（`__init__.py` 导出）以及高频常用入口类。

---

## 1. `deepagents`（SDK）

### 1.1 函数：`create_deep_agent`

**签名（简化）**

```python
create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph
```

| 项 | 说明 |
|---|---|
| 输入 | 模型、工具、中间件、子代理、记忆、后端、中断策略等配置参数。 |
| 输出 | `CompiledStateGraph`（可 `invoke` / `stream` 的 Deep Agent 图）。 |
| 用途 | 一次性组装具备规划、文件、执行、子代理和上下文管理能力的 Agent。 |

**示例**

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "总结当前项目结构"}]})
print(result)
```

### 1.2 类：`FilesystemMiddleware`

**构造输入 / 输出**

- 输入：
  - `backend: BACKEND_TYPES | None = None`
  - `system_prompt: str | None = None`
  - `custom_tool_descriptions: dict[str, str] | None = None`
  - `tool_token_limit_before_evict: int | None = 20000`
  - `max_execute_timeout: int = 3600`
- 输出：`FilesystemMiddleware` 实例

**用途**

- 注入文件相关工具（如 `ls/read_file/write_file/edit_file/glob/grep/execute`）
- 控制工具可见性、输出截断、执行超时上限

### 1.3 类：`MemoryMiddleware`

**构造输入 / 输出**

- 输入：
  - `backend: BACKEND_TYPES`
  - `sources: list[str]`
- 输出：`MemoryMiddleware` 实例

**用途**

- 从记忆文件（常见为 `AGENTS.md`）加载上下文并注入系统提示。

**示例**

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import MemoryMiddleware

mw = MemoryMiddleware(backend=StateBackend, sources=["/memory/AGENTS.md"])
agent = create_deep_agent(middleware=[mw])
print(agent)
```

### 1.4 类：`SubAgentMiddleware`

**构造输入 / 输出（简化）**

- 输入：
  - `subagents: list[SubAgent | CompiledSubAgent]`
  - 以及可选子代理配置参数（如委派描述、模型/工具设置）
- 输出：`SubAgentMiddleware` 实例

**用途**

- 管理 `task` 式委派，将复杂任务拆分为子代理执行。

### 1.5 类型：`SubAgent` / `CompiledSubAgent`

| 类型 | 输入结构 | 输出含义 |
|---|---|---|
| `SubAgent` | `name/description/system_prompt` + 可选 `tools/model/middleware` | 子代理配置对象（未编译） |
| `CompiledSubAgent` | 已编译子图及元信息 | 可直接被主代理调用的子代理对象 |

---

## 2. `deepagents.backends`

### 2.1 类：`StateBackend`

**构造输入 / 输出**

- 输入：`runtime: ToolRuntime`
- 输出：`StateBackend` 实例

**用途**

- 基于运行时状态提供文件存取能力，适合默认/轻量场景。

### 2.2 类：`FilesystemBackend`

**构造输入 / 输出（常用参数）**

- 输入：`root_dir: str | None = None` 等文件系统配置
- 输出：`FilesystemBackend` 实例

**用途**

- 直接基于本地文件系统实现读写、搜索、列目录能力。

### 2.3 类：`StoreBackend`

**构造输入 / 输出**

- 输入：
  - `runtime: ToolRuntime`
  - `namespace: NamespaceFactory | None = None`
- 输出：`StoreBackend` 实例

**用途**

- 将文件语义映射到持久化 store，适合多会话数据持久化。

### 2.4 类：`CompositeBackend`

**构造输入 / 输出（简化）**

- 输入：多个后端实例（按能力组合）
- 输出：`CompositeBackend` 实例

**用途**

- 合并多后端能力（例如文件由一个后端处理、执行由另一个后端处理）。

### 2.5 类：`LocalShellBackend`

**构造输入 / 输出（常用参数）**

- 输入：文件系统根路径、执行相关配置（如默认超时）
- 输出：`LocalShellBackend` 实例

**关键方法输入/输出**

- `execute(command: str, *, timeout: int | None = None) -> ExecuteResponse`
  - 输入：命令字符串、可选超时秒数
  - 输出：`ExecuteResponse`（`output`、`exit_code`、`truncated`）

**示例**

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

agent = create_deep_agent(backend=LocalShellBackend)
print(agent)
```

---

## 3. `deepagents.middleware`

### 3.1 类：`SkillsMiddleware`

**构造输入 / 输出**

- 输入：
  - `backend: BACKEND_TYPES`
  - `sources: list[str]`
- 输出：`SkillsMiddleware` 实例

**用途**

- 从技能目录加载技能文档并注入系统上下文。

### 3.2 类：`SummarizationMiddleware`

**构造输入 / 输出（简化）**

- 输入：模型、后端、摘要阈值等参数
- 输出：`SummarizationMiddleware` 实例

**用途**

- 长对话自动摘要，降低 token 压力并保留关键上下文。

### 3.3 类：`SummarizationToolMiddleware`

**构造输入 / 输出**

- 输入：`summarization: _DeepAgentsSummarizationMiddleware`
- 输出：`SummarizationToolMiddleware` 实例

**用途**

- 为摘要中间件提供工具调用层协同支持。

---

## 4. `deepagents_cli`

### 4.1 函数：`cli_main`

**签名**

```python
cli_main() -> None
```

| 项 | 说明 |
|---|---|
| 输入 | 无直接参数（读取命令行参数与环境配置）。 |
| 输出 | `None` |
| 用途 | CLI 主入口，启动交互式或非交互式会话流程。 |

**示例**

```python
from deepagents_cli import cli_main

cli_main()
```

---

## 5. `deepagents_acp`

> 包未在 `__init__` 显式导出，以下是常用入口。

### 5.1 类：`AgentServerACP`

**构造输入 / 输出**

- 输入：
  - `agent: CompiledStateGraph | Callable[[AgentSessionContext], CompiledStateGraph]`
  - `modes: SessionModeState | None = None`
- 输出：`AgentServerACP` 实例

**用途**

- 将 Deep Agent 封装为 ACP 服务端，供编辑器（如 Zed）连接。

**示例**

```python
import asyncio
from acp import run_agent
from deepagents import create_deep_agent
from deepagents_acp.server import AgentServerACP


async def main() -> None:
    server = AgentServerACP(agent=create_deep_agent())
    await run_agent(server)


asyncio.run(main())
```

---

## 6. `deepagents_harbor`

### 6.1 类：`HarborSandbox`

**构造输入 / 输出**

- 输入：`environment: BaseEnvironment`
- 输出：`HarborSandbox` 实例

**关键方法输入/输出**

- `execute(command: str, *, timeout: int | None = None) -> ExecuteResponse`
  - 输入：命令字符串、可选超时
  - 输出：`ExecuteResponse`

### 6.2 类：`DeepAgentsWrapper`

**构造输入 / 输出（简化）**

- 输入：评测配置参数（模型、环境、任务运行参数）
- 输出：`DeepAgentsWrapper` 实例

**用途**

- 作为 Harbor Agent 适配器，用于在 benchmark 中运行 Deep Agent。

**示例**

```bash
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker
```

---

## 7. Partner 沙箱

### 7.1 类：`DaytonaSandbox`

**构造输入 / 输出**

- 输入：`sandbox: daytona.Sandbox`
- 输出：`DaytonaSandbox` 实例

**关键方法输入/输出**

- `execute(command: str, *, timeout: int | None = None) -> ExecuteResponse`

### 7.2 类：`ModalSandbox`

**构造输入 / 输出**

- 输入：`sandbox: modal.Sandbox`
- 输出：`ModalSandbox` 实例

**关键方法输入/输出**

- `execute(command: str, *, timeout: int | None = None) -> ExecuteResponse`

### 7.3 类：`RunloopSandbox`

**构造输入 / 输出**

- 输入：`devbox: Devbox`
- 输出：`RunloopSandbox` 实例

**关键方法输入/输出**

- `execute(command: str, *, timeout: int | None = None) -> ExecuteResponse`

**统一示例（以 Runloop 为例）**

```python
import os
from runloop_api_client import RunloopSDK
from langchain_runloop import RunloopSandbox

client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])
devbox = client.devbox.create()
try:
    backend = RunloopSandbox(devbox=devbox)
    result = backend.execute("echo hello", timeout=30)
    print(result.output, result.exit_code)
finally:
    devbox.shutdown()
```
