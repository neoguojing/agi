# Deep Agents 模块 API 与关键功能详解（中文）

本文面向“集成者与二次开发者”，按模块展开：

- 可使用 API（类/函数）列表
- 每个关键功能的用途（解决什么问题）
- 何时使用（适用场景）
- 可用性示例（最小可运行片段）

> 说明：优先使用包 `__init__.py` 导出的公共 API。个别未导出的项会标记为“常用入口”。

---

## 1. 模块入口总览

| 模块 | 包路径 | 入口 API | 主要作用 |
|---|---|---|---|
| SDK | `deepagents` | `create_deep_agent` | 组装完整 Deep Agent |
| 后端 | `deepagents.backends` | `StateBackend` / `LocalShellBackend` 等 | 决定文件访问与执行环境 |
| 中间件 | `deepagents.middleware` | `MemoryMiddleware` / `SubAgentMiddleware` 等 | 注入能力与上下文策略 |
| CLI | `deepagents_cli` | `cli_main()` | 启动终端交互式 Agent |
| ACP | `deepagents_acp` | `AgentServerACP`（常用入口） | 接入 ACP 编辑器协议 |
| Harbor | `deepagents_harbor` | `DeepAgentsWrapper` / `HarborSandbox` | 评测框架适配 |
| Daytona | `langchain_daytona` | `DaytonaSandbox` | Daytona 远程沙箱执行 |
| Modal | `langchain_modal` | `ModalSandbox` | Modal 远程沙箱执行 |
| Runloop | `langchain_runloop` | `RunloopSandbox` | Runloop 远程沙箱执行 |

---

## 2. `deepagents`（SDK）

### 2.1 可用 API 列表

#### 函数

| API | 功能 |
|---|---|
| `create_deep_agent(...)` | 创建 Deep Agent，内置规划、文件系统、子代理、摘要等默认能力。 |

#### 类/类型

| API | 功能 |
|---|---|
| `FilesystemMiddleware` | 提供文件相关工具能力与调用行为控制。 |
| `MemoryMiddleware` | 加载记忆文件并注入系统上下文。 |
| `SubAgentMiddleware` | 注册与调度子代理。 |
| `SubAgent` | 子代理配置结构（未编译）。 |
| `CompiledSubAgent` | 已编译子代理结构。 |

### 2.2 关键功能展开

#### 功能 A：快速创建可用 Agent（`create_deep_agent`）

- **用途**：在不手动搭建复杂图编排的情况下，快速获得可执行的智能体。
- **适用场景**：原型验证、业务系统快速集成、替换手写 prompt + tool wiring。
- **你能控制的核心参数**：
  - `model`：选择底层模型
  - `tools`：注入业务工具
  - `system_prompt`：补充角色与规则
  - `middleware`：追加中间件
  - `backend`：选择文件/执行后端
  - `subagents`：配置委派子代理

**可用性示例：最小启动**

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "请总结这个代码库的核心模块"}]}
)
print(result)
```

#### 功能 B：注入业务工具（`tools` 参数）

- **用途**：把你的业务能力（查库、调用内部 API、发送通知）交给 Agent 使用。
- **适用场景**：需要 Agent 访问私有业务逻辑而不仅是通用文件/命令能力。

**可用性示例：添加自定义函数工具**

```python
from deepagents import create_deep_agent


def get_release_date() -> str:
    return "2026-03-01"


agent = create_deep_agent(tools=[get_release_date])
print(agent)
```

#### 功能 C：子代理委派（`SubAgentMiddleware` / `subagents`）

- **用途**：把复杂任务拆成多个角色子任务，减少主代理上下文负担。
- **适用场景**：多步骤研发任务（先调研、再实现、最后验收）。

**可用性示例：注册一个子代理**

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "researcher",
    "description": "负责检索与整理背景资料",
    "system_prompt": "你是检索型子代理，输出结构化结论",
}

agent = create_deep_agent(subagents=[research_subagent])
print(agent)
```

---

## 3. `deepagents.backends`（后端）

### 3.1 可用 API 列表

| API | 功能 |
|---|---|
| `BackendProtocol` | 后端能力契约（读写、列表、搜索、执行等）。 |
| `StateBackend` | 基于运行时状态的后端（默认常用）。 |
| `FilesystemBackend` | 直接映射文件系统的后端。 |
| `StoreBackend` | 基于持久化 store 的后端。 |
| `CompositeBackend` | 组合多个后端能力。 |
| `LocalShellBackend` | 支持本地 shell 执行能力。 |
| `DEFAULT_EXECUTE_TIMEOUT` | 执行命令默认超时常量。 |
| `BackendContext` | 后端上下文类型。 |
| `NamespaceFactory` | Store 命名空间工厂类型。 |

### 3.2 关键功能展开

#### 功能 A：切换执行环境（`backend`）

- **用途**：将同一 Agent 逻辑部署在不同文件/执行环境。
- **适用场景**：本地开发、CI 沙箱、远程容器化运行。

**可用性示例：使用本地 shell 后端**

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

agent = create_deep_agent(backend=LocalShellBackend)
print(agent)
```

#### 功能 B：限制与隔离

- **用途**：通过选择后端来控制 Agent 能访问的资源边界。
- **适用场景**：安全要求较高、需最小权限执行的系统。

**可用性示例：使用状态后端（不直接绑定本地 shell）**

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(backend=StateBackend)
print(agent)
```

---

## 4. `deepagents.middleware`（中间件）

### 4.1 可用 API 列表

| API | 功能 |
|---|---|
| `FilesystemMiddleware` | 文件能力工具注入与调用约束。 |
| `MemoryMiddleware` | 记忆加载与系统提示增强。 |
| `SkillsMiddleware` | 技能目录加载与技能指令注入。 |
| `SubAgentMiddleware` | 子代理注册与调度。 |
| `SubAgent` | 子代理定义结构。 |
| `CompiledSubAgent` | 已编译子代理定义结构。 |
| `SummarizationMiddleware` | 长上下文摘要压缩。 |
| `SummarizationToolMiddleware` | 摘要工具协同辅助。 |

### 4.2 关键功能展开

#### 功能 A：长期记忆注入（`MemoryMiddleware`）

- **用途**：把固定规则或用户偏好（通常在 `AGENTS.md`）长期带入系统上下文。
- **适用场景**：团队编码规范、项目约束、长期偏好记忆。

**可用性示例：加载记忆文件**

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import MemoryMiddleware

memory_mw = MemoryMiddleware(
    backend=StateBackend,
    sources=["/memory/AGENTS.md"],
)
agent = create_deep_agent(middleware=[memory_mw])
print(agent)
```

#### 功能 B：技能注入（`SkillsMiddleware`）

- **用途**：将技能文档（如 `SKILL.md`）作为可复用流程指导 Agent。
- **适用场景**：标准化复杂操作（发布流程、排障流程、评审流程）。

**可用性示例：启用技能来源**

```python
from deepagents import create_deep_agent

agent = create_deep_agent(skills=["/skills/user", "/skills/project"])
print(agent)
```

#### 功能 C：长会话压缩（`SummarizationMiddleware`）

- **用途**：当上下文过长时自动摘要，避免 token 爆炸并保持关键信息。
- **适用场景**：多轮长任务、持续对话型开发。

**可用性示例：默认创建即包含摘要能力**

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
print(agent)
```

---

## 5. `deepagents_cli`

### 5.1 可用 API 列表

| API | 功能 |
|---|---|
| `cli_main()` | CLI 主入口函数。 |
| `__version__` | 当前 CLI 版本。 |

### 5.2 关键功能展开

#### 功能 A：交互式终端代理

- **用途**：直接在终端使用完整 Deep Agent 能力。
- **适用场景**：日常编码、调试、文件批处理、命令执行辅助。

**可用性示例：命令行启动**

```bash
deepagents
```

#### 功能 B：脚本方式调用入口

- **用途**：在 Python 脚本中启动 CLI 入口（便于封装自定义启动脚本）。
- **适用场景**：内部工具链集成、统一启动器。

**可用性示例：Python 启动**

```python
from deepagents_cli import cli_main

cli_main()
```

---

## 6. `deepagents_acp`

> `deepagents_acp.__init__` 未导出公共 API，以下是常用入口。

### 6.1 可用 API 列表

| API | 功能 |
|---|---|
| `AgentServerACP` | 将 Deep Agent 封装为 ACP 服务端。 |
| `python -m deepagents_acp` | 通过模块入口启动 ACP 服务。 |

### 6.2 关键功能展开

#### 功能 A：接入 ACP 编辑器生态

- **用途**：使 Deep Agent 能在支持 ACP 的编辑器中直接使用。
- **适用场景**：希望在 IDE/编辑器内原位调用 Agent，而非切换到独立终端。

**可用性示例：用 `AgentServerACP` 包装 Deep Agent**

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

## 7. `deepagents_harbor`

### 7.1 可用 API 列表

| API | 功能 |
|---|---|
| `DeepAgentsWrapper` | Harbor 需要的 Agent 包装器。 |
| `HarborSandbox` | Harbor 任务运行时的沙箱后端适配。 |

### 7.2 关键功能展开

#### 功能 A：基准评测接入

- **用途**：把 Deep Agent 接入 Harbor 基准（如 Terminal Bench）并收集结果。
- **适用场景**：模型/提示词/工具策略变更前后的效果量化对比。

**可用性示例：运行一组 benchmark**

```bash
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker
```

---

## 8. Partner 沙箱模块

## 8.1 `langchain_daytona`

### 可用 API

| API | 功能 |
|---|---|
| `DaytonaSandbox` | 适配 Daytona sandbox，提供命令执行与文件操作能力。 |

### 关键功能与示例

- **用途**：把 Agent 执行迁移到 Daytona 远程环境。
- **适用场景**：需要远程隔离执行和统一环境。

```python
from daytona import Daytona
from langchain_daytona import DaytonaSandbox

backend = DaytonaSandbox(Daytona().create())
print(backend.execute("echo hello").output)
```

## 8.2 `langchain_modal`

### 可用 API

| API | 功能 |
|---|---|
| `ModalSandbox` | 适配 Modal sandbox，提供命令执行与文件操作能力。 |

### 关键功能与示例

- **用途**：把 Agent 执行迁移到 Modal 计算环境。
- **适用场景**：需要弹性算力、云端运行与隔离。

```python
import modal
from langchain_modal import ModalSandbox

backend = ModalSandbox(modal.Sandbox.create(app=modal.App.lookup("your-app")))
print(backend.execute("echo hello").output)
```

## 8.3 `langchain_runloop`

### 可用 API

| API | 功能 |
|---|---|
| `RunloopSandbox` | 适配 Runloop devbox，提供命令执行与文件操作能力。 |

### 关键功能与示例

- **用途**：把 Agent 执行迁移到 Runloop devbox。
- **适用场景**：需要可控生命周期的远程开发盒。

```python
import os
from runloop_api_client import RunloopSDK
from langchain_runloop import RunloopSandbox

client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])
devbox = client.devbox.create()
try:
    backend = RunloopSandbox(devbox=devbox)
    print(backend.execute("echo hello").output)
finally:
    devbox.shutdown()
```

---

## 9. 组合使用建议

### 方案 A：快速上手

- 组合：`deepagents-cli`
- 说明：零代码启动，适合先验证效果。

### 方案 B：业务系统集成

- 组合：`deepagents` + `deepagents.backends` + `deepagents.middleware`
- 说明：可控度最高，适合生产接入。

### 方案 C：编辑器原位使用

- 组合：`deepagents` + `deepagents_acp`
- 说明：让开发者在编辑器中直接使用 Agent。

### 方案 D：评测驱动优化

- 组合：`deepagents_harbor` + Partner 沙箱
- 说明：可在标准基准上持续评估迭代效果。
