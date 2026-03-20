# DeepAgents Filesystem Backend / Middleware 设计技术文档

## 1. 文档目标

本文档整理 `agi/deepagents/backends/filesystem.py` 与 `agi/deepagents/middleware/filesystem.py` 的设计细节，重点解释：

- `FilesystemBackend` 如何把真实文件系统抽象成统一的 `BackendProtocol`。
- `FilesystemMiddleware` 如何把 backend 能力包装成可供 Agent 调用的工具集。
- 两者如何通过 `ToolMessage`、`Command`、`AgentState`、`CompositeBackend`、`SandboxBackendProtocol` 等组件协同工作。
- 为什么这一套设计既支持“直接操作真实文件系统”，又能兼容 `StateBackend`、`CompositeBackend`、大结果逐出（eviction）以及可选命令执行。

这份文档偏向架构与实现设计，不是简单 API 列表。

---

## 2. 整体架构概览

```text
               ┌──────────────────────────────┐
               │          Agent / LLM         │
               └──────────────┬───────────────┘
                              │
                    wrap_model_call / awrap_model_call
                              │
               ┌──────────────▼───────────────┐
               │     FilesystemMiddleware     │
               │  - tools 注册                │
               │  - prompt 注入               │
               │  - 大结果逐出                │
               │  - backend 能力过滤          │
               └──────────────┬───────────────┘
                              │
                     tool 调用 / ToolRuntime
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌───────────────┐     ┌────────────────┐    ┌──────────────────┐
│FilesystemBackend│     │CompositeBackend│    │StateBackend /    │
│真实文件系统实现 │     │路径路由层       │    │其它 Backend      │
└───────┬────────┘     └────────────────┘    └──────────────────┘
        │
        ▼
真实文件、目录、搜索、批量上传下载、可选命令执行
```

核心分层思路：

1. **Backend 层负责“能力语义”**：列目录、读写文件、搜索、上传下载、执行命令。
2. **Middleware 层负责“Agent 交互语义”**：工具描述、系统提示、参数约束、状态更新、结果压缩与逐出。
3. **State 层负责“可回放的中间结果”**：尤其是 `files` 字段，用于在 LangGraph 状态里保留文件更新。
4. **Composite / Sandbox 等能力通过协议和路由组合，而不是硬编码进工具。**

---

## 3. FilesystemBackend 设计

## 3.1 角色定位

`FilesystemBackend` 是一个直接映射到宿主机文件系统的 backend。它的主要特点是：

- 路径可映射到真实文件系统。
- 内容以文本读写为主。
- 元信息来自真实文件 stat。
- grep / glob / 上传下载 / 可选执行 都统一挂在一个 backend 协议上。
- 设计上**强调能力统一**，而不是强调隔离安全。

这也是它与 `StateBackend`、`StoreBackend`、`SandboxBackend` 的核心差别：

- `FilesystemBackend` 面向本地开发和受控环境；
- 不是天然安全的远程执行后端；
- 更适合与 HITL 或沙箱一起使用。

---

## 3.2 初始化参数的真实含义

### `root_dir`

`root_dir` 并不总是“安全根目录”，而只是路径解析的参考点：

- `virtual_mode=False` 时：
  - 相对路径会落到 `root_dir` 下；
  - 绝对路径仍然可以直接访问；
  - `..` 也可能逃逸出去。
- `virtual_mode=True` 时：
  - 它才扮演“虚拟根目录”的角色；
  - 所有路径都被视为挂在该根目录下的虚拟路径。

### `virtual_mode`

这是整个 backend 最容易误解的参数。

它的本质不是 sandbox，而是**虚拟路径语义（virtual path semantics）**：

- 主要服务于 `CompositeBackend` 这类“上层已经做了路由和路径归一化”的场景；
- 让 routed backend 看到的是稳定的 `/foo/bar.py` 虚拟路径，而不是依赖宿主绝对路径；
- 顺带提供了基础路径越界保护，但不提供进程级隔离。

### `max_file_size_mb`

用于搜索类操作的降级保护：

- 大文件可能不参与 Python fallback 搜索；
- 其目的不是写入限制，而是避免 grep/glob 场景出现性能失控。

---

## 3.3 路径解析：为什么 `_resolve_path()` 是 backend 的核心

`_resolve_path()` 决定了所有文件操作最终落到哪里。

它有两套行为：

### 模式 A：`virtual_mode=True`

- 输入会被规范成虚拟绝对路径：`foo.py` → `/foo.py`
- 明确禁止：
  - `..`
  - `~`
- 然后通过 `self.cwd / vpath.lstrip('/')` 映射到真实目录
- 最后再校验 `relative_to(self.cwd)`，确保仍在根目录内

**意义：**

- 保证 routed backend 的路径语义稳定；
- 防止 Agent 在虚拟路径模式下逃逸；
- 让 `CompositeBackend` 可以安全地剥离路由前缀再下发。

### 模式 B：`virtual_mode=False`

- 绝对路径原样保留；
- 相对路径拼到 cwd；
- 没有额外逃逸限制。

**意义：**

- 最大兼容性；
- 更符合本地 CLI 开发体验；
- 但安全边界几乎完全依赖外部环境。

---

## 3.4 路径输出：`_to_virtual_path()` 的设计价值

如果 `_resolve_path()` 解决“怎么进”，那么 `_to_virtual_path()` 解决“怎么出”。

它把真实文件路径重新映射成 `/` 开头的虚拟路径，用于：

- `ls_info()`
- `glob_info()`
- 可能被 `CompositeBackend` 再次包装的结果路径

这样做的价值是：

1. Agent 看到的路径格式统一。
2. 上层工具不需要知道宿主真实绝对路径。
3. 不同 backend 可以共享一致的路径接口。

---

## 3.5 目录列举：`ls_info()` 的设计特点

`ls_info()` 是一个“信息型接口”，返回 `FileInfo` 风格的结构，而不是字符串。

设计细节：

- **非递归**：只列一层，避免输出爆炸。
- **文件与目录统一建模**：
  - `path`
  - `is_dir`
  - `size`
  - `modified_at`
- **目录以 `/` 结尾**：这是一种轻量但高效的目录标识。
- **尽量容错**：某个子项 stat 失败不会拖垮整个目录结果。
- **最终排序**：按路径排序，保证 deterministic output。

为什么重要：

- 中间件层可以只关心“展示哪些路径”；
- Backend 保持结构化，Middleware 再决定如何截断和格式化。

---

## 3.6 文件读取：`read()` 的设计重点

`read()` 的定位不是“原始 cat”，而是“适合 LLM 消费的文本读取接口”。

关键设计：

1. **按行分页**
   - `offset` + `limit`
   - 天然适配大文件的渐进式阅读

2. **带行号格式化**
   - 通过 `format_content_with_line_numbers()` 输出
   - 对编辑、引用、替换场景极其关键

3. **空文件特殊提醒**
   - `check_empty_content()` 用于返回系统提醒，而不是沉默返回空串

4. **避免 symlink 跳转风险**
   - 使用 `os.O_NOFOLLOW`（平台支持时）

5. **超出 offset 报错而非返回空**
   - 让 Agent 知道分页越界，而不是误判文件为空

这套设计明显是为“代码编辑 Agent”优化的，而不是通用文件浏览器接口。

---

## 3.7 写入与编辑：为什么分成 `write()` 与 `edit()`

### `write()`

`write()` 语义是**创建/覆盖目标文件内容**，但在中间件层通常被约束为“新建文件”。

返回的是 `WriteResult`，而不是简单字符串，原因是：

- backend 需要把“路径、错误、状态更新”一起返回；
- middleware 决定是否把 `files_update` 包成 `Command`；
- 同一个协议可以兼容 `StateBackend` 这类纯状态后端。

### `edit()`

`edit()` 语义是**对现有文件做精确字符串替换**。

设计选择：

- 不让 Agent 直接提交整文件覆盖，降低误改概率；
- 借助 `perform_string_replacement()` 约束替换行为；
- 返回 `EditResult`，包含：
  - 路径
  - 替换次数
  - 错误
  - 状态更新

这体现了一个很明确的产品判断：

> 对 LLM 来说，“基于读后替换”的编辑模型，远比“整文件重写”更稳。

---

## 3.8 搜索：`grep_raw()` 与 `glob_info()` 的职责分离

### `glob_info()`

关注的是**文件名 / 路径模式匹配**：

- 适合找候选文件
- 返回 `FileInfo` 列表
- 常配合 `ls` / `read_file` 使用

### `grep_raw()`

关注的是**文件内容匹配**：

- 搜 literal string，而不是 regex
- 避免 LLM 因 regex 语义出错
- 返回结构化 `GrepMatch`，交给 middleware 再做格式化

职责拆分的好处：

- Backend 保持结构化和可复用；
- Middleware 根据 `output_mode` 决定展示文件名、计数还是内容片段；
- 便于与 `CompositeBackend` 统一聚合。

---

## 3.9 上传 / 下载：面向多模态与二进制内容补齐能力

`upload_files()` / `download_files()` 的存在说明 backend 协议并不只服务“文本”。

这套接口为以下场景提供基础：

- 图片读取
- 二进制文件中转
- 未来的附件同步、工件下载

在 `FilesystemMiddleware` 里，`read_file` 对图片扩展名的特殊处理，就是建立在 `download_files()` 之上的。

---

## 3.10 执行能力：为什么不把 `execute()` 直接做成必选接口

命令执行并不是所有 backend 都应该具备的能力，所以它被单独抽象为 `SandboxBackendProtocol`。

这意味着：

- `FilesystemBackend` 可以有自己的 `execute()` / `aexecute()` 实现；
- 但 middleware 不会假设 backend 一定支持执行；
- 是否向 Agent 暴露 execute tool，要在 middleware 层按能力判断。

这是一个非常典型的**协议分层设计**：

- 文件系统能力 = 基础协议
- 命令执行能力 = 可选扩展协议

---

## 4. FilesystemMiddleware 设计

## 4.1 角色定位

`FilesystemMiddleware` 不直接读写文件；它做的是四件事：

1. 组装工具（`ls` / `read_file` / `write_file` / `edit_file` / `glob` / `grep` / `execute`）
2. 注入系统提示，引导模型正确使用这些工具
3. 把 backend 的结构化返回转换成 Agent 可消费的 `ToolMessage` 或 `Command`
4. 对大结果做逐出（eviction），避免上下文爆炸

换句话说：

- backend 负责“做事”
- middleware 负责“让 LLM 会用、敢用、不会把上下文打爆”

---

## 4.2 状态设计：`FileData`、`_file_data_reducer()`、`FilesystemState`

### `FileData`

这是文件在 Agent state 中的表示：

- `content: list[str]`
- `created_at`
- `modified_at`

注意它不是宿主文件系统的完整镜像，而是**中间件需要保留的最小必要状态**。

### `_file_data_reducer()`

这个 reducer 的关键价值是：**支持删除语义**。

LangGraph 的 state merge 默认是覆盖/合并，但文件系统更新需要额外支持：

- `None` 表示删除某个文件
- 非 `None` 表示更新或新增

所以 reducer 做了两件事：

1. merge 正常更新
2. 把 `None` 当作 tombstone 删除键

这是 middleware 能兼容多 backend 状态更新的基础设施。

### `FilesystemState`

通过：

- `files: Annotated[..., _file_data_reducer]`

把 reducer 绑定到状态字段上，保证工具返回 `Command(update={"files": ...})` 时能正确合并。

---

## 4.3 Tool 描述为什么写得很长

`FilesystemMiddleware` 的 tool description 非常详细，这不是文档冗余，而是设计的一部分。

这些描述在实践中承担了“隐式操作手册”的作用：

- 告诉模型必须先 `ls` / `read_file`
- 告诉模型如何分页
- 告诉模型不要用 shell 的 `cat/find/grep`
- 告诉模型图片如何读取
- 告诉模型 `execute` 的边界和限制

本质上，这些 description 在替代一部分 hard rule，使工具行为“可学会”。

---

## 4.4 工具工厂模式：`_create_*_tool()`

middleware 没有在一个方法里硬编码所有工具，而是拆成多个 `_create_*_tool()`：

- `_create_ls_tool()`
- `_create_read_file_tool()`
- `_create_write_file_tool()`
- `_create_edit_file_tool()`
- `_create_glob_tool()`
- `_create_grep_tool()`
- `_create_execute_tool()`

这种工厂式组织有几个优点：

1. **同步 / 异步实现并列清晰**
2. **每个工具能独立定制 description 和参数注解**
3. **后续可按 backend 能力裁剪工具集合**
4. **更适合单独测试每个工具包装逻辑**

这也是它和很多“把 backend method 直接暴露给 LLM”的简陋方案最大的区别。

---

## 4.5 `_get_backend()`：为什么 middleware 不直接持有一个 backend 实例

middleware 支持两种 backend 来源：

- 直接传入 backend 实例
- 传入 backend factory，运行时再解析

这样做的原因是：

- 某些 backend 依赖 runtime / state / config 才能构建
- `CompositeBackend` / `StateBackend` 组合时，backend 可能跟运行上下文有关
- middleware 要尽量保持可注入和可组合

也就是说，`FilesystemMiddleware` 的设计目标不是“固定绑定一个文件系统”，而是“适配任意符合协议的文件系统能力提供者”。

---

## 4.6 读文件工具为什么最复杂

`read_file` 是所有工具里最复杂的包装之一，因为它同时处理：

1. 路径校验
2. 文本分页
3. 图像读取
4. 空文件提醒
5. 超长结果截断
6. 同步 / 异步两套路径

### 图像分支

如果扩展名是图片：

- 走 `download_files()`
- base64 编码
- 构造 `ToolMessage(content_blocks=[create_image_block(...)])`

这使得同一个 `read_file` 工具可以覆盖文本和图片，不需要额外新增 `read_image`。

### 文本分支

- 调 backend 的 `read()` / `aread()`
- 再按 token 估算进一步截断
- 截断时附带引导文本，提醒模型用分页或格式化工具继续处理

这种“双层控制”非常关键：

- backend 保证按行读取；
- middleware 再保证单次 tool result 不会把上下文撑爆。

---

## 4.7 写入/编辑工具为什么返回 `Command`

`write_file` / `edit_file` 不是简单返回字符串，而是在 backend 提供 `files_update` 时返回：

```python
Command(
    update={
        "files": res.files_update,
        "messages": [ToolMessage(...)],
    }
)
```

这个设计很关键，因为它把两个动作原子化了：

1. 告诉模型“工具执行成功了”
2. 同步把文件状态写回 Agent state

这样带来的好处：

- StateBackend 可以与真实 backend 共享同样的 middleware 包装层
- 后续其它 middleware 可以读取 `state.files`
- 工具结果和状态更新保持一致，不容易出现“消息成功但状态没改”的分裂

---

## 4.8 `glob` / `grep` 包装层的价值

Backend 返回的是结构化信息；middleware 负责把它变成 LLM 更容易消费的形式。

### `glob`

- 主要把 `FileInfo[]` 变成路径列表
- 再统一做 `truncate_if_too_long()`
- 同步版本还加了线程池超时保护

### `grep`

- backend 返回原始命中结果
- middleware 通过 `format_grep_matches()` 根据 `output_mode` 生成不同视图：
  - 文件名
  - 匹配内容
  - 计数
- 最终再统一截断

这说明 middleware 不是“透传层”，而是“面向 LLM 的投影视图层”。

---

## 4.9 execute tool 的设计原则

`execute` 的包装比看上去更谨慎：

1. **先检查 timeout 合法性**
2. **再检查 backend 是否支持执行**
3. **再检查 backend 是否支持 per-command timeout**
4. **最后才实际执行**

命令输出也不是裸返回，而是追加：

- exit code
- succeeded / failed 状态
- truncated 提示

这让 Agent 在多轮推理时能基于结构化的命令结果继续决策。

---

## 4.10 模型调用包装：动态系统提示注入

`wrap_model_call()` / `awrap_model_call()` 做了两个重要动作：

### 1）按 backend 能力动态过滤工具

如果 backend 不支持 `SandboxBackendProtocol`：

- `execute` 会从工具列表中移除
- 同时 prompt 里也不会出现执行工具说明

这避免了“工具列表和实际能力不一致”。

### 2）动态拼装 system prompt

prompt 由两部分组成：

- `FILESYSTEM_SYSTEM_PROMPT`
- 可选的 `EXECUTION_SYSTEM_PROMPT`

然后通过 `append_to_system_message()` 注入。

这种设计让 middleware 能根据运行环境自动改变 Agent 的行为约束，而无需改模型模板。

---

## 4.11 大结果逐出（Eviction）机制

这是 `FilesystemMiddleware` 最有价值的设计之一。

### 设计动机

Agent tool result 很容易过大：

- shell 输出很长
- 搜索结果很多
- 文件内容超长
- multimodal 文本混合内容过多

如果直接把大结果全部塞回对话历史，很快会把上下文打爆。

### 核心做法

当 ToolMessage 文本内容超过阈值时：

1. 提取文本内容
2. 写入 backend 的 `/large_tool_results/{tool_call_id}`
3. 在对话历史里只保留：
   - 文件路径
   - 头尾 preview
   - 原有非文本内容块（如图片）

### 为什么逐出发生在 middleware，而不是 backend

因为“是否超过上下文预算”是 **LLM 交互层问题**，不是 backend 存储问题。

- backend 只负责读写
- middleware 才知道 tool result 要进入上下文

这是职责划分非常清晰的一点。

### 为什么有排除列表 `TOOLS_EXCLUDED_FROM_EVICTION`

并非所有工具都适合逐出：

- `ls` / `glob` / `grep` 已经内建截断
- `read_file` 的问题通常是单行超长，再次逐出未必有帮助
- `edit_file` / `write_file` 结果本来就很小

所以 eviction 是“按工具语义选择性启用”，而不是一刀切。

---

## 4.12 ToolMessage 与 Command 双通道返回模型

`FilesystemMiddleware` 的一个关键实现细节是：

- 小结果可以直接返回 `ToolMessage`
- 带状态更新或逐出结果时返回 `Command`

这说明工具层不是单纯的“字符串 RPC”，而是：

- **消息更新**
- **状态更新**
- **结果重写**

三者的统一调度机制。

这也是它能兼容 LangGraph 状态流的根本原因。

---

## 5. Backend 与 Middleware 的协作边界

可以把二者的职责边界总结成下面这张表：

| 维度 | FilesystemBackend | FilesystemMiddleware |
| --- | --- | --- |
| 路径解析 | 负责 | 不负责，只做路径校验 |
| 实际文件读写 | 负责 | 不负责 |
| 文件搜索 | 负责原始结果 | 负责格式化输出 |
| 图像读取 | 负责下载二进制 | 负责转为 multimodal message |
| 状态更新结构 | 返回 `files_update` | 负责包成 `Command` |
| Prompt 引导 | 不负责 | 负责 |
| 大结果逐出 | 不负责 | 负责 |
| 执行能力探测 | backend 通过协议暴露 | middleware 决定是否向 LLM暴露 |
| 同步/异步 API | backend 提供底层接口 | middleware 提供 tool 包装接口 |

这是一个相当干净的分层。

---

## 6. 这套设计的关键优点

## 6.1 与不同 backend 兼容

虽然名字叫 `FilesystemMiddleware`，但它并不只服务 `FilesystemBackend`：

- `StateBackend`
- `CompositeBackend`
- 支持执行的 sandbox backend
- 未来其它符合协议的 backend

都可以复用同一套工具包装层。

## 6.2 对 LLM 友好

- 工具描述充分
- 输出做了格式化
- 分页策略明确
- 大结果自动逐出
- 图片支持多模态

## 6.3 对工程扩展友好

- tool 工厂模式清晰
- sync/async 对称
- state reducer 可组合
- execute 能力通过协议扩展

## 6.4 对安全与能力做了解耦

- backend 可以强能力但弱隔离
- middleware 负责降低误用概率
- sandbox / HITL 可在体系外层叠加

---

## 7. 设计上的限制与注意点

## 7.1 `virtual_mode` 不是安全沙箱

这是最重要的使用注意事项。

即使启用了 `virtual_mode=True`，它也只是：

- 统一路径语义
- 提供基础路径防逃逸

它**不能**隔离：

- 进程能力
- 网络能力
- 文件描述符复用
- 命令执行副作用

## 7.2 `FilesystemBackend` 仍然更适合受控环境

这套实现默认更偏：

- 本地 CLI
- 开发环境
- 受控 CI

而不适合直接暴露在面向外部用户的 Web API 中。

## 7.3 大结果逐出依赖 backend 可写

Eviction 机制本质上需要 backend 能写文件。如果 backend 的写语义很弱，或者只读，就无法完整发挥作用。

## 7.4 `execute` 是可选能力，不应被默认假设存在

任何依赖 `execute` 的上层逻辑都应该先考虑 backend 能力探测，而不是静态假设工具必然可用。

---

## 8. 对后续扩展的启发

如果后续要设计其它中间件（例如 browser、RAG、artifact、dataset），filesystem 方案给出了几个值得复用的模式：

1. **Backend 负责结构化原始能力，Middleware 负责 LLM 交互语义。**
2. **工具工厂化，避免一个超大函数承载所有逻辑。**
3. **State 更新通过 `Command` 原子提交。**
4. **对大结果统一做逐出，而不是依赖模型自己“少说点”。**
5. **按能力动态暴露工具，不让 prompt 与真实能力错位。**
6. **sync / async 版本保持对称，降低维护成本。**

---

## 9. 总结

`FilesystemBackend` + `FilesystemMiddleware` 这套设计，本质上是在回答一个问题：

> 如何把“真实文件系统能力”安全、稳定、可分页、可组合地暴露给 LLM Agent？

它的答案不是简单封装几个读写 API，而是构建了一整套完整抽象：

- backend 协议统一能力面
- middleware 统一 Agent 交互面
- state reducer 统一状态更新面
- eviction 统一上下文预算控制面
- capability gating 统一运行时能力面

因此，这套实现既是一个文件系统工具集，也是 deepagents 中“如何设计可组合中间件”的代表性样板。
