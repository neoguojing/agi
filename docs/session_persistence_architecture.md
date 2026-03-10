# 基于 DeepAgents 的可扩展 Session 管理与持久化方案

## 1. 背景与目标

本文基于 DeepAgents 的 long-term memory 能力（`CompositeBackend + StateBackend + StoreBackend`）设计一套**可扩展、可多租户、可分层持久化**的 Session 管理方案，用于替换“仅线程内有效”的临时状态管理。

目标：

1. 支持**短期会话态**（单线程、低延迟）与**长期记忆态**（跨线程、跨会话）并存。
2. 支持多租户隔离（`user_id` / `assistant_id` / `org_id`）。
3. 支持可演进的 memory 路由规则（不仅 `/memories/`，还包括 `/profiles/`、`/projects/` 等）。
4. 支持生产级观测、治理、清理、迁移。

---

## 2. 核心设计原则

### 2.1 双层记忆模型（Hot + Durable）

- **Hot（短期）**：线程内会话状态、临时草稿、推理中间结果。
  - 后端：`StateBackend`
  - 生命周期：thread 级
- **Durable（长期）**：用户偏好、项目上下文、研究笔记、可复用知识。
  - 后端：`StoreBackend`
  - 生命周期：跨 thread / 跨会话

### 2.2 路径路由优先，业务语义绑定

采用 `CompositeBackend` 的 path routing：

- `/memories/`：长期持久化
- 其他路径：短期

同时建议扩展为多业务路径：

- `/memories/preferences/`
- `/memories/projects/`
- `/memories/research/`
- `/runtime/`（临时）

### 2.3 Session 与 Memory 解耦

- Session 负责**会话生命周期**（thread、checkpoint、resume、stream）
- Memory 负责**文件系统语义与持久化策略**
- 两者通过 `configurable` + backend 路由连接

---

## 3. 总体架构

```text
Client/API
   |
SessionGateway
   |-- 生成/绑定 thread_id
   |-- 注入 user_id/org_id/conversation_id
   v
DeepAgent Runtime (create_deep_agent)
   |-- TodoListMiddleware（任务规划）
   |-- Memory/Skills Middleware
   |-- SubAgentMiddleware
   |-- FilesystemMiddleware
   v
CompositeBackend Router
   |-- /memories/*  -> StoreBackend  -> BaseStore(Postgres/Redis/...)
   |-- others       -> StateBackend  -> thread state
```

---

## 4. 会话模型设计

### 4.1 Session 主键

建议统一使用：

- `tenant_id`（组织级）
- `user_id`（用户级）
- `assistant_id`（Agent 实例）
- `thread_id`（会话线程）
- `conversation_id`（业务会话，可选）

### 4.2 Configurable 规范

每次调用注入：

```python
config = {
  "configurable": {
    "tenant_id": "org_xxx",
    "user_id": "u_xxx",
    "assistant_id": "deepagent_main",
    "conversation_id": "conv_xxx",
    "thread_id": "th_xxx"
  }
}
```

### 4.3 生命周期

1. 创建会话：生成 `thread_id`
2. 处理中：checkpoint 持续写入
3. 中断恢复：从 checkpoint resume
4. 会话结束：thread 可过期；长期 memory 保留

---

## 5. 持久化分层设计

### 5.1 Backend 路由工厂

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={
            "/memories/": StoreBackend(runtime),
        },
    )
```

### 5.2 Store 选型策略

- 开发：`InMemoryStore`
- 生产：`PostgresStore`
- 规模化：按 tenant 分库/分 schema（或逻辑 namespace）

### 5.3 Namespace 规范

建议以 `(assistant_id, "filesystem")` 为基础，叠加租户维度：

- 逻辑命名：`(tenant_id + ":" + assistant_id, "filesystem")`
- 保障跨租户隔离

---

## 6. Memory 文件目录规范（强建议）

### 6.1 用户偏好

- `/memories/preferences/user_profile.txt`
- `/memories/preferences/interaction_style.txt`

### 6.2 项目知识

- `/memories/projects/{project_id}/requirements.md`
- `/memories/projects/{project_id}/decisions.md`

### 6.3 研究任务

- `/memories/research/{topic}/sources.txt`
- `/memories/research/{topic}/notes.md`
- `/memories/research/{topic}/report.md`

### 6.4 运行时临时文件

- `/runtime/{thread_id}/draft.md`
- `/runtime/{thread_id}/tool_output.json`

---

## 7. 与当前 agi/tasks 的集成建议

当前代码已经引入 deepagent orchestration、subagent、tool registry。下一步建议：

1. 在 `build_main_agent` 增加默认 `backend factory`（CompositeBackend）。
2. 在 `TaskFactory.create_task(TASK_DEEPAGENT)` 路径注入 `store/checkpointer`。
3. 在 `AgiGraph._build_config` 增加 `tenant_id/assistant_id`。
4. 新增 SessionGateway（统一 thread_id 与恢复策略）。

建议新增模块：

```text
agi/session/
  gateway.py
  identity.py
  policies.py
  cleanup.py
```

---


## 8. 编码场景上下文共享、压缩与存储（重点补充）

针对编码任务（大量文件、长历史、多轮编辑）建议引入“**三层上下文模型**”：

### 8.1 三层上下文模型

1. **L0 Working Set（当前工作集）**
   - 内容：当前任务直接相关文件片段、最近对话、最近工具输出
   - 存储：`StateBackend`（短期）
   - 目标：低延迟、最小 token 消耗

2. **L1 Session Digest（会话摘要层）**
   - 内容：当前 thread 的阶段性摘要（目标、已改文件、未完成事项、风险）
   - 存储：`/memories/sessions/{thread_id}/digest.md`（可选持久）
   - 目标：跨长会话压缩上下文

3. **L2 Project Memory（项目记忆层）**
   - 内容：架构决策、模块边界、接口约定、代码风格、历史改动索引
   - 存储：`/memories/projects/{project_id}/...`
   - 目标：跨线程共享与复用

### 8.2 压缩触发策略（建议值）

- Token 超阈值触发：上下文使用率 > 70% 触发轻压缩，> 85% 触发强压缩
- 消息数触发：> 40 条消息触发摘要
- 文件片段触发：同一文件片段累计注入 > 3 次时，写入“文件摘要卡片”替代原文

### 8.3 压缩产物规范（建议模板）

每次摘要输出结构固定，便于后续机器可读：

```yaml
objective: 当前用户目标
done:
  - 已完成事项
changed_files:
  - path: agi/tasks/xxx.py
    why: 修改原因
open_todos:
  - 待完成事项
risks:
  - 风险与假设
next_actions:
  - 下一步动作
```

### 8.4 文件级共享索引（编码任务关键）

新增“文件摘要索引”目录（持久层）：

- `/memories/projects/{project_id}/file_index/{module}.md`

单文件摘要建议包含：
- 责任边界（这个文件做什么）
- 关键类型/函数签名
- 与其他模块依赖关系
- 最近变更记录（最多 N 条）

这样在新线程里可先读索引再按需拉源码，避免全量注入。

### 8.5 代码上下文共享协议（SubAgent 间）

主代理与 subagent 共享时，不直接传全量代码，传“上下文包”：

```json
{
  "task": "修复 retriever 并补测试",
  "working_set": ["agi/tasks/retriever.py", "tests/tasks/task_factory_rag_test.py"],
  "file_summaries": ["..."],
  "constraints": ["保持 API 兼容"],
  "acceptance": ["pytest 指定用例通过"]
}
```

### 8.6 写入策略：何时落盘到长期记忆

仅以下内容写入 `/memories/`：
- 稳定偏好（用户/项目约束）
- 架构决策（ADR）
- 可复用实现经验（如“模块迁移 checklist”）

以下内容仅留在短期：
- 中间推理
- 临时调试日志
- 一次性工具输出

### 8.7 推荐中间件组合（编码任务）

- `TodoListMiddleware`：拆解与跟踪任务
- `SummarizationMiddleware`：长上下文自动压缩
- `MemoryMiddleware`：加载项目规则与长期偏好
- `SkillsMiddleware`：注入编码规范、测试规范、提交流程
- `SubAgentMiddleware`：将大任务分派到检索/实现/测试子代理

### 8.8 失败回退机制

当上下文过长导致性能下降：
1. 优先裁剪 tool args 与历史工具输出
2. 使用最新 session digest 替换旧消息块
3. 限制一次仅允许 N 个文件进入 working set（建议 N=5）
4. 触发“二阶段执行”：先分析/计划，再分步执行

---

## 9. 可扩展能力设计

### 9.1 多层路由扩展

`CompositeBackend.routes` 可扩展：

- `/memories/` -> StoreBackend（长期）
- `/cache/` -> 高速 KV backend（可选）
- `/secure/` -> 加密存储 backend（可选）

### 9.2 多租户策略

- 租户限额（memory 文件数量、总大小）
- 访问控制（仅 owner tenant 可读写）
- 审计日志（谁在何时修改了哪些 memories）

### 9.3 数据治理

- TTL：`/runtime/` 自动清理
- 归档：老旧 `/memories/research/*` 转冷存
- 合并：周期性将碎片文件汇总为摘要索引

---

## 10. 观测与运维

### 10.1 指标

- Session：并发线程数、恢复成功率、平均会话时长
- Memory：读写 QPS、文件增长率、路径命中率
- Agent：规划步数（TodoList）、subagent 调用次数、失败率

### 10.2 日志

关键日志字段：

- `tenant_id`, `user_id`, `thread_id`, `assistant_id`
- `path`, `backend_type`, `operation`, `latency_ms`
- `tool_name` / `subagent_name`

### 10.3 告警

- Store 不可用
- route 误配导致 `/memories/` 落到 transient
- 单租户异常写放大

---

## 11. 安全与合规

1. Memory 路径白名单（仅允许 `/memories/*` 下持久化）
2. 敏感信息脱敏（写入前过滤）
3. 按租户加密（KMS）
4. 删除权支持（按 user_id 全量删除 memories）
5. 审计可追溯

---

## 12. 参考实现蓝图（最小可落地）

### 阶段 A（1~2 周）

- 接入 `CompositeBackend` + `StoreBackend`
- 仅启用 `/memories/` 持久化
- 统一配置注入 `tenant_id/user_id/thread_id`

### 阶段 B（2~4 周）

- 上线 SessionGateway
- 增加 memory 目录规范与系统提示约束
- 新增清理策略（runtime TTL）

### 阶段 C（持续）

- 多层路由（cache/secure）
- 观测体系、配额治理、审计闭环

---

## 13. 最佳实践清单

- [ ] 所有长期文件统一 `/memories/` 前缀
- [ ] 会话请求必须带 `thread_id`
- [ ] 生产必须使用持久 Store（非 InMemory）
- [ ] 建立 memory schema 与路径文档
- [ ] 定期清理临时文件与低价值记忆

---

## 14. 结论

基于 DeepAgents 的 `CompositeBackend` 路径路由能力，可以在不改变工具调用体验的前提下，实现“线程内短期态 + 跨线程长期态”的统一 Session 持久化体系。该方案天然兼容当前 `agi/tasks` 的 subagent 与动态 tools/skills 注册机制，且便于后续在多租户、治理与合规方面持续扩展。
