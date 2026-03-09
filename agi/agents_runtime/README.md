# Agents Runtime Package Layout

为提升模块独立性与工程可维护性，`agi/agents_runtime` 已按职责分层：

- `core/`：类型定义、消息协议、技能与工具注册、子代理描述。
- `engines/`：上下文引擎、记忆引擎、多模态路由与知识融合能力。
- `integration/`：deepagents 后端构建与 legacy 任务/知识适配器。
- `orchestration/`：会话、计划（harness）、HITL 与统一运行时服务编排。
- `sandbox/`：容器沙箱运行时与系统工具。

同时保留了原有模块路径（如 `agi.agents_runtime.memory_engine`）的兼容导出层，避免破坏历史调用。
