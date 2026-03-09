---
summary: src/agents 运行时的设计与架构指南，包括工具链、模型路由、沙箱安全与可扩展性
read_when:
  - 设计类似 OpenClaw 的智能体运行时
  - 重构 src/agents 模块
  - 新增模型提供商、工具或子智能体编排
title: 智能体系统设计
---

# 智能体系统设计

最后更新：2026-03-09

## 概览

本文解释 `src/agents/` 的结构，以及如何从零设计一个同类系统。

如果你要构建类似 OpenClaw 的智能体运行时，可以把该子系统视为 5 个协作层：

1. **作用域与身份层**：agent ID、session key 解析、workspace 继承。
2. **执行层**：嵌入式 runner、运行生命周期、流式输出与取消。
3. **工具层**：工具注册、策略流水线、schema 归一化、沙箱文件安全。
4. **模型与鉴权层**：provider catalog、模型配置解析、auth profile 选择。
5. **状态与可靠性层**：写锁、持久化修复、重试、超时、追踪。

## 设计目标

一个稳定的 coding-agent 运行时应满足：

- **会话连续性**：保持用户会话与智能体上下文的稳定映射。
- **确定性安全**：在工具调用前强制执行路径限制、工具策略与执行约束。
- **多提供商可移植性**：支持多模型提供商并处理兼容性差异。
- **子智能体编排**：支持嵌套任务，并保证可追踪和可清理。
- **运行弹性**：能够应对重试、重启、局部故障和磁盘陈旧状态。

## src/agents 代码地图

- **作用域与工作区**
  - `src/agents/agent-scope.ts`：默认 agent 解析、按 agent 配置合并、session agent 选择。
  - `src/agents/workspace.ts`：workspace 默认值、引导种子文件、onboarding 状态。
  - `src/agents/spawned-context.ts`：派生运行元数据与 workspace 继承。
- **执行运行时**
  - `src/agents/pi-embedded-runner.ts`：嵌入式运行 API 与生命周期导出。
  - `src/agents/pi-embedded-runner/*`：运行路径、历史限制、lane 路由、system prompt 覆盖、工具拆分。
- **工具与策略**
  - `src/agents/pi-tools.ts`：组装 coding tools、channel tools、OpenClaw tools 与多层包装。
  - `src/agents/tool-policy.ts`、`src/agents/tool-policy-pipeline.ts`：allowlist、owner-only、群组策略合成。
  - `src/agents/tool-catalog.ts`：工具元数据归一化与展示。
- **模型与鉴权**
  - `src/agents/models-config.ts`：provider/model 配置持久化与归一化。
  - `src/agents/model-catalog.ts`：模型目录映射与 provider 侧描述。
  - `src/agents/auth-profiles/*`：凭据策略、profile 排序、冷却与健康度启发。
- **子智能体与可靠性**
  - `src/agents/subagent-registry.ts`：子智能体运行跟踪、通告重试退避、孤儿运行回收。
  - `src/agents/session-write-lock.ts`、`src/agents/session-file-repair.ts`：会话文件一致性与修复。
  - `src/agents/timeout.ts`、`src/agents/trace-base.ts`、`src/agents/usage.ts`：超时策略、追踪与用量统计。

## 运行时架构

### 1) 作用域解析

每次运行前需解析：

- 生效的 `agentId`（显式 > session key > 默认）
- 生效的 workspace 目录
- 来自全局与 agent 层叠加后的模型/工具策略

### 2) 嵌入式运行生命周期

runner 提供：

- 入队运行
- 流式中间事件
- 取消运行
- 等待完成

### 3) 工具构建与防护流水线

1. 组装基础工具（read/edit/exec/process/channel/OpenClaw）。
2. 根据 provider 兼容性修正 schema。
3. 应用策略流水线（owner 规则、allowlist、group/subagent 约束）。
4. 包装执行钩子（before-call、abort signal、workspace root guard）。

### 4) Provider 与模型适配

不同 provider 对工具 schema 与行为约束不一致。OpenClaw 通过 provider-aware 适配层处理：

- schema 清洗
- 原生工具重名冲突规避
- auth profile 选择与回退

### 5) 子智能体编排

子智能体运行由 run ID 跟踪，包含：

- 生命周期事件
- 有界重试退避的通告投递
- 孤儿/陈旧运行回收
- 防止重复终态写入的清理标记

## 关键技术与模式

- TypeScript ESM
- TypeBox schema 策略
- 策略流水线（有序变换）
- workspace-root 路径防护
- 显式沙箱上下文与主机编辑权限
- 有上限的重试与过期窗口
- 会话文件写锁与修复
- 强测试共置（co-located tests）

## 可复用系统蓝图

建议拆分为：

- `agent-scope`
- `run-engine`
- `tool-engine`
- `model-engine`
- `state-engine`
- `subagent-engine`

### 关键契约

- **RunRequest**：session key、agent ID、workspace、model hint、policy context。
- **RunEvent**：start、delta、tool-call、tool-result、warning、end。
- **ToolDescriptor**：name、schema、capability tags、safety class。
- **PolicyDecision**：allow/deny/rewrite/redaction。

### 安全基线

- workspace 路径边界保护
- 按渠道与 agent 的工具 allowlist
- 运行与工具双层 timeout
- 异步副作用有界重试
- 有状态操作的幂等键

## 相关文档

- [Gateway 网关架构](/concepts/architecture)
- [智能体循环](/concepts/agent-loop)
- [会话模型](/concepts/session)
- [工具总览](/tools/index)
