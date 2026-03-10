---
name: deepagents-migration
description: Migrate legacy custom chain/graph task flows to unified deepagents concepts using tools for atomic functions and subagents for multi-step domains.
license: MIT
compatibility: Python 3.10+
allowed-tools: read_file, edit_file, execute
metadata:
  owner: agi
  domain: architecture
---

# DeepAgents Migration Skill

Use this skill when the codebase still contains legacy chain-style orchestration and custom graph routing.

## Goal

Unify runtime architecture around:
- `create_deep_agent` as orchestration entry
- `tool` for atomic function calls
- `subagent` for multi-step domain workflows
- optional `skills` for reusable, policy-driven composite behaviors

## Workflow

1. **Inventory old entry points**
   - Find usages of legacy chain builders and route nodes.
2. **Classify behavior**
   - Atomic deterministic action => tool.
   - Multi-step / cross-system logic => subagent or skill.
3. **Replace dispatch layer**
   - Keep API compatibility, but move implementation to tool/subagent runnables.
4. **Add migration guardrails**
   - Mark legacy wrappers as compatibility-only and block new usage in docs/tests.

## Classification Rules

- Use **tool** when operation is one-step and schema-stable.
- Use **subagent** when operation needs planning, retries, or heterogeneous tools.
- Use **skill** when workflow policy needs reusable procedural instructions across tasks.

## References

- Mapping checklist: `references/mapping-checklist.md`
