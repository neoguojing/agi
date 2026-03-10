# Mapping Checklist

## Legacy to Unified Concepts

- `create_chat*` -> deepagent main invoke / simple tool call
- `create_rag` -> `rag_specialist` subagent tools (`rag_query`, `rag_upload_documents`)
- `create_websearch` -> `web_research_specialist` tools
- hand-written graph branches -> deepagent intent + subagent delegation

## Acceptance

- Complex capabilities run through `task(subagent=...)`.
- Simple utility capabilities run as direct tools.
- `TaskFactory` no longer imports legacy `agi.tasks.chat.chains` implementations.
