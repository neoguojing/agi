from __future__ import annotations

from agi.tasks.rag import rag_builtin_tools


rag_subagent = {
    "name": "rag_specialist",
    "description": "Use this for knowledge-base upload/retrieval and document-grounded answers.",
    "system_prompt": "You are a RAG specialist. Manage collections, upload sources, and answer with retrieved evidence.",
    "tools": rag_builtin_tools,
}
