agent_context = {
    # === 核心 ===
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "content": "..."},
    ],

    # === structured output ===
    "structured_response": None,

    # === 中间推理 ===
    "scratchpad": "...",

    # === 工具相关 ===
    "tool_cache": {},

    # === RAG / memory ===
    "retrieved_docs": [],
    "long_term_memory": {},

    # === 控制信息 ===
    "iteration_count": 3,
}