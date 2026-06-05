
from langgraph.graph.state import CompiledStateGraph
from typing import Any

def get_messages(thread_id: str,graph: CompiledStateGraph,offset: int):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 假设你的状态定义中有一个名为 "messages" 的 channel
    messages = snapshot.values.get("messages", None)
    if not messages and len(messages) <= offset:
        return None
    return messages[offset:]

def get_memory_index(thread_id: str,graph: CompiledStateGraph):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 假设你的状态定义中有一个名为 "messages" 的 channel
    messages = snapshot.values.get("memory_index", 0)

    return messages


def update_state(thread_id: str,graph: CompiledStateGraph,key: str,value: Any):
    config = {"configurable": {"thread_id": thread_id}}

    graph.update_state(config, {key: value})