
from langgraph.graph.state import CompiledStateGraph
from typing import Any

def get_messages(thread_id: str,graph: CompiledStateGraph):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 假设你的状态定义中有一个名为 "messages" 的 channel
    messages = snapshot.values.get("messages", [])

    return messages


def update_state(thread_id: str,graph: CompiledStateGraph,key: str,value: Any):
    config = {"configurable": {"thread_id": thread_id}}

    graph.update_state(config, {key: value})