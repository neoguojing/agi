
from langgraph.graph.state import CompiledStateGraph
from typing import Any

def get_messages(thread_id: str,graph: CompiledStateGraph,offset: int):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    # 假设你的状态定义中有一个名为 "messages" 的 channel
    messages = snapshot.values.get("messages", None)
    
    if messages is None or len(messages) <= offset:
        return None
    print(f"***sdadads***{len(messages[offset:])}")
    print(f"***sdadads***{messages[offset:]}")
    return messages[offset:]

def get_memory_index(thread_id: str,graph: CompiledStateGraph):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 假设你的状态定义中有一个名为 "messages" 的 channel
    messages = snapshot.values.get("memory_index", 0)

    return messages

def get_memories(thread_id: str,graph: CompiledStateGraph):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # 假设你的状态定义中有一个名为 "messages" 的 channel
    profile_records = snapshot.values.get("profile_records", {})
    episodic_records = snapshot.values.get("episodic_records", {})
    semantic_records = snapshot.values.get("semantic_records", {})
    return profile_records,episodic_records,semantic_records


def update_state(thread_id: str,graph: CompiledStateGraph,key: str,value: Any):
    config = {"configurable": {"thread_id": thread_id}}

    graph.update_state(config, {key: value})