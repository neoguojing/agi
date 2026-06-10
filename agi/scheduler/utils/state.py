
from langgraph.graph.state import CompiledStateGraph
from typing import Any

async def get_messages(thread_id: str,offset: int,graph: CompiledStateGraph = None,client: Any = None):
    messages = None
    if graph:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        # 假设你的状态定义中有一个名为 "messages" 的 channel
        messages = snapshot.values.get("messages", None)
    
    if client:
        state = await client.threads.get_state(thread_id=thread_id)
        messages = state.values.get("messages", None)

    if messages is None or len(messages) <= offset:
        return None

    return messages[offset:]
