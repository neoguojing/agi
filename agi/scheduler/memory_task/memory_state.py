from typing import Any, Annotated, Union, Optional
from langgraph.channels import LastValue


from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentState,
    ResponseT,
)

from agi.scheduler.memory_task.memory_models import (
    ProfileMemoryRecord,
    EpisodicMemoryRecord,
    SemanticMemoryRecord
)
def memory_reducer(
    state: Optional[dict[str, Any]],  # 状态现在是 dict
    writes: Union[list[Any], dict[str, Optional[Any]]]
) -> dict[str, Any]:
    """
    A high-performance generic reducer using dictionary state.
    """
    # 1. 直接继承历史状态，无须再 O(N) 遍历重建
    merged: dict[str, Any] = state.copy() if state else {}

    # 2. 应用写入
    if isinstance(writes, list):
        # 如果 LLM 或 Tool 传来的是列表 (Upsert)
        for r in writes:
            if hasattr(r, "dedup_key") and r.dedup_key:
                merged[r.dedup_key] = r
                
    elif isinstance(writes, dict):
        # 如果传来的是字典 (包含 Upsert 和 Deletion)
        for k, r in writes.items():
            if r is None:
                merged.pop(k, None)  # 显式删除
            else:
                real_k = r.dedup_key if hasattr(r, "dedup_key") and r.dedup_key else k
                merged[real_k] = r

    # 直接返回字典，不再强制转换为 list
    return merged

class MemoryState(AgentState[ResponseT]): 
    """State schema for the memory organization middleware."""
    # 从 list 变为 dict[str, RecordType]
    profile_records: Annotated[NotRequired[dict[str, ProfileMemoryRecord]], memory_reducer]
    episodic_records: Annotated[NotRequired[dict[str, EpisodicMemoryRecord]], memory_reducer]  
    semantic_records: Annotated[NotRequired[dict[str, SemanticMemoryRecord]], memory_reducer] 
    organization_reason: Annotated[NotRequired[str], LastValue]
    profile_message_index: Annotated[NotRequired[int], LastValue]
    episodic_message_index: Annotated[NotRequired[int], LastValue]
    semantic_message_index: Annotated[NotRequired[int], LastValue]
