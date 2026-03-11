from typing import Annotated, Any
from typing_extensions import NotRequired
from langchain.agents.middleware import AgentState
from langchain.agents.middleware import before_model, hook_config
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime

def merge_artifacts(old: dict, new: dict) -> dict:
    """Reducer: 实现黑板策略，增量更新工件信息"""
    return {**old, **new}

class DeepOrchestratorState(AgentState):
    # 基础：长记忆和用户画像
    long_term_memory: NotRequired[str]
    # 黑板：存储子任务生成的工件（如图片 URL、代码分析结果）
    blackboard: NotRequired[Annotated[dict[str, Any], merge_artifacts]]
    # 渐进式披露：当前激活的工具集标签
    active_domain: NotRequired[str]



@before_model(state_schema=DeepOrchestratorState)
def progressive_context_injection(state: DeepOrchestratorState, runtime: Runtime):
    """
    1. 注入长记忆
    2. 根据黑板内容动态披露上下文
    """
    updates = []
    
    # 注入黑板中的工件快照，避免 LLM 反复询问已生成的内容
    if state.get("blackboard"):
        artifacts_summary = f"\n[Blackboard Artifacts]: {list(state['blackboard'].keys())}"
        updates.append(artifacts_summary)
        
    if state.get("long_term_memory"):
        updates.append(f"\n[Project Context]: {state['long_term_memory']}")

    if not updates:
        return None

    # 修改 System Prompt
    new_system = SystemMessage(content=state["messages"][0].content + "".join(updates))
    return {"messages": [new_system]}