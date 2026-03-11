from typing import Annotated, Any, Dict
from typing_extensions import NotRequired
from langchain.agents.middleware import AgentState
from langchain.agents.middleware import before_model
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime

def scoped_blackboard_reducer(old: dict, new: dict) -> dict:
    """
    智能合并黑板数据：
    支持覆盖更新，同时保留元数据（如存入时间、操作者 user_id）
    """
    merged = old.copy()
    for k, v in new.items():
        # 如果 v 是字典，尝试记录更新时间戳等元数据
        merged[k] = v 
    return merged

class DeepOrchestratorState(AgentState):
    # --- 身份元数据 (由 Runtime 自动填充) ---
    user_id: NotRequired[str]
    thread_id: NotRequired[str]
    
    # --- 核心黑板 (Thread 级别：即时工件) ---
    # 存储：生成的代码、图片、临时计算结果
    blackboard: Annotated[dict[str, Any], scoped_blackboard_reducer]
    
    # --- 对话级记忆 (Conversation 级别：跨 Thread 共享) ---
    # 存储：当前任务的目标、用户的决策偏好
    conversation_context: NotRequired[dict[str, Any]]


@before_model(state_schema=DeepOrchestratorState)
def blackboard_orchestrator_middleware(state: DeepOrchestratorState, runtime: Runtime):
    """
    基于身份标识重构黑板的呈现方式
    """
    # 1. 提取身份标识
    conf = runtime.config.get("configurable", {})
    u_id = conf.get("user_id")
    t_id = conf.get("thread_id")

    # 2. 构建黑板视图 (Blackboard View)
    # 我们不直接塞入整个 blackboard 对象，而是将其转化为“资产清单”
    view = []
    if state.get("blackboard"):
        artifacts = state["blackboard"]
        # 仅将关键工件的摘要披露给 LLM，节省 Token
        summary = ", ".join([f"{k}({type(v).__name__})" for k, v in artifacts.items()])
        view.append(f"\n[Active Thread Artifacts]: {summary}")

    # 3. 注入跨对话的持久化记忆 (从外部 Store 加载)
    # 这里的 store 是 builder 传入的持久化层
    if hasattr(runtime, "store") and u_id:
        # 模拟：获取该用户的全局偏好
        user_pref = runtime.store.get(("preferences", u_id))
        if user_pref:
            view.append(f"\n[User Style Preferences]: {user_pref}")

    # 4. 合并并重写 System Message
    original_sys = state["messages"][0].content
    enriched_content = f"{original_sys}\n--- IDENTITY: {u_id} | THREAD: {t_id} ---\n" + "".join(view)
    
    return {"messages": [SystemMessage(content=enriched_content)]}