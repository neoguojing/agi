import logging
import traceback
import uuid
from typing import Any, Dict, Optional, List, cast
from pydantic import BaseModel, Field

from agi.scheduler.memory_task.memory_tools import (
    MEMORY_SYSTEM_PROMPT,
    consolidate_profile_memory,
    consolidate_episodic_memory,
    consolidate_semantic_memory
)
from agi.scheduler.memory_task.memory_state import MemoryManager, MemoryStateKey
from langchain_core.prompts import ChatPromptTemplate
from langgraph_sdk import get_client
# 🔄 核心对齐：引入标准上下文容器与基类
from agi.scheduler import (
    hub,
    ExternalStateBridge,
    runtime_state_bridge
)
from agi.scheduler.task_hub import TaskContext, BaseRuntime
from agi.agent.models import ModelProvider
from agi.config import LANGGRAPH_MAIN_URL,CLOUD_MODE

logger = logging.getLogger("MemoryTask")

# ------------------------------------------------------------------------------
# 📝 静态元数据与核心映射表 (保持不变)
# ------------------------------------------------------------------------------
TASK_INSTRUCTIONS = {
    "profile": "Call 'consolidate_profile_memory' tool to analyze user core profile based on history and existing memory.",
    "episodic": "Call 'consolidate_episodic_memory' tool to structure recent lifecycle events without repeating semantic or profile items.",
    "semantic": "Call 'consolidate_semantic_memory' tool to extract new declarative knowledge facts that do not overlap with profile/episodic data."
}

TOOL_MAP = {
    "consolidate_profile_memory": consolidate_profile_memory,
    "consolidate_episodic_memory": consolidate_episodic_memory,
    "consolidate_semantic_memory": consolidate_semantic_memory,
}


# ------------------------------------------------------------------------------
# 🧱 1. 环境依赖容器定义 (适配显式继承自 BaseRuntime)
# ------------------------------------------------------------------------------
class MemoryTaskRuntime(BaseRuntime):
    """
    重型环境依赖容器。
    静态依赖（LLM、Client）通过构造函数直接注入；
    动态依赖（Graph、Thread_id、User_id）通过 @property 从桥接器实时拉取。
    """
    def __init__(self, state_bridge: ExternalStateBridge):
        super().__init__()
        self.llm = ModelProvider.get_falback_model()              # 🤖 静态单例依赖
        self.client = get_client(url=LANGGRAPH_MAIN_URL)        # 🔌 静态单例依赖
        self._bridge = state_bridge # 🌁 动态中转桥接器

    @property
    def graph(self) -> Any:
        """动态感知外部线程传入的图实例"""
        return self._bridge.get_value("graph")

    @property
    def thread_id(self) -> str:
        """动态感知外部线程传入的 Thread ID"""
        return self._bridge.get_value("thread_id")

    @property
    def user_id(self) -> str:
        """动态感知外部线程传入的 User ID"""
        return self._bridge.get_value("user_id")

memory_runtime = MemoryTaskRuntime(
    state_bridge=runtime_state_bridge
)
# ------------------------------------------------------------------------------
# 📐 2. 声明式参数契约模型 (Pydantic Schemas 保持不变)
# ------------------------------------------------------------------------------
class ProfileMemorySchema(BaseModel):
    activate_message_threshold: int = Field(default=0, description="触发记忆提取的消息数阈值")
    min_confidence: float = Field(default=0.85, description="语义入库要求极高的置信度")
    model_flavor: str = Field(default="claude-3-5-sonnet", description="知识沉淀选择推理能力更强的模型")

class EpisodicMemorySchema(BaseModel):
    activate_message_threshold: int = Field(default=10, description="触发记忆提取的消息数阈值")

class SemanticMemorySchema(BaseModel):
    activate_message_threshold: int = Field(default=10, description="触发记忆提取的消息数阈值")


# ------------------------------------------------------------------------------
# ⛓️ 3. 核心复用逻辑管线 (全面适配 TaskContext)
# ------------------------------------------------------------------------------
def build_memory_prompt(task_type: str, manager: MemoryManager, messages: Any, instruction: str) -> Any:
    """构建全量 JSONL 上下文 Prompt (纯 CPU 密集运算)"""
    template = ChatPromptTemplate([
        ("system", "{system_prompt}"),
        ("system", "--- PROFILE ---\n{profile_memory}"),
        ("system", "--- EPISODIC ---\n{episodic_memory}"),
        ("system", "--- SEMANTIC ---\n{semantic_memory}"),
        ("placeholder", "{conversation}"),
        ("human", "{instruction}"),
    ])
    return template.invoke({
        "system_prompt": MEMORY_SYSTEM_PROMPT,
        "profile_memory": manager.export_full_jsonl(["profile"]) if task_type == "profile" else "None",
        "episodic_memory": manager.export_full_jsonl(["episodic"]) if task_type == "episodic" else "None",
        "semantic_memory": manager.export_full_jsonl(["semantic"]) if task_type == "semantic" else "None",
        "conversation": messages,
        "instruction": instruction,
    })

# 🔄 适配：直接引入 ctx 上下文容器，彻底消除松散入参
async def execute_memory_consolidation_pipeline(
    task_type: str, 
    threshold: int, 
    ctx: TaskContext
):
    """通用异步记忆提取核心驱动管线"""
    # 强类型断言提示（便于 IDE 补全 runtime 内部的独有属性）
    runtime = cast(MemoryTaskRuntime, ctx.runtime)
    
    # 1. 执行期动态实例化管理器（全部从 ctx.runtime 中无缝解包依赖）
    manager = MemoryManager(
        graph=runtime.graph,
        client=runtime.client,
        thread_id=runtime.thread_id
    )
    
    # 2. 🚦 准入控制流守卫
    await manager.refresh()
    offset = manager.get_memory_index(task_type)
    messages = manager.get_messages()
    
    if not messages or len(messages) < threshold:
        logger.info("[%s] ⏳ 任务 [%s:%s] 未达消息触发阈值 (%d/%d)，跳过本次内存提取。", 
                    ctx.trace_id, task_type, ctx.target_id, len(messages) if messages else 0, threshold)
        return

    mem = manager.get_memories()
    logger.info("[%s] 📝 任务 [%s:%s] 准入校验通过 | 准备调用 LLM | profile:%d episodic:%d semantic:%d", 
                ctx.trace_id, task_type, ctx.target_id,
                len(mem.get('profile_records', {})), 
                len(mem.get('episodic_records', {})), 
                len(mem.get('semantic_records', {})))

    # 3. 绑定工具集并执行非阻塞 LLM 调用
    bound_llm = runtime.llm.bind_tools([
        consolidate_profile_memory,
        consolidate_episodic_memory,
        consolidate_semantic_memory
    ])
    
    instruction = TASK_INSTRUCTIONS.get(task_type, "Consolidate memory.")
    prompt = build_memory_prompt(task_type, manager, messages, instruction)
    
    # 异步非阻塞调用大模型
    result = await bound_llm.ainvoke(prompt)
    tool_calls = getattr(result, "tool_calls", [])
    
    # 4. 处理 Tool Calls 工具路由链
    if not tool_calls:
        logger.info("[%s] ℹ️ 任务 [%s:%s] 大模型未建议任何记忆工具调用。", ctx.trace_id, task_type, ctx.target_id)
        return

    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("args")
        func = TOOL_MAP.get(tool_name)
        
        if not func:
            logger.warning("[%s] ⚠️ 任务 [%s] 找不到对应的工具映射: %s", ctx.trace_id, task_type, tool_name)
            continue

        # 工具异步调用执行
        tool_result = await func.ainvoke(args)
        record_key = f"{task_type}_records"
        
        if isinstance(tool_result, dict):
            update_data = tool_result.get(record_key)
            reason = tool_result.get("organization_reason")
            
            # 5. 通过 Manager 原子化异步回写图状态，并推进索引偏移量
            if update_data:
                await manager.update_state(cast(MemoryStateKey, record_key), update_data)
                if reason:
                    await manager.update_state("organization_reason", reason)
                
                # 索引位置向前安全跃迁
                new_offset = offset + len(messages)
                await manager.update_memory_index(task_type, new_offset)
                logger.info("[%s] ✅ 任务 [%s:%s] 状态同步成功，Index 成功推进至 -> %d", 
                            ctx.trace_id, task_type, ctx.target_id, new_offset)


# ------------------------------------------------------------------------------
# 🚀 4. 旁路原子任务外显层 (彻底对齐为标准双参异步纯函数)
# ------------------------------------------------------------------------------

@hub.cron(
    task_type="profile",
    runtime=memory_runtime,  # 外部注入的 MemoryTaskRuntime 单例
    cron_expr="0 3 * * *",   
    target_id="global",
    params={"activate_message_threshold": 0, "min_confidence": 0.85, "model_flavor": "claude-3-5-sonnet"},
    timeout=120.0
)
# 🔄 适配签名：统一为 (ctx, payload)
async def profile_memory_job(ctx: TaskContext, payload: ProfileMemorySchema):
    """画像记忆整理任务"""
    await execute_memory_consolidation_pipeline(
        task_type="profile",
        threshold=payload.activate_message_threshold,
        ctx=ctx
    )


@hub.cron(
    task_type="episodic",
    runtime=memory_runtime,
    cron_expr="*/10 * * * *",  
    target_id="global",
    params={"activate_message_threshold": 10},
    timeout=60.0
)
# 🔄 适配签名：统一为 (ctx, payload)
async def episodic_memory_job(ctx: TaskContext, payload: EpisodicMemorySchema):
    """情节/事件记忆提取任务"""
    await execute_memory_consolidation_pipeline(
        task_type="episodic",
        threshold=payload.activate_message_threshold,
        ctx=ctx
    )


@hub.cron(
    task_type="semantic",
    runtime=memory_runtime,
    cron_expr="0 * * * *",  
    target_id="global",
    params={"activate_message_threshold": 10},
    timeout=180.0
)
# 🔄 适配签名：统一为 (ctx, payload)
async def semantic_memory_job(ctx: TaskContext, payload: SemanticMemorySchema):
    """语义知识图谱沉淀任务"""
    await execute_memory_consolidation_pipeline(
        task_type="semantic",
        threshold=payload.activate_message_threshold,
        ctx=ctx
    )