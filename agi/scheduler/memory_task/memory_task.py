import abc
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple,cast

from agi.scheduler.base import BaseTaskRuntime, BaseTaskUnit, TaskExecutionResult
from agi.scheduler.utils.state import get_memory_index, get_messages, update_state,get_memories,update_memory_index
from agi.scheduler.memory_task.memory_tools import (
    MEMORY_SYSTEM_PROMPT,
    consolidate_profile_memory,
    consolidate_episodic_memory,
    consolidate_semantic_memory
)
from agi.scheduler.memory_task.memory_state import MemoryManager,MemoryStateKey

from langchain_core.prompts import ChatPromptTemplate


TASK_INSTRUCTIONS = {
    "profile": "Call 'consolidate_profile_memory' tool to analyze user core profile based on history and existing memory.",
    "episodic": "Call 'consolidate_episodic_memory' tool to structure recent lifecycle events without repeating semantic or profile items.",
    "semantic": "Call 'consolidate_semantic_memory' tool to extract new declarative knowledge facts that do not overlap with profile/episodic data."
}

# Mapping of tool names to their corresponding functions
TOOL_MAP = {
    "consolidate_profile_memory": consolidate_profile_memory,
    "consolidate_episodic_memory": consolidate_episodic_memory,
    "consolidate_semantic_memory": consolidate_semantic_memory,
}

logger = logging.getLogger("MemoryTask")


class MemoryTaskRuntime(BaseTaskRuntime):
    """内聚了大模型交互所需要的全部重型环境依赖，让 Store 干净地回归存储本质"""
    def __init__(self, llm: Any, graph: Any, thread_id: str,user_id:str):
        super().__init__()
        self.llm = llm
        self.graph = graph
        self.thread_id = thread_id
        self.user_id = user_id


class BaseMemoryExtractionTask(BaseTaskUnit, abc.ABC):
    """
    生产级记忆提取基类。
    利用 MemoryManager 实现原子化状态管理与全量 JSONL 记忆整理。
    """
    task_type: str = "" 
    
    def __init__(self, runtime: MemoryTaskRuntime, task_id: str, target_id: str, params: dict):
        super().__init__(runtime, task_id, target_id, params)
        # 初始化管理器，接管图状态读写
        self.manager = MemoryManager(self.runtime.graph, self.runtime.thread_id)
        # 初始化偏移量
        self.offset = self.manager.get_memory_index(self.task_type)
        self.messages = None
        
        # 绑定工具集
        self.llm = self.runtime.llm.bind_tools([
            consolidate_profile_memory,
            consolidate_episodic_memory,
            consolidate_semantic_memory
        ])
        logger.info(f"📂 任务 [{self.task_id}] 初始化完成, 当前 Index: {self.offset}")

    def add_index(self, value: int):
        self.offset += value
        self.manager.update_memory_index(self.task_type, self.offset)

    def load_history_messages(self):
        return get_messages(self.runtime.thread_id, self.runtime.graph, self.offset)
    
    def build_prompt(self, order_input: str) -> str:
        """构建全量 JSONL 上下文 Prompt"""
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
            "profile_memory": self.manager.export_full_jsonl(["profile"]),
            "episodic_memory": self.manager.export_full_jsonl(["episodic"]),
            "semantic_memory": self.manager.export_full_jsonl(["semantic"]),
            "conversation": self.messages,
            "instruction": order_input,
        })
    
    def should_trigger(self, store_client: Any) -> bool:
        self.messages = self.load_history_messages()
        if not self.messages:
            return False

        # 刷新状态，确保获取最新内存视图
        self.manager.refresh()
        self.offset = self.manager.get_memory_index(self.task_type)
        
        mem = self.manager.get_memories()
        logger.info(f"📝 任务 [{self.task_id}] 准备就绪 | Index: {self.offset} | "
                    f"P:{len(mem.get('profile_records', {}))} "
                    f"E:{len(mem.get('episodic_records', {}))} "
                    f"S:{len(mem.get('semantic_records', {}))}")
        return True

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        
        logger.info(f"🎬 执行任务: {self.task_id} | 类型: {self.task_type}")
        
        # 1. 调用 LLM
        instruction = TASK_INSTRUCTIONS.get(self.task_type, "Consolidate memory.")
        try:
            result = self.llm.invoke(self.build_prompt(instruction))
        except Exception as e:
            logger.error(f"❌ LLM 调用失败: {e}")
            return TaskExecutionResult(is_success=False, output_data=str(e))

        tool_calls = getattr(result, "tool_calls", [])
        
        # 2. 处理 Tool Calls (支持多调用并行)
        if tool_calls:
            for call in tool_calls:
                tool_name = call.get("name")
                args = call.get("args")
                func = TOOL_MAP.get(tool_name)
                
                if not func:
                    logger.warning(f"⚠️ 工具未找到: {tool_name}")
                    continue

                try:
                    tool_result = func.invoke(args)
                    record_key = f"{self.task_type}_records"
                    
                    # 通过 Manager 原子化更新状态 (利用强类型检查)
                    if isinstance(tool_result, dict):
                        update_data = tool_result.get(record_key)
                        reason = tool_result.get("organization_reason")
                        
                        if update_data:
                            self.manager.update_state(cast(MemoryStateKey, record_key), update_data)
                            if reason:
                                self.manager.update_state("organization_reason", reason)
                            self.add_index(len(self.messages))

                except Exception as e:
                    logger.error(f"❌ 工具执行异常 [{tool_name}]: {e}")
                    # 中断执行以保持原子性，不更新索引，等待下次重试
                    return TaskExecutionResult(is_success=False, output_data=f"Tool error: {str(e)}")
            
        return TaskExecutionResult(is_success=True, output_data="Memory successfully consolidated.")
# ==============================================================================
# 三个派生出的具体旁路独立原子任务 (极致干净，只声明类型和差异化默认配置)
# ==============================================================================
class ProfileMemoryTask(BaseMemoryExtractionTask):
    """画像记忆整理任务：用户画像通常不需要太频繁，默认每天凌晨 3 点跑一次"""
    task_type = "profile"
    default_cron = "*/1 * * * *"


class EpisodicMemoryTask(BaseMemoryExtractionTask):
    """情节/事件记忆提取任务：属于时间敏感型高频任务，默认每 10 分钟盘点一次快照"""
    task_type = "episodic"
    default_cron = "*/1 * * * *"


class SemanticMemoryTask(BaseMemoryExtractionTask):
    """语义知识图谱沉淀任务：属于重型长周期任务，使用更强大的大模型，默认每 1 小时整理一次"""
    task_type = "semantic"
    default_cron = "*/1 * * * *"
    default_params = {
        "min_confidence": 0.85,             # 语义入库要求极高的置信度
        "model_flavor": "claude-3-5-sonnet" # 知识沉淀选择推理能力更强的模型
    }