import abc
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from agi.scheduler.base import BaseTaskRuntime, BaseTaskUnit, TaskExecutionResult
from agi.scheduler.utils.state import get_memory_index, get_messages, update_state
from agi.scheduler.memory_task.memory_tools import (
    MEMORY_SYSTEM_PROMPT,
    consolidate_profile_memory,
    consolidate_episodic_memory,
    consolidate_semantic_memory
)
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger("MemoryTask")


class MemoryTaskRuntime(BaseTaskRuntime):
    """内聚了大模型交互所需要的全部重型环境依赖，让 Store 干净地回归存储本质"""
    def __init__(self, llm: Any, graph: Any, thread_id: str):
        super().__init__()
        self.llm = llm
        self.graph = graph
        self.thread_id = thread_id


class BaseMemoryExtractionTask(BaseTaskUnit, abc.ABC):
    """
    改造后的记忆提取基类。
    支持全量加载 Profile、Episodic、Semantic 三种维度的上下文。
    """
    task_type: str = "" 
    default_cron: str = "0 3 * * *"
    default_params: Dict[str, Any] = {
        "min_confidence": 0.7,
        "model_flavor": "gpt-4o-mini"
    }

    def __init__(self, runtime: MemoryTaskRuntime):
        super().__init__(runtime)
        self.offset = self.load_index()
        self.messages = None
        
        # 💡 此时 self.memories 将演变为一个 Dict[str, Any] 结构
        self.memories: Dict[str, Any] = {}
        
        self.llm = self.runtime.llm.bind_tools([
            consolidate_profile_memory,
            consolidate_episodic_memory,
            consolidate_semantic_memory
        ])

    def add_index(self, value: int):
        self.offset += value

    def load_index(self):
        return get_memory_index(self.runtime.thread_id, self.runtime.graph)

    def load_history_messages(self):
        return get_messages(self.runtime.thread_id, self.runtime.graph, self.offset)
    
    # ==============================================================================
    # 核心变更：全面拉取 3 种维度的全部记忆
    # ==============================================================================
    def load_memories(self, store_client: Any) -> Dict[str, Any]:
        """
        全量记忆灌流：一次性捞出 profile, episodic, semantic 三种核心记忆实体
        """
        mem_ns = ("users", self.target_id, "consolidated_memory")
        memory_types = ["profile", "episodic", "semantic"]
        fetched_memories = {}

        for m_type in memory_types:
            res = store_client.get(namespace=mem_ns, key=m_type)
            
            # 严格防呆解析：提取 Store 包装类中的真实 value，无数据则 fallback 为 "(none)"
            if res and hasattr(res, "value"):
                fetched_memories[m_type] = str(res.value)
            elif res:
                fetched_memories[m_type] = str(res)
            else:
                fetched_memories[m_type] = "(none)"
                
        return fetched_memories
    
    # ==============================================================================
    # 核心变更：重塑 Prompt 模板，建立 3 种记忆的强感知视窗
    # ==============================================================================
    def build_prompt(self, order_input: str) -> str:
        """构建面向大模型的全景结构化记忆提取 Prompt"""
        prompt = PromptTemplate.from_template(
            """{system_prompt}\n\n
            ==================================================
            [EXISTING USER MEMORY CONTEXT]
            --------------------------------------------------
            * CORE PROFILE (画像记忆):
            {profile_memory}
            
            * EPISODIC LOGS (情节/事件记忆):
            {episodic_memory}
            
            * SEMANTIC KNOWLEDGE (语义知识事实):
            {semantic_memory}
            ==================================================\n\n
            
            CONVERSATION_HISTORY (自增量新对话):\n{messages}\n
            
            INSTRUCTION: {instruction}
            """
        )
        
        return prompt.format(
            system_prompt=MEMORY_SYSTEM_PROMPT,
            profile_memory=self.memories.get("profile", "(none)"),
            episodic_memory=self.memories.get("episodic", "(none)"),
            semantic_memory=self.memories.get("semantic", "(none)"),
            messages=str(self.messages),
            instruction=order_input
        )
    
    def should_trigger(self, store_client: Any) -> bool:
        """准入控制"""
        self.messages = self.load_history_messages()
        if not self.messages:
            logger.info(f" Hobson-Skipped: 任务 [{self.task_id}] 未发现新对话消息，跳过本次触发。")
            return False
            
        # 🚀 加载全量 3 种记忆上下文
        self.memories = self.load_memories(store_client)
        return True

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """核心生命周期执行流不变，完全兼容全量上下文"""
        logger.info(f"🎬 启动记忆提取任务: {self.task_id} | 目标类型: {self.task_type}")
        
        if self.task_type == "profile":
            instruction = "Call 'consolidate_profile_memory' tool to analyze user core profile based on history and existing memory."
        elif self.task_type == "episodic":
            instruction = "Call 'consolidate_episodic_memory' tool to structure recent lifecycle events without repeating semantic or profile items."
        else:
            instruction = "Call 'consolidate_semantic_memory' tool to extract new declarative knowledge facts that do not overlap with profile/episodic data."
         
        prompt_text = self.build_prompt(instruction)
        result = self.llm.invoke(prompt_text)

        target_patches = getattr(result, "tool_calls", [])

        # 保持原有状态机跟进
        update_state(self.runtime.thread_id, self.runtime.graph, self.task_type, result)
        
        # 物理落库，保持原名字空间格式
        mem_ns = ("users", self.target_id, "consolidated_memory")
        payload = {
            "patches": target_patches, 
            "extracted_at": datetime.now().isoformat(),
            "meta_params": self.params
        }
        
        if hasattr(store_client, "aput"):
            await store_client.aput(namespace=mem_ns, key=f"{self.task_type}_patches_latest", value=payload)
        else:
            store_client.put(namespace=mem_ns, key=f"{self.task_type}_patches_latest", value=payload)

        if self.messages:
            self.add_index(len(self.messages))

        return TaskExecutionResult(
            is_success=True, 
            output_data=f"成功为用户 [{self.target_id}] 提取并固化了 {len(target_patches)} 条 [{self.task_type}] 记忆补丁。"
        )
# ==============================================================================
# 三个派生出的具体旁路独立原子任务 (极致干净，只声明类型和差异化默认配置)
# ==============================================================================
class ProfileMemoryTask(BaseMemoryExtractionTask):
    """画像记忆整理任务：用户画像通常不需要太频繁，默认每天凌晨 3 点跑一次"""
    task_type = "profile"
    default_cron = "0 3 * * *"


class EpisodicMemoryTask(BaseMemoryExtractionTask):
    """情节/事件记忆提取任务：属于时间敏感型高频任务，默认每 10 分钟盘点一次快照"""
    task_type = "episodic"
    default_cron = "*/10 * * * *"


class SemanticMemoryTask(BaseMemoryExtractionTask):
    """语义知识图谱沉淀任务：属于重型长周期任务，使用更强大的大模型，默认每 1 小时整理一次"""
    task_type = "semantic"
    default_cron = "0 * * * *"
    default_params = {
        "min_confidence": 0.85,             # 语义入库要求极高的置信度
        "model_flavor": "claude-3-5-sonnet" # 知识沉淀选择推理能力更强的模型
    }