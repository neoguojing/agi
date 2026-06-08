import abc
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from agi.scheduler.base import BaseTaskRuntime, BaseTaskUnit, TaskExecutionResult
from agi.scheduler.utils.state import get_memory_index, get_messages, update_state,get_memories,update_memory_index
from agi.scheduler.memory_task.memory_tools import (
    MEMORY_SYSTEM_PROMPT,
    consolidate_profile_memory,
    consolidate_episodic_memory,
    consolidate_semantic_memory
)
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
    改造后的记忆提取基类。
    支持全量加载 Profile、Episodic、Semantic 三种维度的上下文。
    """
    task_type: str = "" 
    default_cron: str = "0 3 * * *"
    default_params: Dict[str, Any] = {
        "min_confidence": 0.7,
        "model_flavor": "gpt-4o-mini"
    }

    def __init__(self, runtime: MemoryTaskRuntime, task_id: str, target_id: str, params: dict):
        super().__init__(runtime, task_id, target_id, params)
        self.offset = self.load_index()
        logger.info(f"📂 任务 [{self.task_id}] 初始化完成, 当前内存偏移量 (Index): {self.offset}")
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
        update_memory_index(self.runtime.thread_id,self.runtime.graph,self.task_type,self.offset)

    def load_index(self):
        return get_memory_index(self.runtime.thread_id, self.runtime.graph,self.task_type)

    def load_history_messages(self):
        return get_messages(self.runtime.thread_id, self.runtime.graph, self.offset)
    
    # ==============================================================================
    # 核心变更：重塑 Prompt 模板，建立 3 种记忆的强感知视窗
    # ==============================================================================
    def build_prompt(self, order_input: str) -> str:
        """构建面向大模型的全景结构化记忆提取 Prompt"""
        template = ChatPromptTemplate(
            [
                ("system", "{system_prompt}"),
                ("system", "{profile_memory}"),
                ("system", "{episodic_memory}"),
                ("system", "{semantic_memory}"),
                ("placeholder", "{conversation}"),
                ("human","{instruction}"),
            ]
        )

        def to_str(memory_dict):
            if memory_dict is None:
                return ""
            json_list_str = json.dumps(
                [v.model_dump() for v in memory_dict.values()],
                ensure_ascii=False,
                default=str
            )
            return json_list_str

        prompt_value = template.invoke(
            {
                "system_prompt": MEMORY_SYSTEM_PROMPT,
                "profile_memory": to_str(self.memories.get("profile", None)),
                "episodic_memory": to_str(self.memories.get("episodic", None)),
                "semantic_memory": to_str(self.memories.get("semantic", None)),
                "conversation": self.messages,
                "instruction": order_input,
            }
        )

        logger.info(f"📝 任务 [{self.task_id}] 构建 Prompt 完成 (Index: {self.offset})")
        return prompt_value
    
    def should_trigger(self, store_client: Any) -> bool:
        """准入控制"""
        self.messages = self.load_history_messages()
        if not self.messages:
            logger.info(f" Hobson-Skipped: 任务 [{self.task_id}] 未发现新对话消息，跳过本次触发。")
            return False

        # 🚀 加载全量 3 种记忆上下文并转换为 Dict 以支持 .get() 访问
        profile_records, episodic_records, semantic_records = get_memories(self.runtime.thread_id, self.runtime.graph)
        self.memories = {
            "profile": profile_records,
            "episodic": episodic_records,
            "semantic": semantic_records
        }

        logger.info(f"📝 任务 [{self.task_id}] 加载 memory 完成 (Index: {self.offset}) | Profile: {len(self.memories.get('profile', []))} Episodic: {len(self.memories.get('episodic', []))} Semantic: {len(self.memories.get('semantic', []))}")

        return True

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """核心生命周期执行流不变，完全兼容全量上下文"""
        logger.info(f"🎬 启动记忆提取任务: {self.task_id} | 目标类型: {self.task_type}")

        instruction = TASK_INSTRUCTIONS.get(self.task_type, "Call appropriate consolidate memory tool.")

        prompt_text = self.build_prompt(instruction)
        try:
            result = self.llm.invoke(prompt_text)
        except Exception as e:
            logger.error(f"❌ LLM invocation failed for task {self.task_id}: {e}")
            return TaskExecutionResult(
                is_success=False,
                output_data=f"LLM error during {self.task_type} memory extraction: {str(e)}"
            )

        tool_calls = getattr(result, "tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            call = tool_calls[0]
            tool_name = call["name"]
            args = call["args"]
            logger.info(f"🛠️ LLM 决定调用工具 [{tool_name}]，参数: {args},{call}")
            # 从映射中获取对应的工具函数并执行
            func = TOOL_MAP.get(tool_name)
            if func:
                try:
                    # 调用工具获得 Command 对象 (来自 memory_tools.py)
                    tool_result = func.invoke(args)
                    logger.info(f"tool_result:{tool_result}")
                    # records_dict = json.loads(tool_result.content)
                    # 确认 update 内容存在且为字典，然后更新状态
                    record_key = f"{self.task_type}_records"
                    update_records = tool_result.get(record_key, None)
                    reason = tool_result.get("organization_reason", None)
                    
                    logger.info(f"update_records:{update_records}")
                    logger.info(f"reason:{reason}")
                    if isinstance(update_records, dict) and len(update_records) > 0:
                        logger.info(f"💾 准备写入状态 [{record_key}], 载荷大小: {len(str(update_records))} 字符")
                        update_state(self.runtime.thread_id, self.runtime.graph, record_key, update_records)
                        update_state(self.runtime.thread_id, self.runtime.graph, "organization_reason", reason)
                        if self.messages:
                            self.add_index(len(self.messages))
                        logger.info(f"✅ 成功更新状态: {record_key} 使用工具 [{tool_name}] 的输出。index:{self.offset}")
                    else:
                        logger.warning(f"⚠️ 工具 [{tool_name}] 返回的更新载荷格式不正确 (expected dict, got {update_records})")
                except Exception as e:
                    logger.error(f"❌ 执行工具 [{tool_name}] 时发生异常: {e}")
            else:
                logger.warning(f"⚠️ 未能在 TOOL_MAP 中找到工具名称: {tool_name}")
        else:
            logger.info("ℹ️ LLM 未产生任何 tool_calls，跳过状态更新。")

        return TaskExecutionResult(
            is_success=True,
            output_data=f"成功处理用户 [{self.target_id}] 的 {self.task_type} 记忆任务。"
        )
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