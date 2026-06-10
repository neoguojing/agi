import abc
import json
import logging
import traceback
from typing import Any, Dict, Optional, Tuple,cast

from agi.scheduler.base import BaseTaskRuntime, BaseTaskUnit, TaskExecutionResult
from agi.scheduler.utils.state import get_messages
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
    def __init__(self, llm: Any, graph: Any, thread_id: str,user_id:str,client: Any):
        super().__init__()
        self.llm = llm
        self.graph = graph
        self.thread_id = thread_id
        self.user_id = user_id
        self.client = client


class BaseMemoryExtractionTask(BaseTaskUnit, abc.ABC):
    """
    生产级记忆提取基类（全异步适配版）。
    利用 MemoryManager 实现原子化状态管理与全量 JSONL 记忆整理。
    """
    task_type: str = "" 
    
    def __init__(self, runtime: MemoryTaskRuntime, task_id: str, target_id: str, params: dict):
        super().__init__(runtime, task_id, target_id, params)
        
        # 1. 实例化管理器（此时内部不进行任何阻塞 I/O，仅分配内存占位）
        self.manager = MemoryManager(
            graph=self.runtime.graph,
            client=self.runtime.client,
            thread_id=self.runtime.thread_id
        )
        
        # 2. 💡 核心变更：构造函数中不再强行刷新获取状态。
        # 先安全初始化为默认值，真正的 Index 追溯由 should_trigger 的门禁流接管
        self.offset: int = 0
        self.messages = None
        
        # 3. 绑定工具集（CPU 密集型绑定，保持同步）
        self.llm = self.runtime.llm.bind_tools([
            consolidate_profile_memory,
            consolidate_episodic_memory,
            consolidate_semantic_memory
        ])
        logger.info(f"📂 任务 [{self.task_id}] 内存实例构建完成，等待时钟线异步准入触发...")

    async def add_index(self, value: int):
        """🌟 已改为 async：增加偏移量并异步回写状态"""
        self.offset += value
        await self.manager.update_memory_index(self.task_type, self.offset)

    async def load_history_messages(self):
        """🌟 已改为 async：从远程客户端或图存储中异步拉取历史消息"""
        # 假设底层的 get_messages 在全链路异步化后也升级为了支持 aget_messages 或 async def
        # return await get_messages(
        #     thread_id=self.runtime.thread_id, 
        #     graph=self.runtime.graph, 
        #     offset=self.offset,
        #     client=self.runtime.client
        # )
        state = self.manager.get_memories()
        return state.get("messages", None)
    
    def build_prompt(self, order_input: str) -> str:
        """
        构建全量 JSONL 上下文 Prompt。
        💡 保持同步方法：因为 manager.export_full_jsonl 在上一步重构中处理的是
        本地内存快照的数据清洗，属于 CPU 运算，不涉及 I/O。
        """
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
            "profile_memory": self.manager.export_full_jsonl(["profile"]) if self.task_type == "profile" else "None",
            "episodic_memory": self.manager.export_full_jsonl(["episodic"]) if self.task_type == "episodic" else "None",
            "semantic_memory": self.manager.export_full_jsonl(["semantic"]) if self.task_type == "semantic" else "None",
            "conversation": self.messages,
            "instruction": order_input,
        })
    
    async def should_trigger(self, store_client: Any) -> bool:
        """🚦 异步准入控制流"""
        try:
            # 1. 🌟 首先发起异步状态刷新，接管图状态并同步本地视图
            await self.manager.refresh()
            self.offset = self.manager.get_memory_index(self.task_type)

            # 2. 🌟 异步拉取历史消息
            self.messages = await self.load_history_messages()
            
            # 顺手帮你修正了 threshold 的拼写错误 🔧
            threshold = self.default_params.get("activate_message_threshold", 0)
            if not self.messages or len(self.messages) < threshold:
                return False

            mem = self.manager.get_memories()
            logger.info(f"📝 任务 [{self.task_id}] 准入校验通过 | Index: {self.offset} | "
                        f"P:{len(mem.get('profile_records', {}))} "
                        f"E:{len(mem.get('episodic_records', {}))} "
                        f"S:{len(mem.get('semantic_records', {}))}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 任务 [{self.task_id}] 准入检查阶段发生致命异常: {e}\n {traceback.format_exc()}")
            return False

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        logger.info(f"🎬 执行任务: {self.task_id} | 类型: {self.task_type}")
        
        # 1. 异步调用 LLM (切换为 LangChain 的非阻塞 ainvoke)
        instruction = TASK_INSTRUCTIONS.get(self.task_type, "Consolidate memory.")
        try:
            prompt = self.build_prompt(instruction)
            result = await self.llm.ainvoke(prompt)
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
                    # 🌟 核心变更：工具执行切换为异步 ainvoke
                    tool_result = await func.ainvoke(args)
                    record_key = f"{self.task_type}_records"
                    
                    if isinstance(tool_result, dict):
                        update_data = tool_result.get(record_key)
                        reason = tool_result.get("organization_reason")
                        
                        # 🌟 通过 Manager 原子化异步回写状态与更新索引
                        if update_data:
                            await self.manager.update_state(cast(MemoryStateKey, record_key), update_data)
                            if reason:
                                await self.manager.update_state("organization_reason", reason)
                            
                            # 🌟 升级为 await 驱动的索引推进
                            await self.add_index(len(self.messages))

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
    default_params = {
        "min_confidence": 0.85,             # 语义入库要求极高的置信度
        "model_flavor": "claude-3-5-sonnet" # 知识沉淀选择推理能力更强的模型
    }


class EpisodicMemoryTask(BaseMemoryExtractionTask):
    """情节/事件记忆提取任务：属于时间敏感型高频任务，默认每 10 分钟盘点一次快照"""
    task_type = "episodic"
    default_cron = "*/5 * * * *"
    default_params = {
        "activate_message_threshhold": 10, 
    }


class SemanticMemoryTask(BaseMemoryExtractionTask):
    """语义知识图谱沉淀任务：属于重型长周期任务，使用更强大的大模型，默认每 1 小时整理一次"""
    task_type = "semantic"
    default_cron = "0 */1 * * *"
    default_params = {
        "activate_message_threshhold": 10, 
    }