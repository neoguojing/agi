import abc
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from agi.scheduler.base import BaseTaskUnit,TaskExecutionResult,BaseTaskRuntime

logger = logging.getLogger("MemoryTask")

class MemoryTaskRuntime(BaseTaskRuntime):
    """内聚了大模型交互所需要的全部重型环境依赖，让 Store 干净地回归存储本质"""
    def __init__(self, llm: Any, graph: Any = None):
        self.llm = llm
        self.graph = graph

class BaseMemoryExtractionTask(BaseTaskUnit, abc.ABC):
    """
    改造后的记忆提取基类。
    继承自 BaseTaskUnit，全面适配纯代码插槽与 Store 驱动架构。
    """
    # 由具体子类覆写
    target_type: str = "" 
    
    # 全局默认配置：默认每天凌晨 3 点跑一次，最低置信度 0.7
    default_cron: str = "0 3 * * *"
    default_params: Dict[str, Any] = {
        "min_confidence": 0.7,
        "model_flavor": "gpt-4o-mini"
    }

    runtime_schema: MemoryTaskRuntime

    def should_trigger(self, store_client: Any) -> bool:
        """
        准入控制：利用注入的 self.target_id 去 store 检查该用户是否有未处理的原始对话快照。
        """
        # 模拟从你的 BaseStore 获取特定用户的对话消息流
        msg_ns = ("users", self.target_id, "raw_messages")
        messages = store_client.search(namespace_prefix=msg_ns, limit=1)
        
        if not messages:
            logger.info(f"⏭️ 主体 {self.target_id} 近期无新对话消息，跳过本次 {self.task_type} 触发。")
            return False
        return True

    async def _call_llm_for_extraction(self, store_client: Any, prompt: str) -> Any:
        """调用大模型进行结构化输出 (利用 store_client 中可能挂载的 llm 句柄或外部轻量客户端)"""
        try:
            # 这里的 llm 可以是挂载在 store_client 上的特殊 handle，或者你在 params 里传入的配置
            llm_client = getattr(store_client, "llm", None)
            if not llm_client:
                logger.error("Store client 中未挂载有效的 LLM 运行时句柄")
                return None
                
            schema = TARGET_SCHEMA_MAP.get(self.target_type)
            llm_with_struct = llm_client.with_structured_output(schema)
            
            # 执行异步大模型请求
            struct_result = await llm_with_struct.ainvoke(prompt)
            return struct_result
        except Exception as e:
            logger.exception(f"任务 {self.task_id} LLM 调用失败: {str(e)}")
            return None

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """
        核心生命周期执行流：完全利用 self.target_id 和 self.params 自包含运行
        """
        logger.info(f"🎬 启动记忆提取任务: {self.task_id} 目标类型: {self.target_type}")
        
        # 1. 捞取该用户的原始对话上下文
        msg_ns = ("users", self.target_id, "raw_messages")
        msg_items = store_client.search(namespace_prefix=msg_ns, limit=100)
        conversation_text = "\n".join([str(item.value.get("content", "")) for item in msg_items])

        # 2. 捞取已有的历史记忆，作为 Few-Shot 上下文提供给大模型
        mem_ns = ("users", self.target_id, "consolidated_memory")
        existing_mem = store_client.get(namespace=mem_ns, key=self.target_type)
        existing_mem_text = str(existing_mem.value) if existing_mem else ""

        # 3. 组装 Prompt
        prompt = build_memory_extraction_prompt(
            conversation=conversation_text,
            existing_memory=existing_mem_text,
            target=self.target_type,
        )

        # 4. 驱动大模型
        llm_payload = await self._call_llm_for_extraction(store_client, prompt)
        if not llm_payload:
            return TaskExecutionResult(is_success=False, error_log="LLM 结构化解析未返回有效载荷")

        # 5. 解析并转换为变更补丁（Patches）
        filtered_extraction = MemoryExtractionResult()
        if isinstance(llm_payload, ProfileMemoryList):
            filtered_extraction.profile_memories = llm_payload.items
        elif isinstance(llm_payload, EpisodicMemoryList):
            filtered_extraction.episodic_memories = llm_payload.items
        elif isinstance(llm_payload, SemanticMemoryList):
            filtered_extraction.semantic_memories = llm_payload.items

        patches = filtered_extraction.to_patches(reason=f"Automatic {self.target_type} memory extraction")
        
        # 6. 利用自包含属性 self.params['min_confidence'] 进行去重和过滤
        # 我们直接将原方法搬配过来，现有的 existing_records 直接通过 store 现场读取
        existing_records_raw = store_client.search(namespace_prefix=("users", self.target_id, "records"), limit=1000)
        existing_records = [item.value for item in existing_records_raw]
        
        patches = self._filter_and_deduplicate_patches(patches, existing_records)
        
        patch_strategy = "replace" if self.target_type == "profile" else "merge"
        target_patches = tuple(
            p.model_copy(update={"strategy": patch_strategy})
            for p in patches if p.target == self.target_type
        )

        if not target_patches:
            return TaskExecutionResult(is_success=True, output_data="未发现高置信度或非重复的记忆变更")

        # 7. 核心业务持久化：将提取出的增量补丁拍进用户的图数据库/存储空间中
        await store_client.aput(
            namespace=mem_ns,
            key=f"{self.target_type}_patches_latest",
            value={"patches": [p.model_dump() for p in target_patches], "extracted_at": datetime.now().isoformat()}
        )

        return TaskExecutionResult(
            is_success=True, 
            output_data=f"成功为 {self.target_type} 提取并固化了 {len(target_patches)} 条记忆补丁。"
        )

    def _filter_and_deduplicate_patches(self, patches: tuple, existing_records: list) -> tuple:
        """保持你原有的去重过滤逻辑完全不动，仅将阈值读取改为自包含属性"""
        updated = []
        seen_keys = set()
        min_conf = self.params.get("min_confidence", 0.7) # 从合并后的参数中安全获取

        for patch in patches:
            operations = []
            for op in patch.operations:
                if op.op != "add":
                    operations.append(op)
                    continue

                confidence = float(op.value.get("confidence", 0.5) or 0.0)
                if confidence < min_conf: # 🛠️ 动态控制线
                    continue

                key = record_dedup_key(self.target_type, op.value)
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)
                operations.append(op)

            if operations:
                updated.append(patch.model_copy(update={"operations": tuple(operations)}))

        return tuple(updated)


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
        "min_confidence": 0.85,          # 语义入库要求极高的置信度
        "model_flavor": "claude-3-5-sonnet" # 知识沉淀选择推理能力更强的模型
    }