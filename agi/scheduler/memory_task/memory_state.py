from typing import Any, Callable, Dict, Generator, List, Literal, Optional, TypeVar, Annotated,Union
from typing import cast,get_args
from langgraph.channels import LastValue
from langgraph.graph.state import CompiledStateGraph

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentState,
    ResponseT,
)

from agi.scheduler.memory_task.memory_models import (
    ProfileMemoryRecord,
    EpisodicMemoryRecord,
    SemanticMemoryRecord,
    MemoryTarget
)

logger = logging.getLogger(__name__)

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


MEMORY_KEY_MAP: Dict[MemoryTarget, str] = {
    "profile": "profile_records",
    "episodic": "episodic_records",
    "semantic": "semantic_records"
}

MemoryRecordT = Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord, dict]
MemoryFilterFn = Callable[[Any], bool]

MemoryStateKey = Literal[
    "profile_records", 
    "episodic_records", 
    "semantic_records", 
    "organization_reason", 
    "profile_message_index", 
    "episodic_message_index", 
    "semantic_message_index"
]

# 运行时校验使用的集合
ALLOWED_MEMORY_KEYS = set(get_args(MemoryStateKey))

INDEX_KEY_MAP: Dict[MemoryTarget, str] = {
    "profile": "profile_message_index",
    "episodic": "episodic_message_index",
    "semantic": "semantic_message_index"
}

# =====================================================
# 2. 场景 2 专用：在线上下文配置对象
# =====================================================
@dataclass
class AgentContextConfig:
    """Agent 运行时上下文提取、剪裁与链式过滤的配置参数"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    limits: Dict[MemoryTarget, int] = field(default_factory=lambda: {
        "profile": 50,
        "episodic": 30,
        "semantic": 50
    })
    
    custom_filters: List[MemoryFilterFn] = field(default_factory=list)
    
    # 默认设为 None，由底层方法接管“自动拉取所有支持的记忆类型”
    targets: Optional[List[MemoryTarget]] = None

# =====================================================
# 3. 核心内存管理服务 (MemoryManager)
# =====================================================
class MemoryManager:
    CURSOR_RECOVERY_RESET = "reset"
    CURSOR_RECOVERY_CLAMP = "clamp"
    CURSOR_RECOVERY_MODE = CURSOR_RECOVERY_RESET

    def __init__(self, runtime:'MemoryTaskRuntime'):
        self.runtime = runtime
        self.state: MemoryState = {}

    @property
    def config(self):
        return {"configurable": {"thread_id": self.runtime.thread_id}}

    async def refresh(self) -> None:
        try:
            if self.runtime.graph:
                snapshot = await self.runtime.graph.aget_state(self.config)
                self.state = snapshot.values if hasattr(snapshot, "values") else snapshot
            elif self.runtime.client:
                snapshot = await self.runtime.client.threads.get_state(thread_id=self.runtime.thread_id)
                self.state = snapshot.get("values") if isinstance(snapshot, dict) else getattr(snapshot, "values", {})

            self.state = cast(MemoryState, self.state or {})
            self.log_state_summary()

        except Exception:
            logger.exception("memory refresh failed thread_id=%s", self.runtime.thread_id)
            raise

    def get_messages(self, limit: Optional[int] = None) -> list:
        messages = self.state.get("messages", [])
        return messages[-limit:] if limit and isinstance(messages, list) else messages

    async def repair_memory_index(self, task_type: MemoryTarget) -> int:
        """
        修复异常Cursor

        场景:
            cursor > len(messages)

        原因:
            - Message Trimming
            - Summary压缩
            - Thread恢复
            - State迁移
        """

        cursor_key = INDEX_KEY_MAP.get(task_type)

        if not cursor_key:
            return 0

        messages = self.get_messages(limit=None)
        message_count = len(messages)

        cursor = self.state.get(cursor_key, 0)

        if cursor <= message_count:
            return cursor

        repaired_cursor = (
            0
            if self.CURSOR_RECOVERY_MODE == self.CURSOR_RECOVERY_RESET
            else message_count
        )

        logger.warning(
            "cursor overflow detected task=%s cursor=%s message_count=%s repaired=%s",
            task_type, cursor, message_count, repaired_cursor
        )

        try:
            if self.runtime.graph:
                await self.runtime.graph.aupdate_state(self.config, {cursor_key: repaired_cursor})
            elif self.runtime.client:
                await self.runtime.client.threads.update_state(
                    thread_id=self.runtime.thread_id,
                    values={cursor_key: repaired_cursor},
                )

            self.state[cursor_key] = repaired_cursor

            logger.info(
                "cursor repaired task=%s old=%s new=%s",
                task_type, cursor, repaired_cursor
            )

        except Exception:
            logger.exception("cursor repair failed task=%s", task_type)
            raise

        return repaired_cursor

    async def get_incremental_messages(self, task_type: MemoryTarget) -> list:
        """
        获取增量消息
        自动修复异常Cursor
        """

        cursor = await self.repair_memory_index(task_type)

        messages = self.get_messages(limit=None)

        if cursor >= len(messages):
            return []

        incremental = messages[cursor:]
        next_cursor = len(messages)

        logger.info(
            "incremental messages fetched task=%s cursor=%s count=%s next_cursor=%s",
            task_type, cursor, len(incremental), next_cursor
        )

        return incremental

    async def commit_memory(self, task_type: MemoryTarget, memory_value: object, cursor: int,reason: str) -> None:
        cursor_key = INDEX_KEY_MAP.get(task_type)

        if not cursor_key:
            raise ValueError(f"cursor key not found: {task_type}")

        current_cursor = self.state.get(cursor_key, 0)

        if cursor <= current_cursor:
            logger.warning(
                "cursor rollback ignored task=%s current=%s new=%s",
                task_type, current_cursor, cursor
            )
            return

        memory_key = MEMORY_KEY_MAP.get(task_type)
        if not memory_key:
            raise ValueError(f"memory key not found: {task_type}")
        
        current_memory = self.state.get(memory_key, {})

        merged_memory = (
            memory_reducer(current_memory, memory_value)
            if memory_key in MEMORY_KEY_MAP.values()
            else memory_value
        )

        payload = {memory_key: merged_memory, cursor_key: cursor,"organization_reason":reason}

        try:
            if self.runtime.graph:
                await self.runtime.graph.aupdate_state(self.config, payload)
            elif self.runtime.client:
                await self.runtime.client.threads.update_state(thread_id=self.runtime.thread_id, values=payload)

            self.state[memory_key] = merged_memory
            self.state[cursor_key] = cursor
            self.state["organization_reason"] = reason

            logger.info(
                "memory committed thread_id=%s task=%s cursor=%s",
                self.runtime.thread_id, task_type, cursor
            )

        except Exception:
            logger.exception(
                "memory commit failed thread_id=%s task=%s",
                self.runtime.thread_id, task_type
            )
            raise

    async def commit_incremental_memory(self, task_type: MemoryTarget, memory_value: object,reason: str) -> None:
        await self.commit_memory(
            task_type=task_type,
            memory_value=memory_value,
            cursor=len(self.get_messages(limit=None)),
            reason=reason
        )
    # =====================================================
    # 🛠️ 底层私有工具链
    # =====================================================

    def log_state_summary(self) -> None:
        """
        🚀 极简一行流：使用 logger.info 打印所有记忆条数与消息消费偏移量
        """
        # 内部安全取值辅助
        def _get(k: str, default: object) -> Any:
            return self.state.get(k, default) if isinstance(self.state, dict) else getattr(self.state, k, default)

        # 1. 提取各个 dict[str, Record] 的长度
        p_dict = _get("profile_records", {})
        e_dict = _get("episodic_records", {})
        s_dict = _get("semantic_records", {})
        m_list = _get("messages", [])
        
        print()
        p_count = len(p_dict) if isinstance(p_dict, dict) else 0
        e_count = len(e_dict) if isinstance(e_dict, dict) else 0
        s_count = len(s_dict) if isinstance(s_dict, dict) else 0
        m_list = len(m_list) if isinstance(m_list, list) else 0
        # 2. 提取各个游标偏移量
        p_idx = _get("profile_message_index", 0)
        e_idx = _get("episodic_message_index", 0)
        s_idx = _get("semantic_message_index", 0)

        # 3. 严格单行输出，包含指标前缀，方便正则/ELK 提取
        logger.info(
            "🧠 [Memory State] Count -> Profile: %d, Episodic: %d, Semantic: %d | Offset -> Profile: %d, Episodic: %d, Semantic: %d, Messages: %d",
            p_count, e_count, s_count, p_idx, e_idx, s_idx,m_list
        )
    def _to_record_list(self, target: MemoryTarget) -> List[MemoryRecordT]:
        """核心适配器：返回强类型的记录列表"""
        state_key = MEMORY_KEY_MAP.get(target)
        if not state_key:
            return []
            
        data = self.state.get(state_key) if isinstance(self.state, dict) else getattr(self.state, state_key, None)
        if not data:
            return []
            
        if isinstance(data, dict):
            data = list(data.values())

        logger.info(f"_to_record_list got {target}:{len(data)}")
        return data

    def _get_utc_timestamp(self, rec: MemoryRecordT) -> datetime:
        if isinstance(rec, str): 
            return datetime.min.replace(tzinfo=timezone.utc)
            
        dt: Optional[object] = None
        if hasattr(rec, 'updated_at'):
            dt = getattr(rec, 'updated_at')
        elif isinstance(rec, dict):
            val = rec.get('updated_at')
            if isinstance(val, str):
                try:
                    dt = datetime.fromisoformat(val)
                except ValueError:
                    dt = None
            elif isinstance(val, datetime):
                dt = val

        if dt is None or not isinstance(dt, datetime):
            return datetime.min.replace(tzinfo=timezone.utc)
        
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    def _ensure_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    def _get_v(self, rec: MemoryRecordT, key: str, default: str = "") -> str:
        if isinstance(rec, dict):
            return str(rec.get(key, default))
        return str(getattr(rec, key, default))

    def _serialize_to_dict(self, rec: MemoryRecordT, target: MemoryTarget) -> Dict[str, object]:
        """返回严格的 Dict[str, object] 而非 Dict[str, Any]"""
        updated_at = self._get_utc_timestamp(rec).isoformat()
        meta = {"memory_type": str(target), "updated_at": updated_at}
        
        if isinstance(rec, str):
            return {**meta, "raw_data": rec}
            
        data_dict: Dict[str, object] = {}
        if isinstance(rec, dict):
            data_dict = rec.copy()
        elif hasattr(rec, 'model_dump'):
            data_dict = rec.model_dump(mode="json")
        elif hasattr(rec, '__dict__'):
            data_dict = {k: v for k, v in rec.__dict__.items() if not k.startswith('_')}
            
        for k, v in data_dict.items():
            if isinstance(v, datetime):
                data_dict[k] = v.isoformat()
                
        return {**meta, **data_dict}

    # =====================================================
    # 🎨 格式化路由
    # =====================================================
    def _format_profile(self, records: List[MemoryRecordT]) -> str:
        lines = [f"--- PROFILE MEMORY ({len(records)}) ---"]
        for rec in records:
            item = {"key": self._get_v(rec, 'key'), "value": self._get_v(rec, 'value')}
            lines.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(lines)

    def _format_episodic(self, records: List[MemoryRecordT]) -> str:
        lines = [f"--- EPISODIC MEMORY ({len(records)}) ---"]
        for rec in records:
            try:
                rec_json_dict = {
                    "id": self._get_v(rec, 'id'),
                    "event_time": str(self._get_v(rec, 'event_time')),
                    "summary": self._get_v(rec, 'summary')
                }
                lines.append(json.dumps(rec_json_dict, ensure_ascii=False))
            except Exception as e:
                logger.warning("⚠️ 传记记忆记录解析失败。Error: %s", str(e))
        return "\n".join(lines)

    def _format_semantic(self, records: List[MemoryRecordT]) -> str:
        lines = [f"--- SEMANTIC MEMORY ({len(records)}) ---"]
        for rec in records:
            item = {
                "subject": self._get_v(rec, 'subject'),
                "predicate": self._get_v(rec, 'predicate'),
                "object": self._get_v(rec, 'object')
            }
            lines.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(lines)

    # =====================================================
    # 🌟 场景 1：全量提取 JSONL 数据
    # =====================================================
    def export_full_jsonl_stream(self, targets: Optional[List[MemoryTarget]] = None) -> Generator[str, None, None]:
        """targets 参数现在被严格约束为 List[MemoryTarget]"""
        selected_targets = list(MEMORY_KEY_MAP.keys()) if targets is None else targets
        for target in selected_targets:
            records = self._to_record_list(target)
            for rec in records:
                yield json.dumps(self._serialize_to_dict(rec, target), ensure_ascii=False)

    def export_full_jsonl(self, targets: Optional[List[MemoryTarget]] = None) -> str:
        return "\n".join(self.export_full_jsonl_stream(targets))

    # =====================================================
    # 🌟 场景 2：在线 Agent 上下文组装
    # =====================================================
    def get_agent_context(self, config: Optional[AgentContextConfig] = None) -> str:
        """config 参数现在被严格约束为 AgentContextConfig 实例"""
        cfg = config or AgentContextConfig()
        sections: List[str] = []
        
        start_utc = self._ensure_utc(cfg.start_time)
        end_utc = self._ensure_utc(cfg.end_time)
        selected_targets = list(MEMORY_KEY_MAP.keys()) if cfg.targets is None else cfg.targets

        formatter_map: Dict[str, Callable[[List[MemoryRecordT]], str]] = {
            "profile": self._format_profile,
            "episodic": self._format_episodic,
            "semantic": self._format_semantic
        }

        for target in selected_targets:
            raw_records = self._to_record_list(target)
            if not raw_records:
                continue

            filtered = [
                r for r in raw_records
                if (not cfg.custom_filters or all(f(r) for f in cfg.custom_filters)) and
                (not start_utc or self._get_utc_timestamp(r) >= start_utc) and
                (not end_utc or self._get_utc_timestamp(r) <= end_utc)
            ]
            if not filtered:
                continue

            filtered.sort(key=self._get_utc_timestamp, reverse=True)
            sliced = filtered[:cfg.limits.get(target, 50)]

            formatter = formatter_map.get(str(target))
            if formatter:
                sections.append(formatter(sliced))

        return "\n\n".join(sections) if sections else "No memory available."