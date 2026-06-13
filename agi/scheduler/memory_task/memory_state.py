from typing import Any, Callable, Dict, Generator, List, Literal, Optional, TypeVar, Annotated,Union
from typing import cast,get_args
from langgraph.channels import LastValue
from langgraph.graph.state import CompiledStateGraph

from aiorwlock import RWLock
import json
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager

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
    MemoryTarget,
    CONTAINER_MAPPING,
    SafeScalarContainer
)

logger = logging.getLogger(__name__)

# --- Constants and Reducers ---

def memory_reducer(
    state: Optional[dict[str, Any]],  # 状态现在是 dict
    writes: Union[list[Any], dict[str, Optional[Any]]]
) -> dict[str, Any]:
    """
    A high-performance generic reducer using dictionary state.
    """
    merged: dict[str, Any] = state.copy() if state else {}
    if isinstance(writes, list):
        for r in writes:
            if hasattr(r, "dedup_key") and r.dedup_key:
                merged[r.dedup_key] = r

    elif isinstance(writes, dict):
        for k, r in writes.items():
            if r is None:
                merged.pop(k, None)  # 显式删除
            else:
                real_k = r.dedup_key if hasattr(r, "dedup_key") and r.dedup_key else k
                merged[real_k] = r

    return merged

class MemoryState(AgentState[ResponseT]):
    """State schema for the memory organization middleware."""
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

ALLOWED_MEMORY_KEYS = set(get_args(MemoryStateKey))

INDEX_KEY_MAP: Dict[MemoryTarget, str] = {
    "profile": "profile_message_index",
    "episodic": "episodic_message_index",
    "semantic": "semantic_message_index"
}

MEMORY_KEY = "memories"

# --- Context Config ---

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
    targets: Optional[List[MemoryTarget]] = None

# --- Core Manager ---

class MemoryManager:
    CURSOR_RECOVERY_RESET = "reset"
    CURSOR_RECOVERY_CLAMP = "clamp"
    CURSOR_RECOVERY_MODE = CURSOR_RECOVERY_CLAMP

    def __init__(self, runtime: 'MemoryTaskRuntime'):
        self.runtime = runtime
        self._state: Dict[str, Any] = {}
        self._messages: List[Any] = []
        self._lock = RWLock()  
        
        # 状态控制
        self._is_initialized = False 
        # 内部游标追踪器：Key 是 task_type，Value 是当前正在处理的终点游标边界
        self._processing_cursors: Dict[str, int] = {}

    # ==================== Getter 属性封装 ====================

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.runtime.thread_id}}

    @property
    def namespace(self) -> tuple:
        return ("agi", f"{self.runtime.user_id}", "memories")

    @property
    def messages(self) -> list:
        return self._messages.copy()

    @property
    def state(self) -> dict:
        return self._state.copy()

    # ==================== 核心内部生命周期 ====================

    async def refresh_messages(self) -> list:
        """
        强制从远程状态机刷新当前线程的消息历史。
        绕过懒加载锁，确保每次调用都能获取最新消息。
        """
        local_messages = []
        if self.runtime.client:
            snapshot = await self.runtime.client.threads.get_state(thread_id=self.runtime.thread_id)
            state = snapshot.get("values") if isinstance(snapshot, dict) else getattr(snapshot, "values", {})
            local_messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        elif self.runtime.graph:
            snapshot = await self.runtime.graph.aget_state(self.config)
            state = snapshot.values if hasattr(snapshot, "values") else snapshot
            local_messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])

        # 安全地更新本地消息缓存
        async with self._lock.writer:
            self._messages = local_messages
            
        return self._messages
    
    async def ensure_initialized(self) -> None:
        """
        确保内存管理器已初始化（懒加载设计）。
        🌟 修复一：直接采用原生数据，去除所有 .value 或 ["value"] 的判断。
        🌟 修复二：将游标和原因一并加入加载列表，防止重启后游标丢失。
        🌟 修复三：引入 Pydantic 容器解析，并加入 try-except 容错，平滑降级防止旧标量导致 orjson 崩溃。
        """
        if self._is_initialized:
            return

        async with self._lock.writer:
            if self._is_initialized:  # 双重检查锁
                return

            logger.info("Initializing memory manager from remote store... thread_id=%s", self.runtime.thread_id)
            local_state_updates = {}

            # 2. 统一加载所有相关的自定义存储 Key（包含记忆主体、游标索引、提交原因）
            keys_to_load = ALLOWED_MEMORY_KEYS
            
            if self.runtime.store:
                for key in keys_to_load:
                    # 🌟 核心修复：把 try 提到最外层，死死护住 aget 这一行
                    try:
                        raw_data = await self.runtime.store.aget(namespace=self.namespace, key=key)
                        if raw_data is not None:
                            # 根据映射自动找到对应的 Pydantic 容器进行反序列化灌注
                            container_cls = CONTAINER_MAPPING.get(key)
                            if container_cls:
                                # 标准路径：用 RootModel 进行高级解包还原
                                container_instance = container_cls.model_validate(raw_data)
                                local_state_updates[key] = container_instance.root
                            else:
                                local_state_updates[key] = raw_data
                                
                    except Exception as e:
                        # 🌟 强力兜底：无论是 aget 内部反序列化历史裸数据崩溃（orjson.JSONDecodeError），
                        # 还是 RootModel 校验失败，统统在这里被拦截。
                        logger.warning(
                            "Failed to load or hydrate key '%s' from remote store. "
                            "This usually happens when old raw scalars (like pure int/str) exist in DB. "
                            "Error: %s", 
                            key, str(e)
                        )
                        
                        # 🌟 绝妙降级：既然底层已经因为不是 JSON 而解析失败了，那它在数据库里大概率就是个裸字符串或数字。
                        # 我们选择暂时在本地内存中初始化为空（或者你可以根据 key 的类型给个 0 或 {} 兜底）。
                        # 当下一次 commit 触发时，新代码会用 SafeScalarContainer/Container 格式把正确的数据写进去，从而洗干净数据库。
                        if "index" in key:
                            local_state_updates[key] = 0  # 游标类 Key 挂了，安全兜底为 0 
                        else:
                            local_state_updates[key] = {} # 记忆体类 Key 挂了，安全兜底为空字典

            self._state.update(local_state_updates)
            self._is_initialized = True
            
            print("*******************{}*******************".format(local_state_updates))
            self.log_state_summary()

    async def repair_memory_index(self, task_type: MemoryTarget) -> int:
        """修复越界的游标"""
        cursor_key = INDEX_KEY_MAP.get(task_type)
        if not cursor_key:
            return 0

        async with self._lock.reader:
            message_count = len(self._messages)
            cursor = self._state.get(cursor_key, 0)

        if cursor <= message_count:
            return cursor

        repaired_cursor = (
            0 if self.CURSOR_RECOVERY_MODE == self.CURSOR_RECOVERY_RESET else message_count
        )

        logger.warning("cursor overflow detected task=%s cursor=%s, repaired=%s", task_type, cursor, repaired_cursor)

        if self.runtime.store:
            try:
                # 写入纯数字游标
                await self.runtime.store.aput(namespace=self.namespace, key=cursor_key, value=repaired_cursor)
            except Exception:
                logger.exception("cursor repair remote store failed")
                raise

        async with self._lock.writer:
            self._state[cursor_key] = repaired_cursor

        return repaired_cursor

    # ==================== 对外业务接口（API） ====================

    async def get_incremental_messages(self, task_type: MemoryTarget) -> list:
        """
        获取增量消息。
        🌟 游标完全在类内部闭环：自动在内存中记录本次处理的“终点线边界”，外部完全不需要感知游标。
        """
        await self.ensure_initialized()
        
        await self.refresh_messages()  # 强制刷新消息，确保拿到最新的消息列表
        
        cursor = await self.repair_memory_index(task_type)

        async with self._lock.writer:
            current_len = len(self._messages)
            if cursor >= current_len:
                self._processing_cursors[task_type] = current_len
                return []
            
            incremental = self._messages[cursor:current_len]
            
            # 🌟 内部闭环：静默默契地记下这一轮大模型处理的截止位置
            self._processing_cursors[task_type] = current_len
            
            logger.info("Fetched incremental messages. task=%s, from_cursor=%s, locked_to_boundary=%s",
                        task_type, cursor, current_len)
            return incremental.copy()

    async def commit_incremental_memory(self, task_type: MemoryTarget, memory_value: object, reason: str) -> None:
        """
        提交增量记忆。
        🌟 外部调用无需传游标：内部自动对齐 get 时记录的终点线，实现完美解耦。
        """
        await self.ensure_initialized()
        
        # 1. 从内部“记事本”获取刚才 get 锁定的目标游标
        async with self._lock.reader:
            target_cursor = self._processing_cursors.get(task_type)
            
        if target_cursor is None:
            # 异常兜底：如果外部直接调 commit 而没调 get，采用当前已知游标
            async with self._lock.reader:
                cursor_key = INDEX_KEY_MAP.get(task_type)
                target_cursor = self._state.get(cursor_key, 0)
            logger.warning("No active processing cursor found for %s, falling back to current cursor: %s", task_type, target_cursor)

        # 2. 推进核心提交
        await self.commit_memory(
            task_type=task_type,
            memory_value=memory_value,
            cursor=target_cursor,
            reason=reason
        )

    async def commit_memory(self, task_type: MemoryTarget, memory_value: object, cursor: int, reason: str) -> None:
        """核心直写式提交（Write-Through 模式）"""
        cursor_key = INDEX_KEY_MAP.get(task_type)
        memory_key = MEMORY_KEY_MAP.get(task_type)
        if not cursor_key or not memory_key:
            raise ValueError(f"cursor or memory key not found for task: {task_type}")

        async with self._lock.reader:
            current_cursor = self._state.get(cursor_key, 0)
            current_memory = self._state.get(memory_key, {})

        # 拦截过期/回滚的提交
        if cursor <= current_cursor:
            logger.warning("cursor rollback ignored task=%s current=%s incoming_commit_cursor=%s", 
                           task_type, current_cursor, cursor)
            return

        # 锁外合并字典数据
        merged_memory = (
            memory_reducer(current_memory, memory_value)
            if memory_key in MEMORY_KEY_MAP.values()
            else memory_value
        )

        # 🌟 极简直写：直接将你原生的纯数据（字典/整数/字符串）写入 Store，不加任何包装
        if self.runtime.store:
            try:
                # 🌟 优雅脱水写入：
                # 1. 数字标量安全包裹
                await self.runtime.store.aput(namespace=self.namespace, key=cursor_key, value=SafeScalarContainer(cursor).model_dump(mode="json"))
                
                # 2. 记忆载体通过映射的容器，一行代码完成高性能对象脱水 (转为纯原生 JSON 字典)
                if memory_value:
                    container_cls = CONTAINER_MAPPING.get(memory_key)
                    if container_cls:
                        # 扔进容器，通过 model_dump(mode="json") 自动将内部所有的 Record 实例完美榨干成纯 native
                        native_payload = container_cls(merged_memory).model_dump(mode="json")
                        print("*******************{}*******************".format(native_payload))
                        await self.runtime.store.aput(namespace=self.namespace, key=memory_key, value=native_payload)

                # 3. 原因文本安全包裹
                await self.runtime.store.aput(namespace=self.namespace, key="organization_reason", value=SafeScalarContainer(reason).model_dump(mode="json"))
            except Exception:
                logger.exception("memory commit remote store failed task=%s", task_type)
                raise

        # 进写锁，极速同步本地内存，并清除追踪器
        async with self._lock.writer:
            self._state[memory_key] = merged_memory
            self._state[cursor_key] = cursor
            self._state["organization_reason"] = reason
            
            # 如果追踪器的任务已经顺利落地，将其移出记事本
            if self._processing_cursors.get(task_type) == cursor:
                self._processing_cursors.pop(task_type, None)
                
            logger.info("memory committed successfully. task=%s cursor=%s", task_type, cursor)

    def _to_record_list(self, target: MemoryTarget) -> list:
        state_key = MEMORY_KEY_MAP.get(target)
        if not state_key: 
            return []
        data = self._state.get(state_key)
        if not data: 
            return []
        return list(data.values()) if isinstance(data, dict) else data

    def log_state_summary(self) -> None:
        def _get(k: str, default: object) -> Any:
            return self.state.get(k, default) if isinstance(self.state, dict) else getattr(self.state, k, default)

        p_dict = _get("profile_records", {})
        e_dict = _get("episodic_records", {})
        s_dict = _get("semantic_records", {})
        m_list = self.messages
        
        p_count = len(p_dict) if isinstance(p_dict, dict) else 0
        e_count = len(e_dict) if isinstance(e_dict, dict) else 0
        s_count = len(s_dict) if isinstance(s_dict, dict) else 0
        m_list_len = len(m_list) if isinstance(m_list, list) else 0

        logger.info(
            "🧠 [Memory State] Count -> Profile: %d, Episodic: %d, Semantic: %d | Offset -> Profile: %d, Episodic: %d, Semantic: %d, Messages: %d",
            p_count, e_count, s_count, _get("profile_message_index", 0), _get("episodic_message_index", 0), _get("semantic_message_index", 0), m_list_len
        )


    def _get_utc_timestamp(self, rec: MemoryRecordT) -> datetime:
        if isinstance(rec, str): return datetime.min.replace(tzinfo=timezone.utc)
        dt = None
        if hasattr(rec, 'updated_at'): dt = getattr(rec, 'updated_at')
        elif isinstance(rec, dict):
            val = rec.get('updated_at')
            if isinstance(val, str):
                try: dt = datetime.fromisoformat(val)
                except ValueError: dt = None
            elif isinstance(val, datetime): dt = val

        if dt is None or not isinstance(dt, datetime):
            return datetime.min.replace(tzinfo=timezone.utc)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    def _ensure_utc(self, dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None: return None
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

    def _get_v(self, rec: MemoryRecordT, key: str, default: str = "") -> str:
        if isinstance(rec, dict): return str(rec.get(key, default))
        return str(getattr(rec, key, default))

    def _serialize_to_dict(self, rec: MemoryRecordT, target: MemoryTarget) -> Dict[str, object]:
        updated_at = self._get_utc_timestamp(rec).isoformat()
        meta = {"memory_type": str(target), "updated_at": updated_at}
        if isinstance(rec, str): return {**meta, "raw_data": rec}

        data_dict: Dict[str, object] = {}
        if isinstance(rec, dict): data_dict = rec.copy()
        elif hasattr(rec, 'model_dump'): data_dict = rec.model_dump(mode="json")
        elif hasattr(rec, '__dict__'):
            data_dict = {k: v for k, v in rec.__dict__.items() if not k.startswith('_')}

        for k, v in data_dict.items():
            if isinstance(v, datetime): data_dict[k] = v.isoformat()
        return {**meta, **data_dict}

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

    def export_full_jsonl_stream(self, targets: Optional[List[MemoryTarget]] = None) -> Generator[str, None, None]:
        selected_targets = list(MEMORY_KEY_MAP.keys()) if targets is None else targets
        for target in selected_targets:
            records = self._to_record_list(target)
            for rec in records:
                yield json.dumps(self._serialize_to_dict(rec, target), ensure_ascii=False)

    def export_full_jsonl(self, targets: Optional[List[MemoryTarget]] = None) -> str:
        return "\n".join(self.export_full_jsonl_stream(targets))

    async def get_agent_context(self, config: Optional[AgentContextConfig] = None) -> str:
        """Assemble context with Read Lock."""
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
            if not raw_records: continue

            filtered = [
                r for r in raw_records
                if (not cfg.custom_filters or all(f(r) for f in cfg.custom_filters)) and
                (not start_utc or self._get_utc_timestamp(r) >= start_utc) and
                (not end_utc or self._get_utc_timestamp(r) <= end_utc)
            ]
            if not filtered: continue

            filtered.sort(key=self._get_utc_timestamp, reverse=True)
            sliced = filtered[:cfg.limits.get(target, 50)]

            formatter = formatter_map.get(str(target))
            if formatter:
                sections.append(formatter(sliced))

        return "\n\n".join(sections) if sections else "No memory available."
