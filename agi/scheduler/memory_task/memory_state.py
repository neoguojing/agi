from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, TypeVar, Annotated,Union
from typing import cast,get_args
from langgraph.channels import LastValue
from aiorwlock import RWLock
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
    SummaryRecord,
    MemoryTarget,
    CONTAINER_MAPPING,
    SafeScalarContainer
)
from agi.scheduler.memory_task.runtime import MemoryTaskRuntime,memory_runtime
from langchain_core.messages import HumanMessage,AnyMessage,convert_to_messages

logger = logging.getLogger(__name__)

# --- Constants and Reducers ---

def memory_reducer(
    state: Optional[dict[str, Any]],
    writes: Union[list[Any], dict[str, Optional[Any]]]
) -> dict[str, Any]:
    """
    A high-performance generic reducer for structured memories (Profile, Episodic, Semantic).
    """
    merged: dict[str, Any] = state.copy() if state else {}
    if isinstance(writes, list):
        for r in writes:
            if hasattr(r, "dedup_key") and r.dedup_key:
                merged[r.dedup_key] = r

    elif isinstance(writes, dict):
        for k, r in writes.items():
            if r is None:
                merged.pop(k, None)
            else:
                real_k = r.dedup_key if hasattr(r, "dedup_key") and r.dedup_key else k
                merged[real_k] = r

    return merged

class MemoryState(AgentState[ResponseT]):
    """
    State schema for the memory organization middleware.

    Separates 'Structured Memories' (using memory_reducer) from 'Conversation Summary' (using LastValue).
    """
    # 1. Structured Memories: Use reducer for delta updates
    profile_records: Annotated[NotRequired[dict[str, ProfileMemoryRecord]], memory_reducer]
    episodic_records: Annotated[NotRequired[dict[str, EpisodicMemoryRecord]], memory_reducer]
    semantic_records: Annotated[NotRequired[dict[str, SemanticMemoryRecord]], memory_reducer]

    # 2. Conversation Summary: Use LastValue for snapshot overwrite.
    # Only the most recent summary record for the current thread is stored here.
    summary_record: Annotated[NotRequired[SummaryRecord | None], LastValue]

    organization_reason: Annotated[NotRequired[str], LastValue]
    profile_message_index: Annotated[NotRequired[int], LastValue]
    episodic_message_index: Annotated[NotRequired[int], LastValue]
    semantic_message_index: Annotated[NotRequired[int], LastValue]

# Only mappings for structured memories
MEMORY_KEY_MAP: Dict[MemoryTarget, str] = {
    "profile": "profile_records",
    "episodic": "episodic_records",
    "semantic": "semantic_records",
}

MemoryRecordT = Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord, SummaryRecord, dict]
MemoryFilterFn = Callable[[Any], bool]

MemoryStateKey = Literal[
    "profile_records",
    "episodic_records",
    "semantic_records",
    "summary_record",
    "organization_reason",
    "profile_message_index",
    "episodic_message_index",
    "semantic_message_index",
]

ALLOWED_MEMORY_KEYS = set(get_args(MemoryStateKey))

INDEX_KEY_MAP: Dict[MemoryTarget, str] = {
    "profile": "profile_message_index",
    "episodic": "episodic_message_index",
    "semantic": "semantic_message_index",
}

MEMORY_KEY = "memories"

# --- Context Config ---

@dataclass
class AgentContextConfig:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    limits: Dict[MemoryTarget, int] = field(default_factory=lambda: {
        "profile": 50,
        "episodic": 30,
        "semantic": 50,
    })

    custom_filters: List[MemoryFilterFn] = field(default_factory=list)
    targets: Optional[List[MemoryTarget]] = None

# --- Core Manager ---

class MemoryManager:
    CURSOR_RECOVERY_RESET = "reset"
    CURSOR_RECOVERY_CLAMP = "clamp"
    CURSOR_RECOVERY_MODE = CURSOR_RECOVERY_CLAMP

    def __init__(self, runtime: MemoryTaskRuntime):
        self.runtime = runtime
        self._state: Dict[str, Any] = {}
        self._messages: List[Any] = []
        self._lock = RWLock()
        self._is_initialized = False
        self._processing_cursors: Dict[str, int] = {}

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

    async def refresh_messages(self) -> list:
        local_messages = []
        if self.runtime.client:
            snapshot = await self.runtime.client.threads.get_state(thread_id=self.runtime.thread_id)
            state = snapshot.get("values") if isinstance(snapshot, dict) else getattr(snapshot, "values", {})
            local_messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        elif self.runtime.graph:
            snapshot = await self.runtime.graph.aget_state(self.config)
            state = snapshot.values if hasattr(snapshot, "values") else snapshot
            local_messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])

        async with self._lock.writer:
            self._messages = local_messages
        return self._messages

    async def ensure_initialized(self) -> None:
        if self._is_initialized:
            return

        async with self._lock.writer:
            if self._is_initialized:
                return

            logger.info("Initializing memory manager from remote store... thread_id=%s", self.runtime.thread_id)
            local_state_updates = {}
            keys_to_load = ALLOWED_MEMORY_KEYS

            if self.runtime.store:
                for key in keys_to_load:
                    try:
                        # Note: Summary records are stored as specific keys per thread in store
                        # aget for "summary_record" will be handled as a generic load
                        raw_data = await self.runtime.store.aget(namespace=self.namespace, key=key)
                        if raw_data is not None:
                            container_cls = CONTAINER_MAPPING.get(key)
                            if container_cls:
                                container_instance = container_cls.model_validate(raw_data.value)
                                local_state_updates[key] = container_instance.root
                            else:
                                local_state_updates[key] = raw_data
                    except Exception as e:
                        logger.warning("Failed to load key '%s': %s", key, str(e))
                        if "index" in key: local_state_updates[key] = 0
                        else: local_state_updates[key] = {}

            self._state.update(local_state_updates)
            self._is_initialized = True
            self.log_state_summary()

    async def repair_memory_index(self, task_type: MemoryTarget) -> int:
        cursor_key = INDEX_KEY_MAP.get(task_type)
        if not cursor_key: return 0
        async with self._lock.reader:
            message_count = len(self._messages)
            cursor = self._state.get(cursor_key, 0)
        if cursor <= message_count: return cursor
        repaired_cursor = 0 if self.CURSOR_RECOVERY_MODE == self.CURSOR_RECOVERY_RESET else message_count
        if self.runtime.store:
            try:
                await self.runtime.store.aput(namespace=self.namespace, key=cursor_key, value=SafeScalarContainer(repaired_cursor).model_dump(mode="json"))
            except Exception: logger.exception("cursor repair failed")
        async with self._lock.writer:
            self._state[cursor_key] = repaired_cursor
        return repaired_cursor

    async def get_incremental_messages(self, task_type: MemoryTarget) -> list:
        await self.ensure_initialized()
        await self.refresh_messages()
        cursor = await self.repair_memory_index(task_type)
        async with self._lock.writer:
            current_len = len(self._messages)
            if cursor >= current_len:
                self._processing_cursors[task_type] = current_len
                return []
            incremental = self._messages[cursor:current_len]
            self._processing_cursors[task_type] = current_len
            return incremental.copy()

    # ==================== Summary-Specific State Management (Isolated) ====================

    async def get_current_summary(self) -> Optional[SummaryRecord]:
        """
        Get the current summary record for the current thread.
        Isolated from structured memory targets.
        """
        await self.ensure_initialized()
        async with self._lock.reader:
            # The state should contain the summary record for the current active thread
            return self._state.get("summary_record")

    async def set_current_summary(self, record: SummaryRecord) -> None:
        """
        Directly set the latest summary snapshot.
        Implements LastValue behavior: overwrite regardless of previous state.
        """
        await self.ensure_initialized()

        async with self._lock.writer:
            # Update local state
            self._state["summary_record"] = record

        # Persist to remote store as a thread-specific snapshot
        if self.runtime.store:
            try:
                # Summary is not a dict-of-records, it's a single record per thread.
                # We store it under a thread-specific key to keep global store clean.
                # Use SummaryContainer to wrap the record if needed,
                # but since it's a single record, we can just dump it if it's a Pydantic model.
                native_val = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
                await self.runtime.store.aput(namespace=self.namespace, key="summary_record", value=native_val)
            except Exception as e:
                logger.exception("summary snapshot persist failed {e}")


    def _build_summary_message(self,summary: str, file_path: Optional[str]) -> List[AnyMessage]:
        if file_path is not None:
            content = f"You are in the middle of a conversation that has been summarized.\n\nThe full conversation history has been saved to {file_path} should you need to refer back to it for details.\n\nA condensed summary follows:\n\n<summary>\n{summary}\n</summary>"
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"
        return [HumanMessage(content=content, additional_kwargs={"lc_source": "summarization"})]

    async def get_effective_summary_context(self) -> list:
        """
        Assemble [SummaryMsg + Incremental Messages].
        """
        await self.ensure_initialized()
        await self.refresh_messages()
        if not self._messages: return []

        full_messages = convert_to_messages(self._messages.copy())
        first_msg = full_messages[0]
        if isinstance(first_msg, HumanMessage) and getattr(first_msg, "additional_kwargs", {}).get("lc_source") == "summarization":
            return full_messages

        summary_rec = await self.get_current_summary()
        if not summary_rec:
            return full_messages

        cutoff = getattr(summary_rec, "cutoff_index", 0)
        summary = getattr(summary_rec, "summary", None)
        file_path = getattr(summary_rec, "file_path", 0)
        summary_msg = self._build_summary_message(summary,file_path)
        return summary_msg + full_messages[cutoff:]

    # ==================== Structured Memory Flow ====================

    async def commit_incremental_memory(self, task_type: MemoryTarget, memory_value: object, reason: str) -> None:
        await self.ensure_initialized()
        async with self._lock.reader:
            target_cursor = self._processing_cursors.get(task_type)
        if target_cursor is None:
            async with self._lock.reader:
                cursor_key = INDEX_KEY_MAP.get(task_type)
                target_cursor = self._state.get(cursor_key, 0)
        await self.commit_memory(task_type=task_type, memory_value=memory_value, cursor=target_cursor, reason=reason)

    async def commit_memory(self, task_type: MemoryTarget, memory_value: object, cursor: int, reason: str) -> None:
        cursor_key = INDEX_KEY_MAP.get(task_type)
        memory_key = MEMORY_KEY_MAP.get(task_type)
        if not cursor_key or not memory_key:
            raise ValueError(f"cursor or memory key not found for task: {task_type}")

        async with self._lock.reader:
            current_cursor = self._state.get(cursor_key, 0)
            current_memory = self._state.get(memory_key, {})

        if cursor <= current_cursor: return

        merged_memory = memory_reducer(current_memory, memory_value) if memory_key in MEMORY_KEY_MAP.values() else memory_value

        if self.runtime.store:
            try:
                await self.runtime.store.aput(namespace=self.namespace, key=cursor_key, value=SafeScalarContainer(cursor).model_dump(mode="json"))
                if memory_value:
                    container_cls = CONTAINER_MAPPING.get(memory_key)
                    if container_cls:
                        native_payload = container_cls(merged_memory).model_dump(mode="json")
                        await self.runtime.store.aput(namespace=self.namespace, key=memory_key, value=native_payload)
                await self.runtime.store.aput(namespace=self.namespace, key="organization_reason", value=SafeScalarContainer(reason).model_dump(mode="json"))
            except Exception: logger.exception("memory commit failed")

        async with self._lock.writer:
            self._state[memory_key] = merged_memory
            self._state[cursor_key] = cursor
            self._state["organization_reason"] = reason
            if self._processing_cursors.get(task_type) == cursor:
                self._processing_cursors.pop(task_type, None)

    def _to_record_list(self, target: MemoryTarget) -> list:
        state_key = MEMORY_KEY_MAP.get(target)
        if not state_key: return []
        data = self._state.get(state_key)
        return list(data.values()) if isinstance(data, dict) else (data or [])

    def log_state_summary(self) -> None:
        def _get(k: str, default: object) -> Any:
            return self.state.get(k, default) if isinstance(self.state, dict) else getattr(self.state, k, default)
        
        p_dict = _get("profile_records", {})
        e_dict = _get("episodic_records", {})
        s_dict = _get("semantic_records", {})
        sum_rec = _get("summary_record", None)
        m_list = self.messages

        # --- 🔍 智能解析 SummaryRecord 的完整内容 ---
        if sum_rec:
            if hasattr(sum_rec, "model_dump_json"): # Pydantic v2
                try:
                    # 转换为美化的 JSON 字符串
                    sum_content = json.dumps(json.loads(sum_rec.model_dump_json()), indent=4, ensure_ascii=False)
                except Exception:
                    sum_content = str(sum_rec)
            elif hasattr(sum_rec, "dict"): # Pydantic v1
                try:
                    sum_content = json.dumps(sum_rec.dict(), indent=4, ensure_ascii=False)
                except Exception:
                    sum_content = str(sum_rec)
            elif isinstance(sum_rec, dict):
                sum_content = json.dumps(sum_rec, indent=4, ensure_ascii=False)
            else:
                sum_content = str(sum_rec)
        else:
            sum_content = "None"

        # --- 🚨 高显眼度、易读性多行日志排版 ---
        log_message = (
            "\n"
            "==================================================================================================\n"
            "🧠 [MEMORY STATE SNAPSHOT]\n"
            "--------------------------------------------------------------------------------------------------\n"
            f"📊 [RECORD COUNTS]  => Profile: {len(p_dict)} | Episodic: {len(e_dict)} | Semantic: {len(s_dict)} | Total Messages: {len(m_list)}\n"
            f"📍 [MESSAGE INDEX]  => Profile: {_get('profile_message_index', 0)} | Episodic: {_get('episodic_message_index', 0)} | Semantic: {_get('semantic_message_index', 0)}\n"
            "--------------------------------------------------------------------------------------------------\n"
            f"📝 [SUMMARY RECORD FULL DETAIL]:\n"
            f"{sum_content}\n"
            "=================================================================================================="
        )
    
        logger.info(log_message)

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
            data_dict = {k: v for k, v in rec.__dict__.copy() if not k.startswith('_')}
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
                rec_json_dict = {"id": self._get_v(rec, 'id'), "event_time": str(self._get_v(rec, 'event_time')), "summary": self._get_v(rec, 'summary')}
                lines.append(json.dumps(rec_json_dict, ensure_ascii=False))
            except Exception as e: logger.warning(" la... error: %s", str(e))
        return "\n".join(lines)

    def _format_semantic(self, records: List[MemoryRecordT]) -> str:
        lines = [f"--- SEMANTIC MEMORY ({len(records)}) ---"]
        for rec in records:
            item = {"subject": self._get_v(rec, 'subject'), "predicate": self._get_v(rec, 'predicate'), "object": self._get_v(rec, 'object')}
            lines.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(lines)

    def export_full_jsonl_stream(self, targets: Optional[List[MemoryTarget]] = None) -> Iterator[str]:
        selected_targets = list(MEMORY_KEY_MAP.keys()) if targets is None else targets
        for target in selected_targets:
            records = self._to_record_list(target)
            for rec in records:
                yield json.dumps(self._serialize_to_dict(rec, target), ensure_ascii=False)

    def export_full_jsonl(self, targets: Optional[List[MemoryTarget]] = None) -> str:
        return "\n".join(self.export_full_jsonl_stream(targets))

    async def get_agent_context(self, config: Optional[AgentContextConfig] = None) -> str:
        cfg = config or AgentContextConfig()
        sections: List[str] = []
        start_utc = self._ensure_utc(cfg.start_time)
        end_utc = self._ensure_utc(cfg.end_time)
        selected_targets = list(MEMORY_KEY_MAP.keys()) if cfg.targets is None else cfg.targets
        formatter_map: Dict[str, Callable[[List[MemoryRecordT]], str]] = {
            "profile": self._format_profile,
            "episodic": self._format_episodic,
            "semantic": self._format_semantic,
        }
        for target in selected_targets:
            raw_records = self._to_record_list(target)
            if not raw_records: continue
            filtered = [r for r in raw_records if (not cfg.custom_filters or all(f(r) for f in cfg.custom_filters)) and (not start_utc or self._get_utc_timestamp(r) >= start_utc) and (not end_utc or self._get_utc_timestamp(r) <= end_utc)]
            if not filtered: continue
            filtered.sort(key=self._get_utc_timestamp, reverse=True)
            sliced = filtered[:cfg.limits.get(target, 50)]
            formatter = formatter_map.get(str(target))
            if formatter: sections.append(formatter(sliced))
        return "\n\n".join(sections) if sections else "No memory available."
    
memory_manager = MemoryManager(memory_runtime)
