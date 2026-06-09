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
    """结合强类型、策略路由与极简 Token 提取的生产级内存服务"""

    def __init__(self, graph: CompiledStateGraph, thread_id: str):
        self.graph = graph
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        
        # 初始化时直接完成快照获取与强类型映射
        self.refresh()

    def refresh(self) -> None:
        """刷新状态快照并强制映射为 MemoryState"""
        snapshot = self.graph.get_state(self.config)
        raw_values = snapshot.values if hasattr(snapshot, "values") else snapshot
        
        # 🛡️ 核心映射：将 runtime dict 映射为强类型契约
        self.state: MemoryState = cast(MemoryState, raw_values)

    def get_memories(self) -> MemoryState:
        """
        不再返回 tuple，直接返回强类型化的 MemoryState 对象。
        后续逻辑通过 typed_state.get('profile_records') 访问，获得完整补全。
        """
        return self.state
    
    def update_state(self, key: MemoryStateKey, value: Any) -> None:
        """
        向图引擎提交符合 MemoryState 契约的状态更新。
        🛡️ 防御性检查：仅允许更新 MemoryState 中定义的合法字段。
        """
        if key not in ALLOWED_MEMORY_KEYS:
            logger.error(f"❌ 非法状态键访问尝试: {key}。仅允许更新: {ALLOWED_MEMORY_KEYS}")
            raise ValueError(f"Invalid state key: {key}")

        # 🌟 直接调用，LangGraph 会自动匹配该字段关联的 memory_reducer 或 LastValue
        self.graph.update_state(self.config, {key: value})
        
        # 写入后同步更新本地快照，确保 MemoryManager 状态即时最新
        self.state[key] = value

    def update_memory_index(self, task_type: MemoryTarget, value: int) -> None:
        """更新指定维度的索引，复用细化的 update_state 逻辑"""
        state_key = INDEX_KEY_MAP.get(task_type)
        if state_key:
            # 这里调用上面重构后的 update_state，获得强类型保护
            self.update_state(cast(MemoryStateKey, state_key), value)

    def get_memory_index(self, task_type: MemoryTarget) -> int:
        """
        根据任务类型获取当前记忆的消费偏移量。
        消除了硬编码的 if-else，使用映射表统一管理。
        """
        # 1. 从映射表中获取对应的状态键 (例如 'profile_message_index')
        state_key = INDEX_KEY_MAP.get(task_type)
        
        if not state_key:
            logger.warning(f"⚠️ 未知的任务类型: {task_type}，返回默认索引 0")
            return 0
            
        # 2. 从当前状态快照中读取，不存在时默认为 0
        # 这里的 self.state 已经是 cast 后的 MemoryState 类型
        return self.state.get(cast(MemoryStateKey, state_key), 0)
    # -------------------------------------------------
    # 底层私有工具链
    # -------------------------------------------------
    def _to_record_list(self, target: MemoryTarget) -> List[Any]:
        state_key = MEMORY_KEY_MAP[target]
        # 兼容 Pydantic BaseConfig、TypedDict 或标准 Class 对象的属性读取
        data = self.state.get(state_key) if isinstance(self.state, dict) else getattr(self.state, state_key, None)
            
        if not data:
            return []
        if isinstance(data, dict):
            return list(data.values())
        if isinstance(data, list):
            return data
        return []

    def _get_utc_timestamp(self, rec: Any) -> datetime:
        """安全提取时间戳用于排序"""
        if isinstance(rec, str): 
            return datetime.min.replace(tzinfo=timezone.utc)
            
        dt: Any = None
        if hasattr(rec, 'updated_at'):
            dt = rec.updated_at
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

    def _get_v(self, rec: Any, key: str, default: str = "") -> str:
        """【唯一核心工具】依照类型契约，平铺提取对象或字典的属性值，替代 hasattr/getattr 嵌套"""
        if isinstance(rec, dict):
            return rec.get(key, default)
        return getattr(rec, key, default)

    def _serialize_to_dict(self, rec: Any, target: MemoryTarget) -> Dict[str, Any]:
        """场景 1 专用：将任意 Record 序列化为平铺的标准 Dict"""
        updated_at = self._get_utc_timestamp(rec).isoformat()
        meta = {"memory_type": target, "updated_at": updated_at}
        
        if isinstance(rec, str):
            return {**meta, "raw_data": rec}
            
        data_dict = {}
        if isinstance(rec, dict):
            data_dict = rec.copy()
        elif hasattr(rec, '__dict__'):
            data_dict = {k: v for k, v in rec.__dict__.items() if not k.startswith('_')}
            
        for k, v in data_dict.items():
            if isinstance(v, datetime):
                data_dict[k] = v.isoformat()
                
        return {**meta, **data_dict}

    # -------------------------------------------------
    # 【极致 Token 节省排版策略】严格仅提取必要字段
    # -------------------------------------------------
    def _format_profile(self, records: List[Any]) -> str:
        lines = ["--- PROFILE MEMORY ---"]
        for rec in records:
            if isinstance(rec, str): 
                lines.append(f"- {rec}")
            else:
                k = self._get_v(rec, 'key')
                v = self._get_v(rec, 'value')
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _format_episodic(self, records: List[Any]) -> str:
        lines = ["--- EPISODIC MEMORY ---"]
        for rec in records:
            if isinstance(rec, str): 
                lines.append(f"- {rec}")
            else:
                t = self._get_v(rec, 'event_time', 'Unknown time')
                s = self._get_v(rec, 'summary')
                lines.append(f"- [{t}] {s}")
        return "\n".join(lines)

    def _format_semantic(self, records: List[Any]) -> str:
        compact_data = []
        for rec in records:
            if isinstance(rec, str): 
                compact_data.append({"raw_fact": rec})
            else:
                compact_data.append({
                    "s": self._get_v(rec, 'subject'),
                    "p": self._get_v(rec, 'predicate'),
                    "o": self._get_v(rec, 'object')
                })
        return f"--- SEMANTIC MEMORY ---\n{json.dumps(compact_data, ensure_ascii=False)}"

    # =====================================================
    # 场景 1 生产接口：全量提取 JSONL 数据 (离线任务/辅助整理)
    # =====================================================
    def export_full_jsonl_stream(self, targets: Optional[List[MemoryTarget]] = None) -> Generator[str, None, None]:
        # 如果未指定，自动拉取当前支持的所有类型
        selected_targets = list(MEMORY_KEY_MAP.keys()) if targets is None else targets
        for target in selected_targets:
            records = self._to_record_list(target)
            for rec in records:
                yield json.dumps(self._serialize_to_dict(rec, target), ensure_ascii=False)

    def export_full_jsonl(self, targets: Optional[List[MemoryTarget]] = None) -> str:
        return "\n".join(self.export_full_jsonl_stream(targets))

    # =====================================================
    # 场景 2 生产接口：在线 Agent 上下文组装 (带动态过滤/裁剪)
    # =====================================================
    def get_agent_context(self, config: Optional[AgentContextConfig] = None) -> str:
        """
        生成严格控制上下文 Token 的格式化字符串。
        通过流式过滤 -> 时间倒序 -> 安全切片 -> 动态策略分发
        """
        cfg = config or AgentContextConfig()
        sections: List[str] = []
        
        start_utc = self._ensure_utc(cfg.start_time)
        end_utc = self._ensure_utc(cfg.end_time)

        # 如果 Config 中未指定 targets，默认拉取所有类型
        selected_targets = list(MEMORY_KEY_MAP.keys()) if cfg.targets is None else cfg.targets

        for target in selected_targets:
            raw_records = self._to_record_list(target)
            if not raw_records:
                continue

            # 1. 统一管道过滤
            filtered = [
                r for r in raw_records
                if (not cfg.custom_filters or all(f(r) for f in cfg.custom_filters)) and
                (not start_utc or self._get_utc_timestamp(r) >= start_utc) and
                (not end_utc or self._get_utc_timestamp(r) <= end_utc)
            ]
            if not filtered:
                continue

            # 2. 排序与安全切片
            filtered.sort(key=self._get_utc_timestamp, reverse=True)
            sliced = filtered[:cfg.limits.get(target, 50)]

            # 3. 动态路由排版分发
            formatter = getattr(self, f"_format_{target}", None)
            if formatter:
                sections.append(formatter(sliced))

        return "\n\n".join(sections) if sections else "No memory available."