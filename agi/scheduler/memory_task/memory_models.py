"""Typed memory records and patch primitives.

Usage:
    These classes define the structured contract between:
    - LLM memory extraction
    - memory maintenance tasks
    - storage backends

    The schema is optimized for:
    - structured LLM output (Flat and Simple)
    - JSON serialization
    - patch-based memory updates
    - graph/vector memory systems
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator, field_serializer, RootModel, model_serializer, model_validator

# =========================================================
# Type Aliases
# =========================================================

class BaseMemoryRecord(BaseModel):
    pass

MemoryTarget = Literal["profile", "episodic", "semantic", "summary"]

# =========================================================
# Profile Memory
# =========================================================

class ProfileMemoryRecord(BaseMemoryRecord):

    key: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Profile attribute key "
            "(example: 'favorite_language', 'job_title'). "
            "Must not be empty."
        )
    )

    value: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Profile attribute value. "
            "Must not be empty."
        )
    )

    confidence: float = Field(
        default=0.5,
        description=(
            "Option. Confidence score between "
            "0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

    @property
    def dedup_key(self) -> str:
        """Unique key for deduplication and updates."""
        return self.key.strip().lower() if self.key else ""

# =========================================================
# Episodic Memory (Simplified for LLM)
# =========================================================

class EpisodicMemoryRecord(BaseMemoryRecord):

    id: str = Field(
        default_factory=lambda: f"ep_{uuid4().hex[:8]}",
        description="Unique identifier for the memory record. Permanent and immutable."
    )

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "REQUIRED. Short summary of the event. "
            "Must not be empty."
        )
    )

    event_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description=(
            "Option. Event occurrence time "
            "in ISO datetime string format. "
            "Must not be empty."
        )
    )

    # =====================================================
    # 1. 入向拦截（Validation 防御）：把各种妖魔鬼怪的输入转为 datetime
    # =====================================================
    @field_validator('event_time', mode='before')
    @classmethod
    def normalize_datetime(cls, v: Any) -> Any:
        # 情况 A：如果已经是 datetime 对象（比如从内部代码直接传入）
        if isinstance(v, datetime):
            # 强行补充时区，防止 naive datetime 报错
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v

        # 情况 B：如果是 Unix 时间戳（int 或 float，比如来自某些缓存或数据库）
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)

        # 情况 C：如果是字符串（最常遇到，比如大模型输出或 JSON 文件）
        if isinstance(v, str):
            v = v.strip()
            # 兜底：处理非法的空字符串占位符
            if not v or v.lower() in ('unknown time', 'none', 'null', ''):
                return datetime.now(timezone.utc)

            try:
                # 尝试标准的 ISO 格式解析（兼容带 Z 或不带 Z 的格式）
                normalized_str = v.replace('Z', '+00:00')
                dt = datetime.fromisoformat(normalized_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                # 极端兜底：如果大模型胡乱写了一个非 ISO 格式（如 "2026/06/10 10:00"）
                try:
                    from dateutil import parser
                    dt = parser.parse(v)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    # 实在解析不了，用当前时间兜底，保证系统不死
                    return datetime.now(timezone.utc)

        # 其他完全无法识别的类型，直接交给 Pydantic 原生报错，或者直接兜底
        return v

    # =====================================================
    # 2. 出向规范（Serialization 统一）：确保吐出给外面的一定是标准 ISO 字符串
    # =====================================================
    @field_serializer('event_time')
    def serialize_datetime(self, v: Any, _info) -> str:
        # 情况 A：如果是完美的 datetime 对象，正常序列化
        if isinstance(v, datetime):
            return v.isoformat().replace('+00:00', 'Z')

        # 情况 B：如果由于某些黑魔法（如 reducer 合并、手动赋值）它已经变成了 str
        if isinstance(v, str):
            # 顺手帮它把时区尾缀标准化，防止大模型看着难受
            return v.strip().replace('+00:00', 'Z')

        # 情况 C：极端兜底
        return str(v)

    participants: list[str] = Field(
        default_factory=list,
        description=(
            "Option. List of entities or people "
            "involved in the event. "
            "List must not be empty."
        )
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Option. Confidence score "
            "between 0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

    @property
    def dedup_key(self) -> str:
        # 🌟 现在的去重/检索主键直接绑定 ID
        return self.id

# =========================================================
# Semantic Memory (Simplified for LLM)
# =========================================================

AgentMemoryPredicate = Literal[
    "is_a",
    "same_as",
    "likes",
    "prefers",
    "uses",
    "owns",
    "knows",
    "member_of",
    "works_at",
    "created",
    "located_in",
    "learned",
    "depends_on",
    "related_to",
    "interested_in"
]

class SemanticMemoryRecord(BaseMemoryRecord):

    subject: str = Field(
        ...,
        min_length=1,
        max_length=40,
        pattern=r"^[^.。!?！？\n]+$",  # 🛑 禁用句号和换行，逼迫它输出短语
        description=(
            "REQUIRED. Canonical subject entity name (e.g., 'OpenAI', 'Python'). "
            "Must be a single noun or short noun-phrase. MAXIMUM 40 characters. NO sentences."
        )
    )

    predicate: AgentMemoryPredicate = Field(
        ...,
        description=(
            "REQUIRED. The relationship link verb between subject and object. "
            "MUST be one of these exact tokens: 'is_a', 'same_as', 'likes', 'prefers', "
            "'uses', 'owns', 'knows', 'member_of', 'works_at', 'created', 'located_in', "
            "'learned', 'depends_on', 'related_to','interested_in'."
        )
    )

    object: str = Field(
        ...,
        min_length=1,
        max_length=50,  # 🛑 宾语通常可能稍长（如特定概念），但也必须限制
        pattern=r"^[^.。!?！？\n]+$",
        description=(
            "REQUIRED. Target entity, concept, or literal value. "
            "Examples: 'San_Francisco', 'v4.0', 'Artificial_Intelligence'."
        )
    )

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Option. Confidence score "
            "between 0.0 and 1.0."
        )
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="creation timestamp."
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="update timestamp."
    )

    @property
    def dedup_key(self) -> str:
        subject = self.subject.strip().lower() if getattr(self, "subject", None) else ""
        predicate = getattr(self, "predicate", "")
        obj = self.object.strip().lower() if getattr(self, "object", None) else ""
        return f"{subject}:{predicate}:{obj}" if subject and obj else ""

# =========================================================
# Summary Memory (Added for event_tasks)
# =========================================================

class SummaryRecord(BaseMemoryRecord):
    summary: str = Field(...)
    source_conversation_id: str = Field(...) # Reference to the conversation that triggered this
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    cutoff_index: int = Field(default=0, description="The index in the original history where truncation occurred.")
    new_messages: list[AnyMessage] = Field(default_factory=list, description="The summarized conversation flow.")

    @property
    def dedup_key(self) -> str:
        return self.source_conversation_id

# =========================================================
# 🌟 终极优雅：定义强类型存储容器 (将黑魔法封装在内)
# =========================================================

class ProfileContainer(RootModel[Dict[str, ProfileMemoryRecord]]):
    """接管整个 Profile 字典的导入吐出"""
    pass

class EpisodicContainer(RootModel[Dict[str, EpisodicMemoryRecord]]):
    """接管整个 Episodic 字典的导入吐出"""
    pass

class SemanticContainer(RootModel[Dict[str, SemanticMemoryRecord]]):
    """接管整个 Semantic 字典的导入吐出"""
    pass

class SummaryContainer(RootModel[Dict[str, SummaryRecord]]):
    """接管整个 Summary 字典的导入吐出"""
    pass

class SafeScalarContainer(RootModel[Any]):
    """
    专门用来包装游标数字、原因文本等 LangGraph 底层 orjson 无法直接解析的标量。
    写入时自动包成 {"root": value}，读取时自动解包。
    """

    # 1. 读入（反序列化）时触发：拦截输入数据进行拆箱
    @model_validator(mode='before')
    @classmethod
    def unwrap_root(cls, data: Any) -> Any:
        # 如果读取到的是我们包装过的字典格式 {"root": value}，则直接把 value 提取出来
        if isinstance(data, dict) and "root" in data and len(data) == 1:
            return data["root"]
        return data

    # 2. 写入（序列化）时触发：强制装箱成字典
    @model_serializer
    def wrap_into_dict(self) -> dict[str, Any]:
        # 不论内部包裹的是 int, str 还是其他标量，输出给 orjson 时一律包一层字典
        return {"root": self.root}

    # 可选：重写 __str__ 或 __repr__ 让调试打印更直观
    def __str__(self):
        return str(self.root)

# 全局映射字典：让底层 Manager 瞬间看懂如何处理数据
CONTAINER_MAPPING = {
    "profile_records": ProfileContainer,
    "episodic_records": EpisodicContainer,
    "semantic_records": SemanticContainer,
    "summary_records": SummaryContainer,
    "profile_message_index": SafeScalarContainer,
    "episodic_message_index": SafeScalarContainer,
    "semantic_message_index": SafeScalarContainer,
    "organization_reason": SafeScalarContainer,
}
