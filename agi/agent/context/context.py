
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from langchain_core.messages import AnyMessage
USER_PROFILE_CONTEXT = "user_profile"

def get_session_context_id(session_id):
    return f'{session_id}_metadata'

def get_session_entity_id(session_id):
    return f'{session_id}_entities'

@dataclass
class Context:
    user_id: str
    conversation_id: str
    messages_cursor: int = 0
    messages: Optional[List[AnyMessage]] = field(default_factory=list)

    def advance_cursor(self, step: Optional[int] = None):
        """
        增加游标位置。
        1. 如果指定 step，则前进 step 位。
        2. 如果不指定 step，则直接对齐到当前消息列表的末尾。
        """
        current_msg_count = len(self.messages)
        
        if step is None:
            self.messages_cursor = current_msg_count
        else:
            new_position = self.messages_cursor + step
            # 边界保护：游标不能超过消息总数
            self.messages_cursor = min(new_position, current_msg_count)
            
        print(f"--- [Cursor] 游标向前推进至索引: {self.messages_cursor} ---")

    def reset_cursor_to_start(self):
        """将游标重置为起点 (0)"""
        self.messages_cursor = 0
        print(f"--- [Cursor] 游标已重置为 0 ---")

    def rollback_cursor(self, step: int):
        """
        回滚游标。
        适用于认知进化失败或消息回滚（Undo）场景。
        """
        new_position = self.messages_cursor - step
        # 边界保护：游标不能小于 0
        self.messages_cursor = max(0, new_position)
        print(f"--- [Cursor] 游标回退至索引: {self.messages_cursor} ---")

    def set_incremental_messages(self,messages:List[AnyMessage]):
        """
        获取从游标位置到当前的“增量消息切片”。
        用于子 Agent 提取长期记忆。
        """
        self.messages = messages[self.messages_cursor:]

    @property
    def has_pending_messages(self) -> bool:
        """检查是否有未被游标覆盖（未处理）的消息"""
        return len(self.messages) > self.messages_cursor
    
class UserPersona(BaseModel):
    full_name: Optional[str] = None
    job_role: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    communication_style: str = "professional"
    technical_level: Dict[str, str] = Field(default_factory=dict)
    current_goals: List[str] = Field(default_factory=list)

class SessionState(BaseModel):
    current_intent: Optional[str] = None
    user_emotion: str = "neutral"
    topic_drift: bool = False

class Entity(BaseModel):
    name: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class UnifiedUpdateResult(BaseModel):
    user_profile: UserPersona
    session_state: SessionState
    new_entities: List[Entity] = Field(default_factory=list)

def merge_dict(old: dict, new: dict):
    merged = old.copy()
    for k, v in new.items():
        if v not in [None, "", [], {}]:
            merged[k] = v
    return merged

class BackendMixin:
    def _load_json(self, data, default):
        import json
        if not data:
            return default
        if isinstance(data, (dict, list)):
            return data
        try:
            return json.loads(data)
        except Exception:
            return default

