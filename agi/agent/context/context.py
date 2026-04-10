from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

USER_PROFILE_CONTEXT = "user_profile"

def get_session_context_id(session_id):
    return f'{session_id}_metadata'

def get_session_entity_id(session_id):
    return f'{session_id}_entities'

@dataclass
class Context:
    user_id: str
    conversation_id: str

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

