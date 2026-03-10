from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from agi.config import AGI_ASSISTANT_ID, AGI_TENANT_ID


@dataclass(frozen=True)
class SessionIdentity:
    tenant_id: str
    assistant_id: str
    user_id: str
    conversation_id: str
    thread_id: str

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SessionIdentity":
        conversation_id = state.get("conversation_id", "")
        return cls(
            tenant_id=state.get("tenant_id", AGI_TENANT_ID),
            assistant_id=state.get("assistant_id", AGI_ASSISTANT_ID),
            user_id=state.get("user_id", "default_tenant"),
            conversation_id=conversation_id,
            thread_id=state.get("thread_id") or conversation_id or str(uuid.uuid4()),
        )


def to_configurable(identity: SessionIdentity, *, need_speech: bool = False) -> dict[str, Any]:
    return {
        "tenant_id": identity.tenant_id,
        "assistant_id": identity.assistant_id,
        "user_id": identity.user_id,
        "conversation_id": identity.conversation_id,
        "thread_id": identity.thread_id,
        "need_speech": need_speech,
    }
