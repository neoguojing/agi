import asyncio
import json
import logging
import traceback
from typing import List, Dict, Optional, Any

from pydantic import BaseModel, Field
from deepagents.backends.protocol import BackendProtocol

from .context import (
    USER_PROFILE_CONTEXT,
    get_session_context_id,
    get_session_entity_id,
    BackendMixin,
)
from agi.utils.common import extract_messages_content

logger = logging.getLogger(__name__)

# =========================
# Prompt Constants
# =========================

PROMPT_USER_PROFILE = """
Task: Update the User's persistent profile based on the latest interaction.
Context:
- Existing Profile: {current_val}
- User Message: {user_msg}
- AI Response: {ai_msg}

Instructions:
1. Extract ONLY long-term, factual, or preference-based information.
2. Ignore transient session-specific requests.
3. Keep 'technical_level' updated based on demonstrated knowledge.
4. Output should strictly follow the UserPersona schema.
"""

PROMPT_SESSION_STATE = """
Task: Analyze the immediate state and intent of the current conversation turn.
User Message: {user_msg}

Instructions:
- current_intent: The specific action the user wants to achieve NOW.
- user_emotion: Detect tone.
- topic_drift: true ONLY if user changes subject abruptly.
"""

PROMPT_ENTITY_EXTRACTION = """
Task: Identify and extract named entities mentioned in the dialogue.
Context:
- User Message: {user_msg}
- AI Response: {ai_msg}

Instructions:
- Extract: Projects, Order IDs, Tools, Organizations, People.
- Assign type
- Include attributes if present
"""

# =========================
# Models
# =========================

class UserPersona(BaseModel):
    full_name: Optional[str] = None
    job_role: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    communication_style: str = "professional"
    technical_level: Dict[str, str] = Field(default_factory=dict)
    current_goals: List[str] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    current_intent: Optional[str] = None
    user_emotion: Optional[str] = None
    topic_drift: bool = False


class Entity(BaseModel):
    name: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)


class EntityList(BaseModel):
    entities: List[Entity] = Field(default_factory=list)


# =========================
# Core Updater
# =========================

class UnifiedContextUpdater(BackendMixin):
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend

        self.user_extractor = model.with_structured_output(UserPersona)
        self.session_extractor = model.with_structured_output(SessionUpdate)
        self.entity_extractor = model.with_structured_output(EntityList)

    # ---------- backend ----------
    def _get_backend(self, runtime) -> Optional[BackendProtocol]:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    # ---------- entry ----------
    async def update(self, runtime, messages, ai_response):
        try:
            if not messages or not ai_response:
                return

            user_msg = extract_messages_content(messages[-1])
            ai_msg = extract_messages_content(ai_response[-1])

            if not user_msg or len(user_msg) < 3:
                return

            user_id = runtime.context.user_id
            session_id = runtime.context.conversation_id

            # ⚠️ entity 单独串行（避免并发覆盖）
            await asyncio.gather(
                self._update_user_profile(runtime, user_id, user_msg, ai_msg),
                self._update_session_context(runtime, user_id, session_id, user_msg),
            )

            await self._update_entities(runtime, user_id, session_id, user_msg, ai_msg)

        except Exception as e:
            logger.error(f"UnifiedContextUpdater failed: {e}")
            traceback.print_exc()

    # =========================
    # User Profile
    # =========================

    async def _update_user_profile(self, runtime, user_id, user_msg, ai_msg):
        import json
        import traceback

        try:
            backend = self._get_backend(runtime)
            file_path = "/profiles/profile.json"

            # ✅ 默认值（schema 保底）
            current_val = UserPersona().model_dump()

            # =========================
            # 1. 读取已有数据（安全版）
            # =========================
            if backend is not None:
                raw = await backend.aread(file_path)

                print(f"[READ RAW] {repr(raw)}")

                existing = self._load_json(raw, None)  # ⚠️ 不用 {}

                if isinstance(existing, dict):
                    current_val = existing
                else:
                    print("[WARN] invalid JSON, keep default")

            else:
                existing = await runtime.store.aget(user_id, USER_PROFILE_CONTEXT)
                if existing and isinstance(existing.value, dict):
                    current_val = existing.value

            # =========================
            # 2. 构造 prompt
            # =========================
            prompt = PROMPT_USER_PROFILE.format(
                current_val=json.dumps(current_val, ensure_ascii=False),
                user_msg=user_msg,
                ai_msg=ai_msg,
            )

            # =========================
            # 3. 调 LLM（加保护）
            # =========================
            patch = await self.user_extractor.ainvoke(prompt)

            if patch is None:
                print("[WARN] patch is None")
                return

            patch_dict = patch.model_dump(exclude_none=True)

            if not isinstance(patch_dict, dict):
                print("[WARN] patch not dict")
                return

            # =========================
            # 4. merge
            # =========================
            updated_val = self._smart_merge_user(current_val, patch_dict)

            # =========================
            # 5. 写入（安全 + 可读）
            # =========================
            if backend is not None:
                await backend.awrite(
                    file_path,
                    json.dumps(updated_val, ensure_ascii=False, indent=2)  # ✅
                )
            else:
                await runtime.store.aput(
                    user_id,
                    USER_PROFILE_CONTEXT,
                    updated_val
                )

            print(f"[WRITE OK] {file_path}")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"User Profile Update Error: {e}")

    # =========================
    # Session
    # =========================

    async def _update_session_context(self, runtime, user_id, session_id, user_msg):
        try:
            backend = self._get_backend(runtime)
            file_path = "/sessions/session.json"

            prompt = PROMPT_SESSION_STATE.format(user_msg=user_msg)
            update = await self.session_extractor.ainvoke(prompt)
            update_val = update.model_dump()

            if backend is not None:
                await backend.awrite(file_path, json.dumps(update_val))
            else:
                await runtime.store.aput(
                    user_id,
                    get_session_context_id(session_id),
                    update_val
                )

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Session Update Error: {e}")

    # =========================
    # Entities
    # =========================

    async def _update_entities(self, runtime, user_id, session_id, user_msg, ai_msg):
        try:
            backend = self._get_backend(runtime)
            file_path = "/entities/entities.json"

            prompt = PROMPT_ENTITY_EXTRACTION.format(
                user_msg=user_msg,
                ai_msg=ai_msg
            )

            new_entities = await self.entity_extractor.ainvoke(prompt)

            if not new_entities.entities:
                return

            # -------- read existing --------
            if backend is not None:
                raw = await backend.aread(file_path)
                existing_list = self._load_json(raw, [])
            else:
                existing = await runtime.store.aget(
                    user_id,
                    get_session_entity_id(session_id)
                )
                existing_list = existing.value if existing else []

            # -------- merge --------
            seen_names = {
                e.get("name")
                for e in existing_list
                if isinstance(e, dict) and "name" in e
            }

            for ent in new_entities.entities:
                if ent.name not in seen_names:
                    existing_list.append(ent.model_dump())

            # -------- write --------
            if backend is not None:
                await backend.awrite(file_path, json.dumps(existing_list))
            else:
                await runtime.store.aput(
                    user_id,
                    get_session_entity_id(session_id),
                    existing_list
                )

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Entity Update Error: {e}")

    # =========================
    # Merge Logic
    # =========================

    def _smart_merge_user(self, old: dict, new: dict) -> dict:
        for key in ["full_name", "job_role", "communication_style"]:
            if new.get(key):
                old[key] = new[key]

        for key in ["interests", "current_goals"]:
            if new.get(key):
                old[key] = list(set(old.get(key, []) + new[key]))

        if new.get("technical_level"):
            old.setdefault("technical_level", {}).update(new["technical_level"])

        return old