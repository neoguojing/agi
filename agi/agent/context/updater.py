import asyncio
from asyncio.log import logger
from typing import List, Dict, Optional,Any
from pydantic import BaseModel, Field
from .context import USER_PROFILE_CONTEXT,get_session_context_id,get_session_entity_id
from agi.utils.common import extract_messages_content
from deepagents.backends.protocol import BackendProtocol
import traceback
# --- Prompt Constants ---

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
- current_intent: The specific action the user wants to achieve NOW (e.g., 'debugging code', 'booking flight').
- user_emotion: Detect tone (e.g., 'frustrated', 'satisfied', 'neutral').
- topic_drift: Set to true ONLY if the user abruptly changes the subject from the previous turn.
"""

PROMPT_ENTITY_EXTRACTION = """
Task: Identify and extract named entities mentioned in the dialogue.
Context:
- User Message: {user_msg}
- AI Response: {ai_msg}

Instructions:
- Extract specific objects: Projects, Order IDs, Software Tools, Organizations, or People.
- Assign a 'type' to each entity.
- Capture key 'attributes' if mentioned (e.g., status='pending', version='2.0').
"""


class UserPersona(BaseModel):
    """
    Represents a persistent profile of the user to tailor AI interactions.
    This schema is used by the LLM to extract and update user-specific context.
    """
    
    # --- Static / Demographic Information ---
    full_name: Optional[str] = Field(
        None, 
        description="The user's legal or preferred full name."
    )
    job_role: Optional[str] = Field(
        None, 
        description="The user's current professional title or primary occupation (e.g., 'Senior DevOps Engineer')."
    )
    
    # --- Dynamic Preferences ---
    interests: List[str] = Field(
        default_factory=list, 
        description="A list of topics, industries, or technologies the user shows consistent interest in."
    )
    communication_style: str = Field(
        default="professional", 
        description="The user's preferred tone and language style (e.g., 'humorous', 'concise', 'academic', 'layman-friendly')."
    )
    
    # --- Competency & Depth Control ---
    technical_level: Dict[str, str] = Field(
        default_factory=dict, 
        description=(
            "A mapping of specific domains to the user's proficiency level. "
            "Example: {'Python': 'expert', 'Quantum Physics': 'beginner'}. "
            "Used to calibrate the technical depth of AI responses."
        )
    )

    # --- Temporal Context ---
    current_goals: List[str] = Field(
        default_factory=list, 
        description="Active projects, immediate objectives, or long-term milestones the user is currently working toward."
    )

class SessionUpdate(BaseModel):
    """
    Captures the transient state of the current conversation turn.
    Used to adjust the Agent's behavior and tone in real-time.
    """
    current_intent: Optional[str] = Field(
        None, 
        description="The primary goal or action the user wants to achieve in this specific turn (e.g., 'technical_debugging', 'product_inquiry')."
    )
    user_emotion: Optional[str] = Field(
        None, 
        description="The detected emotional state of the user (e.g., 'frustrated', 'satisfied', 'curious', 'neutral')."
    )
    topic_drift: bool = Field(
        False, 
        description="Set to true if the user has significantly shifted the subject away from the previous context."
    )

class Entity(BaseModel):
    """
    Represents a specific named object mentioned in the conversation.
    Helps maintain continuity for specific items across long sessions.
    """
    name: str = Field(
        ..., 
        description="The unique identifier or name of the entity (e.g., 'Project Phoenix', 'Order #12345')."
    )
    type: str = Field(
        ..., 
        description="The category of the entity (e.g., 'Project', 'Organization', 'Software_Tool', 'Ticket_ID')."
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Key-value pairs of metadata associated with the entity (e.g., {'status': 'active', 'priority': 'high'})."
    )

class EntityList(BaseModel):
    """A wrapper for structured entity extraction."""
    entities: List[Entity] = Field(default_factory=list)

class UnifiedContextUpdater:
    def __init__(self, model,backend:BackendProtocol=None):
        self.model = model
        self.backend = backend
        # 绑定结构化输出
        self.user_extractor = model.with_structured_output(UserPersona)
        self.session_extractor = model.with_structured_output(SessionUpdate)
        self.entity_extractor = model.with_structured_output(EntityList)

    async def update(self, runtime, messages, ai_response):
        """
        统一更新入口：并行处理用户画像、Session和实体
        """
        try:
            user_id = runtime.context.user_id
            session_id = runtime.context.conversation_id
            user_msg = extract_messages_content(messages[-1])
            ai_msg = extract_messages_content(ai_response[-1])
            # print(f"--- [Context Engine] 用户消息: {user_msg} | AI响应: {ai_msg} ---")
            # 过滤过短的噪音消息
            if len(user_msg) < 3: 
                print(f"--- [Context Engine] 用户消息过短，跳过更新 ---")
                return

            # 定义并行任务
            tasks = [
                self._update_user_profile(runtime, user_id, user_msg, ai_msg),
                self._update_session_context(runtime, user_id, session_id, user_msg),
                self._update_entities(runtime, user_id, session_id, user_msg, ai_msg)
            ]
            
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"UnifiedContextUpdater update failed: {e}")
            return # 如果连消息都提取不了，后续都没法做

    async def _update_user_profile(self, runtime, user_id, user_msg, ai_msg):
        try:
            current_val = UserPersona().model_dump()
            file_path = f"/profiles/{user_id}/profile.json"

            # 读取现有数据
            if self.backend:
                existing = await self.backend.aread(file_path)
                if existing:
                    current_val = existing
            else:
                existing = await runtime.store.aget(user_id, USER_PROFILE_CONTEXT)
                if existing:
                    current_val = existing.value

            # LLM 提取补丁
            prompt = PROMPT_USER_PROFILE.format(
                current_val=current_val, user_msg=user_msg, ai_msg=ai_msg
            )
            patch = await self.user_extractor.ainvoke(prompt)

            # 合并
            updated_val = self._smart_merge_user(current_val, patch.model_dump(exclude_none=True))

            # 写入
            if self.backend:
                await self.backend.awrite(file_path, updated_val)
            else:
                await runtime.store.aput(user_id, USER_PROFILE_CONTEXT, updated_val)

        except Exception as e:
            traceback.print_exc()
            print(f"User Profile Update Error: {e}")


    async def _update_session_context(self, runtime, user_id, session_id, user_msg):
        try:
            file_path = f"/sessions/{user_id}/{session_id}/session.json"
            prompt = PROMPT_SESSION_STATE.format(user_msg=user_msg)
            update = await self.session_extractor.ainvoke(prompt)
            update_val = update.model_dump()

            if self.backend:
                await self.backend.awrite(file_path, update_val)
            else:
                await runtime.store.aput(user_id, get_session_context_id(session_id), update_val)

        except Exception as e:
            traceback.print_exc()
            print(f"Session Update Error: {e}")


    async def _update_entities(self, runtime, user_id, session_id, user_msg, ai_msg):
        try:
            file_path = f"/entities/{user_id}/{session_id}/entities.json"
            prompt = PROMPT_ENTITY_EXTRACTION.format(user_msg=user_msg, ai_msg=ai_msg)
            new_entities = await self.entity_extractor.ainvoke(prompt)

            if not new_entities.entities:
                return

            # 读取旧实体
            if self.backend:
                existing_list = await self.backend.aread(file_path) or []
            else:
                existing = await runtime.store.aget(user_id, get_session_entity_id(session_id))
                existing_list = existing.value if existing else []

            # 合并去重
            seen_names = {e['name'] for e in existing_list}
            for ent in new_entities.entities:
                if ent.name not in seen_names:
                    existing_list.append(ent.model_dump())

            # 写入
            if self.backend:
                await self.backend.awrite(file_path, existing_list)
            else:
                await runtime.store.aput(user_id, get_session_entity_id(session_id), existing_list)

        except Exception as e:
            traceback.print_exc()
            print(f"Entity Update Error: {e}")

    def _smart_merge_user(self, old: dict, new: dict) -> dict:
        """保持 UserPersona 逻辑一致"""
        for key in ["full_name", "job_role", "communication_style"]:
            if new.get(key): old[key] = new[key]
        for key in ["interests", "current_goals"]:
            if new.get(key):
                old[key] = list(set(old.get(key, []) + new[key]))
        if new.get("technical_level"):
            old.setdefault("technical_level", {}).update(new["technical_level"])
        return old