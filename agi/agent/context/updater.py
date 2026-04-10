import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage
from .context import UnifiedUpdateResult, merge_dict
from agi.utils.common import extract_messages_content

logger = logging.getLogger(__name__)


class UnifiedContextManager:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.extractor = llm_model.with_structured_output(UnifiedUpdateResult)

        # ✅ cache 改为存 JSON string（最小改动）
        self._cache: Dict[str, str] = {}

    # =========================
    # 获取上下文（JSON）
    # =========================
    async def get_context(self, runtime) -> str:
        """
        返回 JSON 字符串（不再是 Markdown）
        """
        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        cache_key = f"{user_id}:{session_id}"

        # 1. cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 2. 并发读取
        tasks = [
            runtime.store.aget(user_id, "user_profile"),
            runtime.store.aget(user_id, f"{session_id}:session"),
            runtime.store.aget(user_id, f"{session_id}:entities"),
        ]

        try:
            profile_data, session_data, entities_data = await asyncio.gather(*tasks)

            profile_dict = (profile_data.value or {}) if profile_data else {}
            session_dict = (session_data.value or {}) if session_data else {}
            entities_list = (entities_data.value or []) if entities_data else []

        except Exception as e:
            logger.error(f"Store read error: {e}")
            profile_dict, session_dict, entities_list = {}, {}, []

        # 3. 构建 JSON context
        context = {
            "environment": self._build_environment_context(),
            "user_profile": profile_dict,
            "session_state": session_dict,
            "entities": entities_list,
        }

        # 4. 转 JSON string
        final_json = json.dumps(context, ensure_ascii=False)

        # 5. cache
        self._cache[cache_key] = final_json

        return final_json

    # =========================
    # 更新上下文
    # =========================
    async def update(self, runtime, messages: List[BaseMessage]):
        if not messages:
            return

        user_id = runtime.context.user_id
        session_id = runtime.context.conversation_id
        cache_key = f"{user_id}:{session_id}"

        # 1. 读取当前数据
        tasks = [
            runtime.store.aget(user_id, "user_profile"),
            runtime.store.aget(user_id, f"{session_id}:session"),
            runtime.store.aget(user_id, f"{session_id}:entities"),
        ]

        try:
            results = await asyncio.gather(*tasks)

            current_profile = (results[0].value or {}) if results[0] else {}
            current_session = (results[1].value or {}) if results[1] else {}
            current_entities = (results[2].value or []) if results[2] else []

        except Exception as e:
            logger.error(f"Read for update failed: {e}")
            current_profile, current_session, current_entities = {}, {}, []

        try:
            # 2. 构建 prompt
            history_text = self._format_messages(messages)

            prompt = self._build_unified_prompt(
                history=history_text,
                curr_profile=json.dumps(current_profile, ensure_ascii=False),
                curr_session=json.dumps(current_session, ensure_ascii=False),
                curr_entities=json.dumps(current_entities, ensure_ascii=False),
            )

            # 3. LLM 调用
            try:
                result = await self.extractor.ainvoke(prompt)

                if (
                    not result.user_profile
                    and not result.session_state
                    and not result.new_entities
                ):
                    logger.info("LLM returned empty update, skipping store write.")
                    return
                print(f"----------------{result}")
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                return

            # 4. merge
            merged_profile = merge_dict(
                current_profile, result.user_profile.model_dump()
            )
            merged_session = merge_dict(
                current_session, result.session_state.model_dump()
            )
            merged_entities = await self._merge_entities_list(
                current_entities, result.new_entities
            )

            # 5. 写 store
            store_tasks = [
                runtime.store.aput(user_id, "user_profile", merged_profile),
                runtime.store.aput(user_id, f"{session_id}:session", merged_session),
                runtime.store.aput(user_id, f"{session_id}:entities", merged_entities),
            ]

            await asyncio.gather(*store_tasks, return_exceptions=True)

            # 6. 刷新 cache
            context = {
                "environment": self._build_environment_context(),
                "user_profile": merged_profile,
                "session_state": merged_session,
                "entities": merged_entities,
            }
            final_json = json.dumps(context, ensure_ascii=False)
            self._cache[cache_key] = final_json

            logger.info(f"Context updated successfully for session {session_id}:{self._cache[cache_key]}")

        except Exception as e:
            logger.error(f"Context update failed: {e}", exc_info=True)

    # =========================
    # 工具方法
    # =========================

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        lines = []
        for msg in messages:
            role = msg.type
            content = extract_messages_content(msg)
            if role != "system" and content:
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _build_unified_prompt(
        self, history, curr_profile, curr_session, curr_entities
    ):
        schema_json = json.dumps(
            UnifiedUpdateResult.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are an expert context manager.

### Conversation History
{history}

### Current State
Profile: {curr_profile}
Session: {curr_session}
Entities: {curr_entities}

### Task
Update:
- user_profile
- session_state
- new_entities

Rules:
- Only include NEW or CHANGED data
- Do NOT repeat unchanged entities

### Output Schema
{schema_json}

Return ONLY JSON.
"""

    def _build_environment_context(self) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        return {
            "current_time_utc": now_utc.isoformat(),
            "weekday_utc": now_utc.strftime("%A"),
            "timestamp": int(now_utc.timestamp()),
            "environment": {
                "python_version": sys.version.split()[0],
            },
        }

    async def _merge_entities_list(self, existing_list, new_list):
        entity_map = {}

        for e in existing_list:
            if isinstance(e, dict) and "type" in e and "name" in e:
                key = f"{e['type']}:{e['name']}"
                entity_map[key] = e

        for ent in (new_list or []):
            key = f"{ent.type}:{ent.name}"

            if key in entity_map:
                entity_map[key].update(ent.model_dump())
            else:
                entity_map[key] = ent.model_dump()

        return list(entity_map.values())