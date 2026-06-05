import asyncio
import json
import platform
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Awaitable, Any, Annotated, Union, Optional
from venv import logger
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, InjectedToolCallId
from langgraph.types import Command
from langchain.tools import ToolRuntime,tool
from langgraph.channels import LastValue


from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)

from deepagents.backends.protocol import BackendProtocol
from agi.agent.prompt import get_middleware_prompt
from agi.utils.common import append_to_system_message
from agi.agent.context.memory import (
    MemoryMaintenanceManager,
    MessageProvider
)
from agi.agent.context.memory_models import (
    MemoryTarget,
    MemoryExtractionResult,
    ProfileMemoryRecord,
    EpisodicMemoryRecord,
    SemanticMemoryRecord
)

from agi.agent.context.memory_store import (
    EPISODIC_MAX_RECORDS,
    EPISODIC_RETENTION_DAYS,
    SEMANTIC_MAX_RECORDS,
    SEMANTIC_RETENTION_DAYS,
    record_dedup_key,
)

# --- State and Input Definitions ---



def profile_memory_delta_reducer(
    state: Optional[list[ProfileMemoryRecord]],
    writes: list[ProfileMemoryRecord],
) -> list[ProfileMemoryRecord]:

    merged = {
        record_dedup_key("profile", r.model_dump()): r
        for r in (state or [])
        if record_dedup_key("profile", r.model_dump()) is not None
    }

    order = list(merged.keys())

    for r in writes:
        k = record_dedup_key("profile", r.model_dump())
        if k and k not in merged:
            order.append(k)
        if k:
            merged[k] = r

    return [merged[k] for k in order if k in merged]

def episodic_memory_delta_reducer(
    state: Optional[list[EpisodicMemoryRecord]],
    writes: list[EpisodicMemoryRecord],
) -> list[EpisodicMemoryRecord]:

    expire_before = datetime.now(timezone.utc) - timedelta(days=EPISODIC_RETENTION_DAYS)

    def ts(v: str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    merged = {
        record_dedup_key("episodic", r.model_dump()): r
        for r in (state or [])
        if (not r.event_time or ts(r.event_time) >= expire_before)
        and record_dedup_key("episodic", r.model_dump()) is not None
    }

    merged.update({
        record_dedup_key("episodic", r.model_dump()): r
        for r in writes
        if record_dedup_key("episodic", r.model_dump()) is not None
    })

    return sorted(
        merged.values(),
        key=lambda r: ts(r.event_time),
        reverse=True,
    )[:EPISODIC_MAX_RECORDS]

def semantic_memory_delta_reducer(
    state: Optional[list[SemanticMemoryRecord]],
    writes: list[SemanticMemoryRecord],
) -> list[SemanticMemoryRecord]:

    expire_before = datetime.now(timezone.utc) - timedelta(days=SEMANTIC_RETENTION_DAYS)

    def ts(r: SemanticMemoryRecord):
        try:
            return datetime.fromisoformat(
                r.updated_at.replace("Z", "+00:00")
            )
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    merged = {
        record_dedup_key("semantic", r.model_dump()): r
        for r in (state or [])
        if (not getattr(r, "updated_at", None)
        or ts(r) >= expire_before)
        and record_dedup_key("semantic", r.model_dump()) is not None
    }

    merged.update({
        record_dedup_key("semantic", r.model_dump()): r
        for r in writes
        if record_dedup_key("semantic", r.model_dump()) is not None
    })

    return sorted(
        merged.values(),
        key=ts,
        reverse=True,
    )[:SEMANTIC_MAX_RECORDS]

class MemoryState(AgentState[ResponseT]):
    """State schema for the memory organization middleware."""
    # The memory records the model wants to persist
    profile_records: Annotated[NotRequired[list[ProfileMemoryRecord]], profile_memory_delta_reducer]
    episodic_records: Annotated[NotRequired[list[EpisodicMemoryRecord]], episodic_memory_delta_reducer]  
    semantic_records: Annotated[NotRequired[list[SemanticMemoryRecord]],semantic_memory_delta_reducer] 
    # The type of memory target chosen by the model
    pending_target: Annotated[NotRequired[MemoryTarget], LastValue]
    # Reason for the current organization request
    organization_reason: Annotated[NotRequired[str], LastValue]

class OrganizeMemoryInput(BaseModel):
    """Input schema for the `organize_memory` tool.

    This tool acts as a protocol for the model to explicitly define what
    information should be transitioned from short-term context to long-term memory.
    """
    target: MemoryTarget = Field(
        description="The type of memory to update. Valid values: 'profile' (user traits), 'episodic' (events), 'semantic' (facts)."
    )
    records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]] = Field(
        description="The memory records to persist. Must match the target type: ProfileMemoryRecord for 'profile', EpisodicMemoryRecord for 'episodic', or SemanticMemoryRecord for 'semantic'."
    )
    reason: str = Field(
        default="Explicit request to organize memory",
        description="The reason for organizing memory, helps in auditing why this information was saved."
    )

# --- Prompts ---

ORGANIZE_MEMORY_SYSTEM_PROMPT = """## Long-Term Memory Manager
You can proactively manage your long-term memory using the `organize_memory` tool. 
Treat this tool as your "Memory Maintenance" interface. It supports simultaneous adding/updating (Upsert) and removing (Deletion) of memories.

Whenever you encounter new 'golden' information, notice a shift in user preferences, or need to consolidate conflicting facts, use this tool immediately to keep your memory state accurate, compact, and deduplicated.
"""

# =====================================================================
# 2. TOOL DESCRIPTION VARIABLE
# =====================================================================
ORGANIZE_MEMORY_TOOL_DESCRIPTION = """Use this tool to explicitly upsert (add/update) and delete information in your long-term memory.

## 1. Choose Target
- `profile`: For persistent preferences, habits, or user identity (e.g., 'I always use VS Code').
- `episodic`: For significant events, decisions, or milestones (e.g., 'Project architecture finalized').
- `semantic`: For stable factual knowledge triples (e.g., 'database' -> 'uses' -> 'PostgreSQL').

## 2. How to Apply Delta Updates
- **upserts**: List of full records to ADD or UPDATE. If the key already exists, it will be safely overwritten.
- **deletions**: List of STRING keys to REMOVE. Use this to delete obsolete, conflicting, or redundant memories.
  - For `profile`: Use the exact key (e.g., ["favorite_ide"]).
  - For `episodic`: Use "summary_date" format (e.g., ["beta tested_2026-06-05"]).
  - For `semantic`: Use "subject:predicate:object" format (e.g., ["db:uses:mysql"]).

## Guarantees
Information is immediately processed via dictionary reducers. 'deletions' are explicitly popped from the state, and 'upserts' are merged.
"""
    
@tool(description=ORGANIZE_MEMORY_TOOL_DESCRIPTION)
def organize_memory(
    target: MemoryTarget, # Literal["profile", "episodic", "semantic"]
    reason: str, 
    tool_call_id: Annotated[str, InjectedToolCallId],
    upserts: Optional[List[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]]] = Field(default=[], description="Records to add or update."),
    deletions: Optional[List[str]] = Field(default=[], description="String keys of memories to completely remove.")
) -> Command[Any]:
    try:
        # 初始化用于更新的字典
        target_dict = {}

        # ==========================================
        # 1. 统一处理 Upserts (提取唯一 Key 归一化)
        # ==========================================
        if upserts:
            for record in upserts:
                key = None
                
                # Profile 模式解析
                if target == "profile" and hasattr(record, "key") and record.key:
                    key = record.key.strip().lower()
                
                # Episodic 模式解析
                elif target == "episodic" and hasattr(record, "summary") and record.summary:
                    date_str = record.event_time[:10] if getattr(record, "event_time", None) else "anytime"
                    key = f"{record.summary.strip().lower()}_{date_str}"
                
                # Semantic 模式解析
                elif target == "semantic" and hasattr(record, "subject") and record.subject:
                    key = f"{record.subject.strip().lower()}:{record.predicate}:{record.object.strip().lower()}"

                # 存入字典
                if key:
                    target_dict[key] = record

        # ==========================================
        # 2. 统一处理 Deletions (显式赋值为 None 触发删除)
        # ==========================================
        if deletions:
            for delete_key in deletions:
                if delete_key:
                    target_dict[delete_key.strip().lower()] = None

        # ==========================================
        # 3. 构造精准的 Command Update 载荷
        # ==========================================
        # 基础更新载荷
        update_payload = {
            "pending_target": target,
            "organization_reason": reason,
            "messages": [
                ToolMessage(
                    content=f"Memory reorganized for target '{target}'. Upserted {len(upserts or [])} items, Deleted {len(deletions or [])} items. Reason: {reason}", 
                    tool_call_id=tool_call_id
                )
            ]
        }

        # 仅将修改后的字典打入对应的 State 字段
        # （不修改的字段完全不写，依靠 LangGraph 原生浅合并保留旧数据）
        if target == "profile":
            update_payload["profile_records"] = target_dict
        elif target == "episodic":
            update_payload["episodic_records"] = target_dict
        elif target == "semantic":
            update_payload["semantic_records"] = target_dict

        return Command(update=update_payload)

    except Exception as e:
        logger.exception(f"Failed to organize memory for target {target}. Error: {e}")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to organize memory for {target}. Error: {e}", 
                        tool_call_id=tool_call_id, 
                        status="error" # 注意：langchain较新版本才支持status参数，若报错可移除
                    )
                ],
            }
        )

# Dynamically create the organize_memory tool with the custom description


# --- Middleware ---

class MiddlewareMessageProvider(MessageProvider):
    """Dynamic message provider that can be updated by the middleware."""
    def __init__(self):
        self._messages = []

    def update_messages(self, messages: List[BaseMessage]):
        self._messages = messages

    def get_messages(self) -> List[BaseMessage]:
        return self._messages

class ContextEngineeringMiddleware(AgentMiddleware[MemoryState[ResponseT],ContextT, ResponseT]):
    """
    上下文工程中间件：
    1. 动态注入模型 Prompt
    2. 后台异步执行记忆提取 (MemoryMaintenanceManager)
    3. 动态注入最新记忆 (format_memory_for_llm)
    4. 拦截 `organize_memory` 工具调用，将模型直接提供的记录同步刷入记忆存储
    """
    state_schema = MemoryState

    def __init__(
        self,
        backend = None,
        llm = None
    ):
        self.backend = backend
        self.llm = llm
        self.memory_manager = None
        self.message_provider = MiddlewareMessageProvider()
        self.memory_cache = None
        self.tools = [
            StructuredTool.from_function(
                name="organize_memory",
                description=ORGANIZE_MEMORY_TOOL_DESCRIPTION,
                func=self._organize_memory,
                coroutine=self._aorganize_memory,
                args_schema=OrganizeMemoryInput,
                infer_schema=False,
            )
        ]

    def _organize_memory(
        self,
        runtime: ToolRuntime[ContextT, MemoryState[ResponseT]],
        target: MemoryTarget,
        reason: str,
        upserts: Optional[List[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]]] = Field(default=[], description="Records to add or update."),
        deletions: Optional[List[str]] = Field(default=[], description="String keys of memories to completely remove.")
    ) -> Command[Any]:
        """Synchronously persist key information into long-term memory."""
        try:
            # 初始化用于更新的字典
            target_dict = {}

            # ==========================================
            # 1. 统一处理 Upserts (提取唯一 Key 归一化)
            # ==========================================
            if upserts:
                for record in upserts:
                    key = None
                    
                    # Profile 模式解析
                    if target == "profile" and hasattr(record, "key") and record.key:
                        key = record.key.strip().lower()
                    
                    # Episodic 模式解析
                    elif target == "episodic" and hasattr(record, "summary") and record.summary:
                        date_str = record.event_time[:10] if getattr(record, "event_time", None) else "anytime"
                        key = f"{record.summary.strip().lower()}_{date_str}"
                    
                    # Semantic 模式解析
                    elif target == "semantic" and hasattr(record, "subject") and record.subject:
                        key = f"{record.subject.strip().lower()}:{record.predicate}:{record.object.strip().lower()}"

                    # 存入字典
                    if key:
                        target_dict[key] = record

            # ==========================================
            # 2. 统一处理 Deletions (显式赋值为 None 触发删除)
            # ==========================================
            if deletions:
                for delete_key in deletions:
                    if delete_key:
                        target_dict[delete_key.strip().lower()] = None

            # ==========================================
            # 3. 构造精准的 Command Update 载荷
            # ==========================================
            # 基础更新载荷
            update_payload = {
                "pending_target": target,
                "organization_reason": reason,
                "messages": [
                    ToolMessage(
                        content=f"Memory reorganized for target '{target}'. Upserted {len(upserts or [])} items, Deleted {len(deletions or [])} items. Reason: {reason}", 
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }

            # 仅将修改后的字典打入对应的 State 字段
            # （不修改的字段完全不写，依靠 LangGraph 原生浅合并保留旧数据）
            if target == "profile":
                update_payload["profile_records"] = target_dict
            elif target == "episodic":
                update_payload["episodic_records"] = target_dict
            elif target == "semantic":
                update_payload["semantic_records"] = target_dict

            return Command(update=update_payload)

        except Exception as e:
            logger.exception(f"Failed to organize memory for target {target}. Error: {e}")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Failed to organize memory for {target}. Error: {e}", 
                            tool_call_id=runtime.tool_call_id,
                            status="error" # 注意：langchain较新版本才支持status参数，若报错可移除
                        )
                    ],
                }
            )

    async def _aorganize_memory(
        self,
        runtime: ToolRuntime[ContextT, MemoryState[ResponseT]],
        target: MemoryTarget,
        reason: Optional[str],
        upserts: Optional[List[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]]] = Field(default=[], description="Records to add or update."),
        deletions: Optional[List[str]] = Field(default=[], description="String keys of memories to completely remove.")
    ) -> Command[Any]:
        """Persist key information from the current conversation into long-term memory."""
        return self._organize_memory(runtime, target=target,reason=reason,upserts=upserts,deletions=deletions)

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _format_environment_context(self, runtime) -> str:
        try:
            now = datetime.now(timezone.utc).isoformat()

            env_info = {
                "current_time_utc": now,
                "timezone": "UTC",
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version()
            }

            if runtime:
                env_info.update({
                    "user_id": getattr(runtime.context, "user_id", None),
                    "conversation_id": getattr(runtime.context, "conversation_id", None),
                })

            env_str = json.dumps(env_info, indent=2, ensure_ascii=False)

            return f"""
    <environment>
    {env_str}
    </environment>
    """.strip()

        except Exception as e:
            logger.error(f"Failed to build environment context: {e}")
            return "<environment>(failed to load)</environment>"

    def format_memory_for_llm(
        self,
        state: AgentState
    ) -> str:
        """
        Formats structured memory into a human-readable string
        suitable for LLM context injection.
        """

        sections: list[str] = []

        # =====================================================
        # Profile Memory
        # =====================================================
        if state.get('profile_records'):

            lines = [
                "--- PROFILE MEMORY ---"
            ]

            for rec in state.get('profile_records'):

                lines.append(
                    f"- {rec.key}: {rec.value}"
                )

            sections.append(
                "\n".join(lines)
            )

        # =====================================================
        # Episodic Memory
        # =====================================================
        if state.get('episodic_records'):

            lines = [
                "--- EPISODIC MEMORY ---"
            ]

            for rec in state.get('episodic_records'):

                event_time = (
                    rec.event_time
                    or "Unknown time"
                )

                lines.append(
                    f"- [{event_time}] {rec.summary}"
                )

            sections.append(
                "\n".join(lines)
            )

        # =====================================================
        # Semantic Memory
        # =====================================================
        if state.get('semantic_records'):

            lines = [
                "--- SEMANTIC MEMORY ---"
            ]

            for rec in state.get('semantic_records'):

                lines.append(
                    f"- {rec.subject} "
                    f"{rec.predicate} "
                    f"{rec.object}"
                )

            sections.append(
                "\n".join(lines)
            )

        if not sections:
            return "No memory available."

        return "\n\n".join(sections)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:

        runtime = request.runtime
        backend = self._get_backend(runtime)

        self.message_provider.update_messages(request.messages)

        if self.memory_manager is None and self.llm is not None:
            self.memory_manager = MemoryMaintenanceManager(
                llm=self.llm,
                backend=backend,
                messages=self.message_provider
            )
            # await self.memory_manager.start()

            self.memory_cache = await self.memory_manager.load_memories()

        # Load memories from files

        request.state["profile_records"] =  profile_memory_delta_reducer(request.state["profile_records"],self.memory_cache.profile_memories)
        request.state["episodic_records"] = episodic_memory_delta_reducer(request.state["episodic_records"],self.memory_cache.episodic_memories)
        request.state["semantic_records"] = semantic_memory_delta_reducer(request.state["semantic_records"],self.memory_cache.semantic_memories)

        request = request.override(state=request.state)
            
        memory_body = self.format_memory_for_llm(request.state)

        memory_context_str = get_middleware_prompt("context").format(agent_memory=memory_body)
        env_context_str = self._format_environment_context(runtime)

        injected_context_str = f"""
        {ORGANIZE_MEMORY_SYSTEM_PROMPT}

        {env_context_str}

        {memory_context_str}
        """.strip()

        request = request.override(
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        try:
            response = await handler(request)
            return response
        except Exception as e:
            logger.exception("ContextEngineeringMiddleware model call failed: %s", e)
            return ModelResponse(result=[AIMessage(content=f"Model call failed: {type(e).__name__}: {e}")])

    async def stop(self):
        """Clean up memory manager and flush final state."""
        if self.memory_manager:
            await self.memory_manager.stop()

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")
