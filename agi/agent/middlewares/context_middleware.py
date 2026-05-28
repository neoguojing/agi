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

ORGANIZE_MEMORY_TOOL_DESCRIPTION = """Use this tool to explicitly persist key information from the current conversation into your long-term memory.

## When to Use This Tool
1. **Stable Preferences**: When the user shares a persistent preference (e.g., 'I always use VS Code for Python'). Use target `profile`.
2. **Key Milestones**: When a significant decision or event occurs (e.g., 'The project architecture was finalized'). Use target `episodic`.
3. **Factual Knowledge**: When you discover a stable fact about the project or user (e.g., 'The production server is located in us-east-1'). Use target `semantic`.
4. **Explicit Requests**: When the user says 'Remember this'.

## How to Use This Tool
- **Choose Target**: Select `profile`, `episodic`, or `semantic` based on the nature of the information.
- **Provide Records**: Provide a list of records matching the target schema:
    - `profile`: list of {key, value, confidence}
    - `episodic`: list of {summary, event_time, participants, confidence}
    - `semantic`: list of {subject, predicate, object, confidence}
- **Provide Reason**: Explain why this is being saved.

## Guarantees
Information provided through this tool is immediately processed, deduplicated, and saved. It will be available in your system prompt in the next turn.
"""

ORGANIZE_MEMORY_SYSTEM_PROMPT = """## `organize_memory` tool
You can proactively manage your long-term memory using the `organize_memory` tool.
Treat this tool as a 'Save' button. If you encounter a 'golden' piece of information, use this tool immediately to ensure it is not lost.
"""

# --- Tool Implementation ---
@tool(description=ORGANIZE_MEMORY_TOOL_DESCRIPTION)
def organize_memory(
    records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]],
    target: MemoryTarget,
    reason: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    try:
        """Trigger memory organization by updating the state."""

        return Command(
            update={
                "pending_target": target,
                "profile_records": records if target == "profile" else [],
                "episodic_records": records if target == "episodic" else [],
                "semantic_records": records if target == "semantic" else [],
                "organization_reason": reason,
                "messages": [ToolMessage(content=f"Memory records for {target} received. Reason: {reason}", tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(f"Memory records for {target} failed. Reason: {e}", tool_call_id=tool_call_id,statu="error")
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
        records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]],
        target: MemoryTarget,
        reason: str
    ) -> Command[Any]:
        try:
            """Persist key information from the current conversation into long-term memory."""
            extraction_result = MemoryExtractionResult(
                profile_memories=records if target == "profile" else [],
                episodic_memories=records if target == "episodic" else [],
                semantic_memories=records if target == "semantic" else [],
            )
            patches = extraction_result.to_patches(reason=reason)
            applied_count = 0
            for patch in patches:
                self.memory_manager.store.apply_patch(patch)
                applied_count += 1

            return Command(
                update={
                    "pending_target": target,
                    "profile_records": records if target == "profile" else [],
                    "episodic_records": records if target == "episodic" else [],
                    "semantic_records": records if target == "semantic" else [],
                    "organization_reason": reason,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully persisted {applied_count} memory patches to {target} store. Reason: {reason}",
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                }
            )
        except Exception as e:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Failed to persist memories to {target}: {e}",
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                }
            )

    async def _aorganize_memory(
        self,
        runtime: ToolRuntime[ContextT, MemoryState[ResponseT]],
        records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]],
        target: MemoryTarget,
        reason: str
    ) -> Command[Any]:
        """Persist key information from the current conversation into long-term memory."""
        return self._organize_memory(runtime, records, target, reason)

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
            await self.memory_manager.start()

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

    @override
    async def aafter_model(
        self, state: MemoryState[ResponseT], runtime: Any
    ) -> dict[str, Any] | None:
        """Handle the persistence of memory records provided by the model."""
        # Persistence is now handled directly by the organize_memory tool.
        return None

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")
