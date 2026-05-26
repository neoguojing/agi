import json
import platform
import datetime
import os
import asyncio
from typing import Callable, List, Awaitable, Any, Annotated, Literal, cast, Union
from venv import logger
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, InjectedToolCallId,tool
from langgraph.types import Command
from langchain.tools import ToolRuntime

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
    format_memory_for_llm,
    MessageProvider
)
from agi.agent.context.memory_models import (
    MemoryTarget,
    MemoryExtractionResult,
    ProfileMemoryRecord,
    EpisodicMemoryRecord,
    SemanticMemoryRecord
)

# --- State and Input Definitions ---

class MemoryState(AgentState[ResponseT]):
    """State schema for the memory organization middleware."""
    # The memory records the model wants to persist
    pending_records: Annotated[NotRequired[MemoryExtractionResult], OmitFromInput]
    # The type of memory target chosen by the model
    pending_target: NotRequired[MemoryTarget]
    # Reason for the current organization request
    organization_reason: NotRequired[str]

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
    """Trigger memory organization by updating the state."""
    import pdb;pdb.set_trace()
    extraction_result = MemoryExtractionResult(
        profile_memories=records if target == "profile" else [],
        episodic_memories=records if target == "episodic" else [],
        semantic_memories=records if target == "semantic" else []
    )
    return Command(
        update={
            "pending_target": target,
            "pending_records": extraction_result,
            "organization_reason": reason,
            "messages": [ToolMessage(content=f"Memory records for {target} received. Reason: {reason}", tool_call_id=tool_call_id)],
        }
    )

# Dynamically create the organize_memory tool with the custom description
def _organize_memory(
    runtime: ToolRuntime[ContextT, MemoryState[ResponseT]], 
    records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]],
    target: MemoryTarget,
    reason: str
) -> Command[Any]:
    import pdb;pdb.set_trace()

    """Create and manage a structured task list for your current work session."""
    extraction_result = MemoryExtractionResult(
        profile_memories=records if target == "profile" else [],
        episodic_memories=records if target == "episodic" else [],
        semantic_memories=records if target == "semantic" else []
    )
    return Command(
        update={
            "pending_target": target,
            "pending_records": extraction_result,
            "organization_reason": reason,
            "messages": [
                ToolMessage(f"Memory records for {target} received. Reason: {reason}", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


async def _aorganize_memory(
    runtime: ToolRuntime[ContextT, MemoryState[ResponseT]], 
    records: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]],
    target: MemoryTarget,
    reason: str
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return _organize_memory(runtime, records,target,reason)


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
                func=_organize_memory,
                coroutine=_aorganize_memory,
                args_schema=OrganizeMemoryInput,
                infer_schema=False,
            )
        ]

    def _get_backend(self, runtime) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _format_environment_context(self, runtime) -> str:
        try:
            now = datetime.datetime.utcnow().isoformat()

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
            # import pdb;pdb.set_trace()

            self.memory_cache = await self.memory_manager.load_memories()

        # Load memories from files
        if request.state.get('pending_target') is None:
            request.state['pending_target'] = self.memory_cache
            request = request.override(state=request.state)
            
        memory_body = format_memory_for_llm(request.state['pending_target'])

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
    def after_model(
        self, state: MemoryState, runtime: Any
    ) -> dict[str, Any] | None:
        """Check for `organize_memory` tool calls and handle them synchronously."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        org_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "organize_memory"]

        if not org_calls:
            return None

        return None

    @override
    async def aafter_model(
        self, state: MemoryState[ResponseT], runtime: Any
    ) -> dict[str, Any] | None:
        """Handle the persistence of memory records provided by the model."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        org_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "organize_memory"]

        if not org_calls:
            return None

        call = org_calls[0]

        if self.memory_manager is None:
            return {
                "messages": [
                    ToolMessage(
                        content="Error: Memory manager not initialized. Could not save memories.",
                        tool_call_id=call["id"],
                        status="error"
                    )
                ]
            }

        try:
            # Extract requested records and target from state
            target = state.get("pending_target")
            records = state.get("pending_records")
            reason = state.get("organization_reason", "Manual organization")

            if target and records:
                patches = records.to_patches(reason=reason)
                applied_count = 0
                for patch in patches:
                    self.memory_manager.store.apply_patch(patch)
                    applied_count += 1

                result_text = f"Successfully persisted {applied_count} memory patches to {target} store."
            else:
                result_text = "No memory records or target provided to organize."

            return {
                "messages": [
                    ToolMessage(
                        content=result_text,
                        tool_call_id=call["id"]
                    )
                ],
                "pending_records": None,
                "pending_target": None,
                "organization_reason": None,
            }
        except Exception as e:
            logger.exception("Failed to persist memories in aafter_model")
            return {
                "messages": [
                    ToolMessage(
                        content=f"Error persisting memories: {str(e)}",
                        tool_call_id=call["id"],
                        status="error"
                    )
                ]
            }

    def _log_debug_info(self, ctx_data: str, total_count: int):
        print(f"--- [Context Engine] 注入数据: {ctx_data} | 消息流长度: {total_count} ---")
