import asyncio
import json
import platform
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Awaitable, Any, Annotated, Union, Optional
from venv import logger
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import StructuredTool, InjectedToolCallId
from langgraph.types import Command
from langchain.tools import ToolRuntime,tool
from langgraph.channels import LastValue
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import get_buffer_string
from langchain_core.messages import AnyMessage

from pydantic import BaseModel, Field

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

from agi.scheduler.memory_task.memory_models import (
    MemoryTarget,
    ProfileMemoryRecord,
    EpisodicMemoryRecord,
    SemanticMemoryRecord
)
from agi.scheduler.memory_task.memory_state import MemoryState
from agi.scheduler import runtime_state_bridge,hub,memory_manager


class OrganizeMemoryInput(BaseModel):
    """Input schema for the `organize_memory` tool."""
    target: MemoryTarget = Field(
        description="The type of memory to update. Valid values: 'profile', 'episodic', or 'semantic'."
    )
    upserts: list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]] = Field(
        default_factory=list,
        description="Records to add or update. Ensure they match the 'target' type."
    )
    deletions: list[str] = Field(
        default_factory=list,
        description="String keys of memories to completely remove."
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
Constraint: Must be tied to explicit user intent or project outcomes; strictly exclude trivial chitchat and intermediate debugging steps.
- `semantic`: For stable factual knowledge triples (e.g., 'user' -> 'use' -> 'PostgreSQL').
Constraint: Only extract user-provided facts or long-term preferences; strictly ignore AI-generated explanations, generic definitions, and ephemeral context.

## 2. How to Apply Delta Updates
- **upserts**: List of full records to ADD or UPDATE. If the key already exists, it will be safely overwritten.
- **deletions**: List of STRING keys to REMOVE. Use this to delete obsolete, conflicting, or redundant memories.
  - For `profile`: Use the exact key (e.g., ["favorite_ide"]).
  - For `episodic`: Use "id" format (e.g., ["ep_7c8d2e1a"]).
  - For `semantic`: Use "subject:predicate:object" format (e.g., ["db:uses:mysql"]).

## Guarantees
Information is immediately processed via dictionary reducers. 'deletions' are explicitly popped from the state, and 'upserts' are merged.
"""

class ContextEngineeringMiddleware(AgentMiddleware[MemoryState[ResponseT],ContextT, ResponseT]):
    """
    上下文工程中间件：
    1. 动态注入模型 Prompt
    2. 后台异步执行记忆提取 (MemoryMaintenanceManager)
    3. 动态注入最新记忆 (format_memory_for_llm)
    4. 拦截 `organize_memory` 工具调用，将模型直接提供的记录同步刷入记忆存储
    5. 实现压缩会话覆盖：将全量消息流替换为 [摘要消息 + 增量消息]
    """
    state_schema = MemoryState

    def __init__(
        self,
        backend = None,
        llm = None
    ):
        self.backend = backend
        self.llm = llm
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

        self.scheduler = None


    def _organize_memory(
        self,
        runtime: ToolRuntime[ContextT, MemoryState[ResponseT]],
        target: MemoryTarget,
        reason: str,
        upserts: Optional[list[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]]] = None,
        deletions: Optional[list[str]] = None
    ) -> Command[MemoryState]:
        """Synchronously persist key information into long-term memory."""
        pass

    async def _aorganize_memory(
        self,
        runtime: ToolRuntime[ContextT, MemoryState[ResponseT]],
        target: MemoryTarget,
        reason: Optional[str],
        upserts: Optional[List[Union[ProfileMemoryRecord, EpisodicMemoryRecord, SemanticMemoryRecord]]] = Field(default=[], description="Records to add or update."),
        deletions: Optional[List[str]] = Field(default=[], description="String keys of memories to completely remove.")
    ) -> Command[Any]:
        """Persist key information from the current conversation into long-term memory."""
        try:
            target_dict = {}
            upserts = upserts or []
            deletions = deletions or []

            # 1. 统一处理 Upserts (依赖内部封装的 dedup_key)
            for record in upserts:
                key = getattr(record, "dedup_key", None)
                if key:
                    target_dict[key] = record

            # 2. 统一处理 Deletions (赋值为 None 触发 Reducer 删除)
            for delete_key in deletions:
                if delete_key:
                    target_dict[delete_key.strip().lower()] = None

            # 3. Persist to Store via MemoryManager
            await memory_manager.commit_incremental_memory(
                task_type=target,
                memory_value=target_dict,
                reason=reason or "Explicit request to organize memory"
            )

            # 4. Construct strong-typed update payload for LangGraph state
            update_payload: MemoryState = {
                "organization_reason": reason or "Explicit request to organize memory",
                "messages": [
                    ToolMessage(
                        content=f"Memory reorganized for target '{target}'. Upserted {len(upserts)} items, Deleted {len(deletions)} items. Reason: {reason}",
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }

            return Command(update=update_payload, graph="main")

        except Exception as e:
            logger.exception(f"Failed to organize memory for target {target}. Error: {e}")

            error_payload: MemoryState = {
                "messages": [
                    ToolMessage(
                        content=f"Failed to organize memory for {target}. Error: {e}",
                        tool_call_id=runtime.tool_call_id,
                        status="error"
                    )
                ]
            }
            return Command(update=error_payload)

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
                    "thread_id": getattr(runtime.context, "thread_id", None),
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

    def _truncate_tool_args(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """
        Runtime optimization: Truncate large tool arguments for a leaner context.
        Implements the logic from summarization.py to clip 'write_file' and 'edit_file' args.
        """
        MAX_ARG_LENGTH = 2000
        TRUNCATION_TEXT = "...(argument truncated)"

        # We only truncate messages that are not in the most recent window (approx 20)
        cutoff = len(messages) - 20
        if cutoff <= 0:
            return messages

        truncated_messages = []
        for i, msg in enumerate(messages):
            if i < cutoff and isinstance(msg, AIMessage) and msg.tool_calls:
                new_tool_calls = []
                for tc in msg.tool_calls:
                    if tc.get("name") in {"write_file", "edit_file"}:
                        args = tc.get("args", {})
                        if isinstance(args, dict):
                            new_args = {}
                            for k, v in args.items():
                                if isinstance(v, str) and len(v) > MAX_ARG_LENGTH:
                                    new_args[k] = v[:20] + TRUNCATION_TEXT
                                else:
                                    new_args[k] = v
                            new_tool_calls.append({**tc, "args": new_args})
                        else:
                            new_tool_calls.append(tc)
                    else:
                        new_tool_calls.append(tc)

                truncated_msg = msg.model_copy()
                truncated_msg.tool_calls = new_tool_calls
                truncated_messages.append(truncated_msg)
            else:
                truncated_messages.append(msg)
        return truncated_messages

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        runtime = request.runtime

        # 1. [Conversation Compaction] Replace full history with effective context: [Summary + Incremental]
        # This implements the core logic from summarization.py to prevent context bloat.
        effective_messages = await memory_manager.get_effective_summary_context()

        if len(effective_messages) > 0:
        # 2. [Runtime Optimization] Truncate large tool arguments in the effective window
            effective_messages = self._truncate_tool_args(effective_messages)
            request = request.override(
                messages=effective_messages
            )

        # 3. Build injected context (Memory + Environment)
        memory_body = await memory_manager.get_agent_context()
        memory_context_str = get_middleware_prompt("context").format(agent_memory=memory_body)
        env_context_str = self._format_environment_context(runtime)

        injected_context_str = f"""
        {ORGANIZE_MEMORY_SYSTEM_PROMPT}

        {env_context_str}

        {memory_context_str}
        """.strip()

        # 4. Override both messages and system prompt for the final LLM call
        request = request.override(
            system_message=append_to_system_message(request.system_message, injected_context_str)
        )

        try:
            response = await handler(request)
            last_message = response.result[-1]
            finish_reason = ((last_message.get('response_metadata') if isinstance(last_message, dict) else getattr(last_message, 'response_metadata', {})) or {}).get('finish_reason')
            if finish_reason: 
                payload = {
                    "current_message_count": len(response.result) + len(request.messages),
                    "current_token_count": ((last_message.get('usage_metadata') if isinstance(last_message, dict) else getattr(last_message, 'usage_metadata', {})) or {}).get('total_tokens', 0),
                    "msg_threshold": 20,
                    "token_threshold": 20000
                }
                await hub.emit(
                    "event_summary",
                    payload
                )
            return response
        except Exception as e:
            logger.exception("ContextEngineeringMiddleware model call failed: %s", e)
            return ModelResponse(result=[AIMessage(content=f"Model call failed: {type(e).__name__}: {e}")])

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> None:  # ty: ignore[invalid-method-override]
        """Load memory content before agent execution (synchronous).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        pass

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> None:  # ty: ignore[invalid-method-override]
        """Load memory content before agent execution.

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # 启动job调度
        await hub.start()
        # 注入运行时参数
        runtime_state_bridge.update_dynamic_deps("thread_id",config['configurable']['thread_id'])
        runtime_state_bridge.update_dynamic_deps("user_id",runtime.context.user_id)
