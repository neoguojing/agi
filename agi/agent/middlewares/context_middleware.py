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
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

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
from agi.scheduler.memory_task.memory_state import MemoryState,MemoryManager
from agi.scheduler import runtime_state_bridge,hub


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

            # 3. 构造强类型且无冗余字段的 Update Payload
            update_payload: MemoryState = {
                "organization_reason": reason,
                "messages": [
                    ToolMessage(
                        content=f"Memory reorganized for target '{target}'. Upserted {len(upserts)} items, Deleted {len(deletions)} items. Reason: {reason}", 
                        tool_call_id=runtime.tool_call_id
                    )
                ]
            }

            # 4. 根据 Target 将重组好的数据写入 State 对应的字段
            if target == "profile":
                update_payload["profile_records"] = target_dict  # type: ignore
            elif target == "episodic":
                update_payload["episodic_records"] = target_dict # type: ignore
            elif target == "semantic":
                update_payload["semantic_records"] = target_dict # type: ignore

            return Command(update=update_payload)

        except Exception as e:
            logger.exception(f"Failed to organize memory for target {target}. Error: {e}")
            
            # 异常情况也遵循强类型的 State 返回
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

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:

        runtime = request.runtime
        
        memory_manager = MemoryManager(state = request.state)
        memory_body = memory_manager.get_agent_context()

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
        runtime_state_bridge.update_dynamic_deps("thread_id",config['metadata']['thread_id'])
        runtime_state_bridge.update_dynamic_deps("user_id",runtime.context.user_id)

