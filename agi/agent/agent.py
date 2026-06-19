"""Main AGI agent assembly and LangGraph entrypoints.

This module builds the main agent directly with ``create_deep_agent``.  Async
subagent graphs are constructed in ``agi.agent.subagents`` so their lifecycle
and dependencies stay independent from the main-agent helpers here.
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any,Optional
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from dataclasses import dataclass

from agi.agent.middlewares import (
    ContextEngineeringMiddleware,
    DebugLLMContextMiddleware,
    MultimodalBase64Middleware,
    ToolContextMiddleware,
)
from agi.agent.middlewares.tool_context_middleware import ToolContextMiddleware
from agi.agent.models import ModelProvider
from agi.agent.subagents import buildin_agents, buildin_async_agents, make_backend
from agi.agent.tools import buildin_tools
from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from agi.config import DEFAULT_DB_URI
from agi.agent.deep_agent import create_deep_agent
from agi.scheduler import runtime_state_bridge

logger = logging.getLogger(__name__)

DB_URI = os.getenv("AGI_CHECKPOINT_DB_URI", DEFAULT_DB_URI)
RuntimeResources = dict[str, Any]

_async_agent: Any = None
_async_connections: list[Any] = []

@dataclass
class Context:
    user_id: Optional[str] = "admin"
    thread_id: Optional[str] = str(uuid.uuid4())


    
async def create_async_resources(db_uri: str = DB_URI) -> RuntimeResources:
    pool = AsyncConnectionPool(conninfo=db_uri, open=True)
    checkpointer = AsyncPostgresSaver(conn=pool)
    store = AsyncPostgresStore(conn=pool)

    # Setup must run on independent short-lived connections so it is not
    # affected by transactions held by the runtime pool.
    async with AsyncPostgresSaver.from_conn_string(db_uri) as setup_checkpointer:
        await setup_checkpointer.setup()
    async with AsyncPostgresStore.from_conn_string(db_uri) as setup_store:
        await setup_store.setup()

    return {"checkpointer": checkpointer, "store": store, "connections": [pool]}


async def close_connections(connections: Sequence[Any]) -> None:
    for conn in connections:
        close = getattr(conn, "close", None)
        if close is None:
            continue
        result = close()
        if hasattr(result, "__await__"):
            await result


def create_main_agent(
    *,
    checkpointer: Any = None,
    store: Any = None,
    backend: Any = make_backend,
):
    """Create the top-level AGI agent by calling ``create_deep_agent`` directly."""

    llm = ModelProvider.get_chat_model()
    fallback_llm = ModelProvider.get_falback_model()
    kwargs = {
        "name": "main",
        "context_schema": Context,
        "model": llm,
        "backend": backend,
        "tools": list(buildin_tools),
        "system_prompt": "",
        "subagents": [*buildin_agents, *buildin_async_agents],
        "middleware": [
            ContextEngineeringMiddleware(backend=backend, llm=fallback_llm),
            ToolContextMiddleware(backend=backend),
            ModelFallbackMiddleware(llm, *ModelProvider.get_chat_models()[1:]),
            MultimodalBase64Middleware(),
            DebugLLMContextMiddleware(),
        ],
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if store is not None:
        kwargs["store"] = store
    logger.debug(
        "Creating main agent graph",
        extra={
            "tools": len(kwargs["tools"]),
            "subagents": len(kwargs["subagents"]),
            "middleware": len(kwargs["middleware"]),
            "persistent": checkpointer is not None or store is not None,
        },
    )
    return create_deep_agent(**kwargs)


async def get_async_agent() -> CompiledStateGraph:
    global _async_agent, _async_connections
    if _async_agent is None:
        resources = await create_async_resources()
        _async_connections = resources["connections"]
        _async_agent = create_main_agent(
            checkpointer=resources["checkpointer"],
            store=resources["store"],
        )
        runtime_state_bridge.update_dynamic_deps("graph",_async_agent)
        runtime_state_bridge.update_dynamic_deps("store",resources["store"])
        runtime_state_bridge.update_dynamic_deps("backend",make_backend)
    return _async_agent


def _prepare_config(config: dict[str, Any] | None, state: Mapping[str, Any]) -> dict[str, Any]:
    if config is not None:
        return config
    return {"configurable": {"thread_id": state.get("thread_id", str(uuid.uuid4()))}}

def _prepare_context(context: Context | None, state: Mapping[str, Any]) -> Context:
    if context is not None:
        return context
    return Context(user_id=state.get("user_id"), thread_id=state.get("thread_id"))


async def invoke_agent_async(state: dict[str, Any], config: dict[str, Any] | None = None, context: Context | None = None, **kwargs: Any):
    agent = await get_async_agent()
    return await agent.ainvoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs,
    )


async def stream_agent_async(
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
    context: Context | None = None,
    **kwargs: Any,
) -> AsyncGenerator[Any, None]:
    agent = await get_async_agent()
    async for part in agent.astream(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        stream_mode=kwargs.pop("stream_mode", ["messages", "updates"]),
        version="v2",
        **kwargs,
    ):
        yield part

    