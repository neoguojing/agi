"""Main AGI agent assembly and LangGraph entrypoints.

This module builds the main agent directly with ``create_deep_agent``.  Async
subagent graphs are constructed in ``agi.agent.subagents`` so their lifecycle
and dependencies stay independent from the main-agent helpers here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any

from psycopg_pool import AsyncConnectionPool, ConnectionPool

from agi.agent.context import Context
from agi.agent.middlewares import (
    ContextEngineeringMiddleware,
    DebugLLMContextMiddleware,
    MultimodalBase64Middleware,
)
from agi.agent.models import ModelProvider
from agi.agent.subagents import buildin_agents, buildin_async_agents, make_backend
from agi.agent.tools import buildin_tools
from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from .deep_agent import create_deep_agent

logger = logging.getLogger(__name__)

DEFAULT_DB_URI = "postgres://admin:123456@localhost:5432/langchain?sslmode=disable"
DB_URI = os.getenv("AGI_CHECKPOINT_DB_URI", DEFAULT_DB_URI)
RuntimeResources = dict[str, Any]

_sync_agent: Any = None
_async_agent: Any = None
_async_connections: list[Any] = []


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


def create_sync_resources(db_uri: str = DB_URI) -> RuntimeResources:
    pool = ConnectionPool(conninfo=db_uri)
    checkpointer = PostgresSaver(conn=pool)
    store = PostgresStore(conn=pool)
    checkpointer.setup()
    store.setup()
    return {"checkpointer": checkpointer, "store": store, "connections": [pool]}


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


def get_sync_agent():
    global _sync_agent
    if _sync_agent is None:
        resources = create_sync_resources()
        _sync_agent = create_main_agent(
            checkpointer=resources["checkpointer"],
            store=resources["store"],
        )
    return _sync_agent


async def get_async_agent():
    global _async_agent, _async_connections
    if _async_agent is None:
        resources = await create_async_resources()
        _async_connections = resources["connections"]
        _async_agent = create_main_agent(
            checkpointer=resources["checkpointer"],
            store=resources["store"],
        )
    return _async_agent


async def close_async_agent() -> None:
    global _async_agent, _async_connections
    await close_connections(_async_connections)
    _async_connections = []
    _async_agent = None

    return AgentRuntime(profile or AgentProfile.main()).compile(resources)

def _prepare_config(config: dict[str, Any] | None, state: Mapping[str, Any]) -> dict[str, Any]:
    if config is not None:
        return config
    return {"configurable": {"thread_id": state.get("thread_id", str(uuid.uuid4()))}}

def build_agent_from_subagent(subagent: Mapping[str, Any], **kwargs: Any):
    """Compile a subagent spec as a standalone/main agent graph."""

def _prepare_context(context: Context | None, state: Mapping[str, Any]) -> Context:
    if context is not None:
        return context
    return Context(user_id=state.get("user_id"), conversation_id=state.get("conversation_id"))


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


def invoke_agent_sync(state: dict[str, Any], config: dict[str, Any] | None = None, context: Context | None = None, **kwargs: Any):
    agent = get_sync_agent()
    return agent.invoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs,
    )


@contextlib.asynccontextmanager
async def main_graph(config: Any = None):  # noqa: ARG001 - LangGraph passes RunnableConfig.
    """LangGraph factory for the top-level assistant."""

    resources = await create_async_resources()
    try:
        yield create_main_agent(
            checkpointer=resources["checkpointer"],
            store=resources["store"],
        )
    finally:
        await close_connections(resources["connections"])


# Common LangGraph convention: allow ``./agi/agent/agent.py:graph`` too.
graph = main_graph


if __name__ == "__main__":
    print("--- Running Sync ---")
    sync_res = invoke_agent_sync({"messages": [{"role": "user", "content": "ls"}]})
    print(f"Result: {sync_res['messages'][-1].content}")

    async def main() -> None:
        print("\n--- Running Async ---")
        async for chunk in stream_agent_async({"messages": [{"role": "user", "content": "whoami"}]}):
            print(f"Stream Chunk: {chunk}")
        await close_async_agent()

    asyncio.run(main())
