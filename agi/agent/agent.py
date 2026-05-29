"""Main AGI agent assembly and LangGraph entrypoints.

This module only builds and runs the main agent.  Async subagent graphs are
constructed in ``agi.agent.subagents`` so their lifecycle and dependencies stay
independent from the main-agent runtime.
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
AgentOptions = dict[str, Any]
RuntimeResources = dict[str, Any]


class AgentRuntime:
    """Build, persist, and run only the top-level AGI DeepAgent graph."""

    def __init__(self, db_uri: str = DB_URI, backend: Any = make_backend):
        self.db_uri = db_uri
        self.backend = backend
        self._sync_agent: Any = None
        self._async_agent: Any = None
        self._async_connections: list[Any] = []

    def main_options(self) -> AgentOptions:
        """Return graph options for the top-level assistant."""

        return self._options(
            name="main",
            model=ModelProvider.get_chat_model(),
            fallback_model=ModelProvider.get_falback_model(),
            tools=list(buildin_tools),
            system_prompt="",
            subagents=list(buildin_agents),
            async_subagents=list(buildin_async_agents),
        )

    def compile_graph(self, options: AgentOptions, resources: RuntimeResources | None = None):
        """Compile one DeepAgent graph from already prepared options."""

        kwargs = dict(options)
        if resources is not None:
            kwargs.update(checkpointer=resources["checkpointer"], store=resources["store"])
        logger.debug(
            "Compiling agent graph",
            extra={
                "agent_name": kwargs.get("name"),
                "tools": len(kwargs.get("tools", [])),
                "subagents": len(kwargs.get("subagents", [])),
                "middleware": len(kwargs.get("middleware", [])),
                "persistent": resources is not None,
            },
        )
        return create_deep_agent(**kwargs)

    def compile_main(self, resources: RuntimeResources | None = None):
        """Compile the top-level assistant graph."""

        return self.compile_graph(self.main_options(), resources)

    def sync_agent(self):
        if self._sync_agent is None:
            self._sync_agent = self.compile_main(self._sync_resources())
        return self._sync_agent

    async def async_agent(self):
        if self._async_agent is None:
            resources = await self._async_resources()
            self._async_connections = resources["connections"]
            self._async_agent = self.compile_main(resources)
        return self._async_agent

    async def close(self) -> None:
        for conn in self._async_connections:
            close = getattr(conn, "close", None)
            if close is None:
                continue
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._async_connections = []
        self._async_agent = None

    def _options(
        self,
        *,
        name: str,
        model: Any,
        fallback_model: Any,
        tools: Sequence[Any],
        system_prompt: str,
        subagents: Sequence[Any],
        async_subagents: Sequence[Any],
        extra_middleware: Sequence[Any] = (),
    ) -> AgentOptions:
        middleware = [
            ContextEngineeringMiddleware(backend=self.backend, llm=fallback_model),
            ModelFallbackMiddleware(model, *ModelProvider.get_chat_models()[1:]),
            MultimodalBase64Middleware(),
            DebugLLMContextMiddleware(),
            *extra_middleware,
        ]
        return {
            "name": name,
            "context_schema": Context,
            "model": model,
            "backend": self.backend,
            "tools": list(tools),
            "system_prompt": system_prompt,
            "subagents": [*subagents, *async_subagents],
            "middleware": middleware,
        }

    def _sync_resources(self) -> RuntimeResources:
        pool = ConnectionPool(conninfo=self.db_uri)
        checkpointer = PostgresSaver(conn=pool)
        store = PostgresStore(conn=pool)
        checkpointer.setup()
        store.setup()
        return {"checkpointer": checkpointer, "store": store, "connections": [pool]}

    async def _async_resources(self) -> RuntimeResources:
        pool = AsyncConnectionPool(conninfo=self.db_uri, open=True)
        checkpointer = AsyncPostgresSaver(conn=pool)
        store = AsyncPostgresStore(conn=pool)

        # Setup must run on independent short-lived connections so it is not
        # affected by transactions held by the runtime pool.
        async with AsyncPostgresSaver.from_conn_string(self.db_uri) as setup_checkpointer:
            await setup_checkpointer.setup()
        async with AsyncPostgresStore.from_conn_string(self.db_uri) as setup_store:
            await setup_store.setup()

        return {"checkpointer": checkpointer, "store": store, "connections": [pool]}


agent_runtime = AgentRuntime()


def _prepare_config(config: dict[str, Any] | None, state: Mapping[str, Any]) -> dict[str, Any]:
    if config is not None:
        return config
    return {"configurable": {"thread_id": state.get("thread_id", str(uuid.uuid4()))}}


def _prepare_context(context: Context | None, state: Mapping[str, Any]) -> Context:
    if context is not None:
        return context
    return Context(user_id=state.get("user_id"), conversation_id=state.get("conversation_id"))


async def invoke_agent_async(state: dict[str, Any], config: dict[str, Any] | None = None, context: Context | None = None, **kwargs: Any):
    agent = await agent_runtime.async_agent()
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
    agent = await agent_runtime.async_agent()
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
    agent = agent_runtime.sync_agent()
    return agent.invoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs,
    )


@contextlib.asynccontextmanager
async def main_graph(config: Any = None):  # noqa: ARG001 - LangGraph passes RunnableConfig.
    """LangGraph factory for the top-level assistant."""

    runtime = AgentRuntime()
    try:
        yield await runtime.async_agent()
    finally:
        await runtime.close()


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
        await agent_runtime.close()

    asyncio.run(main())
