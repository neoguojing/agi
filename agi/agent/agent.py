"""Main AGI agent assembly and LangGraph entrypoints.

This module intentionally keeps the agent lifecycle small and explicit:

1. Build an ``AgentProfile`` (main agent or a promoted subagent).
2. Create persistence resources.
3. Compile one DeepAgent graph.

The exported ``main_graph`` factory is used by ``langgraph dev``. Async subagents
registered in the same ``langgraph.json`` can omit ``url`` and communicate via
LangGraph SDK ASGI transport in the same process; adding ``url`` to an async
subagent switches it to HTTP transport for direct cross-server communication.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field, replace
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


@dataclass(slots=True)
class AgentRuntimeResources:
    """Checkpoint/store handles plus the connections that own them."""

    checkpointer: Any
    store: Any
    connections: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class AgentProfile:
    """Declarative configuration for a DeepAgent graph."""

    name: str = "main"
    model: Any = None
    fallback_model: Any = None
    embeddings: Any = None
    system_prompt: str = ""
    tools: list[Any] = field(default_factory=list)
    subagents: list[dict[str, Any]] = field(default_factory=list)
    async_subagents: list[dict[str, Any]] = field(default_factory=list)
    middleware: list[Any] = field(default_factory=list)
    backend: Any = make_backend
    context_schema: type[Context] = Context

    @classmethod
    def main(cls) -> "AgentProfile":
        """Build the default top-level agent profile."""

        return cls(
            name="main",
            model=ModelProvider.get_chat_model(),
            fallback_model=ModelProvider.get_falback_model(),
            embeddings=ModelProvider.get_embeddings(provider="ollama", model_name="embeddinggemma:latest"),
            tools=list(buildin_tools),
            subagents=list(buildin_agents),
            async_subagents=list(buildin_async_agents),
        )

    @classmethod
    def from_subagent(
        cls,
        subagent: Mapping[str, Any],
        *,
        include_default_subagents: bool = False,
        async_subagents: Sequence[Mapping[str, Any]] | None = None,
    ) -> "AgentProfile":
        """Promote a subagent spec into a first-class/main graph profile.

        A promoted subagent keeps its own prompt, tools, model, and middleware,
        but can be registered in ``langgraph.json`` and called directly as a
        normal LangGraph assistant. By default it does not inherit sibling
        subagents, keeping the graph focused and avoiding delegation cycles.
        """

        base = cls.main()
        return replace(
            base,
            name=str(subagent["name"]),
            model=subagent.get("model", base.model),
            system_prompt=subagent.get("system_prompt", ""),
            tools=list(subagent.get("tools", [])),
            subagents=list(buildin_agents) if include_default_subagents else [],
            async_subagents=[dict(item) for item in async_subagents or ()],
            middleware=list(subagent.get("middleware", [])),
        )

    def with_system_prompt(self, prompt: str) -> "AgentProfile":
        return replace(self, system_prompt=prompt)

    def with_middleware(self, middleware: Sequence[Any]) -> "AgentProfile":
        return replace(self, middleware=[*self.middleware, *middleware])

    def options(self) -> dict[str, Any]:
        """Return kwargs accepted by ``create_deep_agent``."""

        return {
            "name": self.name,
            "context_schema": self.context_schema,
            "model": self.model,
            "backend": self.backend,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "subagents": [*self.subagents, *self.async_subagents],
            "middleware": build_main_middleware(self.model, self.fallback_model, self.middleware),
        }


def build_main_middleware(llm: Any, fallback_llm: Any, extra: Sequence[Any] = ()) -> list[Any]:
    """Middleware shared by main graphs and promoted subagent graphs."""

    return [
        ContextEngineeringMiddleware(backend=make_backend, llm=fallback_llm),
        ModelFallbackMiddleware(llm, *ModelProvider.get_chat_models()[1:]),
        MultimodalBase64Middleware(),
        DebugLLMContextMiddleware(),
        *extra,
    ]


class AgentPersistence:
    """Creates Postgres checkpoint/store resources for sync and async graphs."""

    def __init__(self, db_uri: str = DB_URI):
        self.db_uri = db_uri

    def create_sync(self) -> AgentRuntimeResources:
        pool = ConnectionPool(conninfo=self.db_uri)
        checkpointer = PostgresSaver(conn=pool)
        store = PostgresStore(conn=pool)
        checkpointer.setup()
        store.setup()
        return AgentRuntimeResources(checkpointer=checkpointer, store=store, connections=[pool])

    async def create_async(self) -> AgentRuntimeResources:
        pool = AsyncConnectionPool(conninfo=self.db_uri, open=True)
        checkpointer = AsyncPostgresSaver(conn=pool)
        store = AsyncPostgresStore(conn=pool)

        # Setup must run on independent short-lived connections so it is not
        # affected by transactions held by the runtime pool.
        async with AsyncPostgresSaver.from_conn_string(self.db_uri) as setup_checkpointer:
            await setup_checkpointer.setup()
        async with AsyncPostgresStore.from_conn_string(self.db_uri) as setup_store:
            await setup_store.setup()

        return AgentRuntimeResources(checkpointer=checkpointer, store=store, connections=[pool])

    async def close(self, connections: Sequence[Any]) -> None:
        for conn in connections:
            close = getattr(conn, "close", None)
            if close is None:
                continue
            result = close()
            if hasattr(result, "__await__"):
                await result


class AgentRuntime:
    """Small lifecycle wrapper for one profile."""

    def __init__(self, profile: AgentProfile | None = None, persistence: AgentPersistence | None = None):
        self.profile = profile or AgentProfile.main()
        self.persistence = persistence or AgentPersistence()
        self._sync_agent: Any = None
        self._async_agent: Any = None
        self._async_connections: list[Any] = []

    def compile(self, resources: AgentRuntimeResources | None = None):
        kwargs = self.profile.options()
        if resources is not None:
            kwargs.update(checkpointer=resources.checkpointer, store=resources.store)
        return create_deep_agent(**kwargs)

    def sync_agent(self):
        if self._sync_agent is None:
            self._sync_agent = self.compile(self.persistence.create_sync())
        return self._sync_agent

    async def async_agent(self):
        if self._async_agent is None:
            resources = await self.persistence.create_async()
            self._async_connections = resources.connections
            self._async_agent = self.compile(resources)
        return self._async_agent

    async def close(self) -> None:
        await self.persistence.close(self._async_connections)
        self._async_connections = []
        self._async_agent = None


agent_runtime = AgentRuntime()


def build_agent(profile: AgentProfile | None = None, resources: AgentRuntimeResources | None = None):
    """Compile any profile as a standalone DeepAgent graph."""

    return AgentRuntime(profile or AgentProfile.main()).compile(resources)


def build_agent_from_subagent(subagent: Mapping[str, Any], **kwargs: Any):
    """Compile a subagent spec as a standalone/main agent graph."""

    return build_agent(AgentProfile.from_subagent(subagent, **kwargs))


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
    """LangGraph factory for the top-level assistant.

    Register this graph together with async subagents in ``langgraph.json``.
    Local async subagents should omit ``url`` so LangGraph SDK can use ASGI
    in-process transport; subagents with ``url`` use HTTP transport instead.
    """

    runtime = AgentRuntime(AgentProfile.main())
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
