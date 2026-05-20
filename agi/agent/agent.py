import asyncio
import logging
import sqlite3
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional
from pathlib import Path

import aiosqlite
from agi.agent.context import Context
from agi.agent.middlewares import (
    ContextEngineeringMiddleware,
    DebugLLMContextMiddleware,
    MemoryMiddleware,
    MultimodalBase64Middleware,
)
from agi.agent.models import ModelProvider
from agi.agent.prompt import BACKGROUD_SYSTEM_PROMPT
from agi.agent.subagents import buildin_agents, make_backend
from agi.agent.tools import buildin_tools
from agi.config import OLLAMA_DEFAULT_MODE,CACHE_DIR
from .deep_agent import create_deep_agent
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware, SummarizationToolMiddleware
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

logger = logging.getLogger(__name__)

DB_PATH_CHECKPOINT = Path(CACHE_DIR) / "agent_checkpoint.db"
DB_PATH_STRORE = Path(CACHE_DIR) / "agent_store.db"


@dataclass
class AgentRuntimeResources:
    """Container for runtime resources used by `create_deep_agent`."""

    checkpointer: Any
    store: Any
    connections: list[aiosqlite.Connection]


class AgentPersistenceFactory:
    """Responsible for checkpoint/store resource creation and cleanup only."""

    def __init__(self, checkpoint_db_path: str = DB_PATH_CHECKPOINT):
        self.checkpoint_db_path = checkpoint_db_path

    def create_sync_resources(self) -> AgentRuntimeResources:
        conn = sqlite3.connect(self.checkpoint_db_path, check_same_thread=False)
        return AgentRuntimeResources(
            checkpointer=SqliteSaver(conn),
            store=SqliteStore(conn=conn),
            connections=[],
        )

    async def create_async_resources(self) -> AgentRuntimeResources:
        conn_saver = await aiosqlite.connect(self.checkpoint_db_path)
        await conn_saver.execute("PRAGMA journal_mode=WAL")
        await conn_saver.execute("PRAGMA synchronous=NORMAL")

        saver = AsyncSqliteSaver(conn=conn_saver)
        await saver.setup()

        return AgentRuntimeResources(
            checkpointer=saver,
            store=InMemoryStore(),
            connections=[conn_saver],
        )

    async def close_async_connections(self, connections: list[aiosqlite.Connection]) -> None:
        for conn in connections:
            await conn.close()


class AgentMiddlewareFactory:
    """Responsible for middleware assembly only."""

    SUMMARY_TRIM_CONFIG = {
        "trigger": ("tokens", 30000),
        "keep": ("messages", 10),
        "max_length": 2000,
        "truncation_text": "...(truncated)",
    }

    @staticmethod
    def build_main(llm: Any, fallback_llm: Any, extra_middlewares: list[Any]) -> list[Any]:
        return [
            ContextEngineeringMiddleware(backend=make_backend,llm=fallback_llm),
            ModelFallbackMiddleware(llm,*ModelProvider.get_chat_models()),
            DebugLLMContextMiddleware(),
            MultimodalBase64Middleware(),
            *extra_middlewares,
        ]


class DeepAgentBuilder:
    """Responsible for agent-level configuration only (no runtime resources)."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.llm = ModelProvider.get_chat_model()
        self.fallback_llm = ModelProvider.get_falback_model()
        self.embd = ModelProvider.get_embeddings(provider="ollama", model_name="embeddinggemma:latest")

        self.system_prompt = ""
        self.tools = list(buildin_tools)
        self.subagents = list(buildin_agents)
        self.middlewares: list[Any] = []
        self.backend = make_backend

    def clone(self) -> "DeepAgentBuilder":
        new = DeepAgentBuilder(self.name)
        new.llm = self.llm
        new.fallback_llm = self.fallback_llm
        new.embd = self.embd
        new.system_prompt = self.system_prompt
        new.tools = list(self.tools)
        new.subagents = list(self.subagents)
        new.middlewares = list(self.middlewares)
        new.backend = self.backend
        return new

    def with_model(self, provider: str, name: str) -> "DeepAgentBuilder":
        self.llm = ModelProvider.get_chat_model()
        return self

    def with_system_prompt(self, prompt: str) -> "DeepAgentBuilder":
        self.system_prompt = prompt
        return self

    def with_middleware(self, middlewares: list[Any]) -> "DeepAgentBuilder":
        self.middlewares.extend(middlewares)
        return self

    def _build_base_options(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "context_schema": Context,
        }

    def build_options(self) -> Dict[str, Any]:
        return {
            **self._build_base_options(),
            "model": self.llm,
            "fallback_model": self.fallback_llm,
            "backend": self.backend,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "subagents": self.subagents,
            "middleware": AgentMiddlewareFactory.build_main(
                llm=self.llm,
                fallback_llm=self.fallback_llm,
                extra_middlewares=self.middlewares,
            ),
        }


class DeepAgentManager:
    """Responsible for agent runtime lifecycle and orchestration only."""

    def __init__(self, builder: DeepAgentBuilder, persistence_factory: Optional[AgentPersistenceFactory] = None):
        self.builder = builder
        self.persistence_factory = persistence_factory or AgentPersistenceFactory()

        self._sync_agent = None
        self._async_agent = None

        self._async_connections: list[aiosqlite.Connection] = []

    def get_sync_agent(self):
        if self._sync_agent is None:
            resources = self.persistence_factory.create_sync_resources()
            self._sync_agent = create_deep_agent(
                **self.builder.build_options(),
                checkpointer=resources.checkpointer,
                store=resources.store,
            )
        return self._sync_agent

    async def _init_async_agent(self):
        resources = await self.persistence_factory.create_async_resources()
        self._async_connections = resources.connections
        self._async_agent = create_deep_agent(
            **self.builder.build_options(),
            checkpointer=resources.checkpointer,
            store=resources.store,
        )

    async def get_async_agent(self):
        if self._async_agent is None:
            await self._init_async_agent()
        return self._async_agent

    async def close(self):
        await self.persistence_factory.close_async_connections(self._async_connections)
        self._async_connections = []
        self._async_agent = None


agent_manager = DeepAgentManager(DeepAgentBuilder())


def _prepare_config(config: Optional[Dict], state: Dict) -> Dict:
    return config or {"configurable": {"thread_id": state.get("thread_id", str(uuid.uuid4()))}}


def _prepare_context(context: Optional[Context], state: Dict) -> Context:
    return context or Context(user_id=state.get("user_id"), conversation_id=state.get("conversation_id"))


async def invoke_agent_async(state: Dict, config: Dict = None, context: Context = None, **kwargs):
    agent = await agent_manager.get_async_agent()
    return await agent.invoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs,
    )


async def stream_agent_async(state: Dict, config: Dict = None, context: Context = None, **kwargs) -> AsyncGenerator:
    agent = await agent_manager.get_async_agent()
    async for part in agent.astream(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        stream_mode=kwargs.pop("stream_mode", ["messages", "updates"]),
        version="v2",
        **kwargs,
    ):
        yield part


def invoke_agent_sync(state: Dict, config: Dict = None, context: Context = None, **kwargs):
    agent = agent_manager.get_sync_agent()
    return agent.invoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs,
    )


if __name__ == "__main__":
    print("--- Running Sync ---")
    sync_res = invoke_agent_sync({"messages": [{"role": "user", "content": "ls"}]})
    print(f"Result: {sync_res['messages'][-1].content}")

    async def main():
        print("\n--- Running Async ---")
        async for chunk in stream_agent_async({"messages": [{"role": "user", "content": "whoami"}]}):
            print(f"Stream Chunk: {chunk}")
        await agent_manager.close()

    asyncio.run(main())
