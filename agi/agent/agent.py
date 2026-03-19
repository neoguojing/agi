import os
import uuid
import sqlite3
import aiosqlite
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from contextlib import asynccontextmanager

from agi.rag.retriever import MultiCollectionRAGManager
from agi.agent.models import ModelProvider
from agi.agent.middlewares import DebugLLMContextMiddleware, ContextEngineeringMiddleware
from agi.agent.tools import buildin_tools
from agi.agent.subagents import buildin_agents
from agi.agent.context import Context

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

# --- Constants & Config ---
DB_PATH = "agent_state.db"

class DeepAgentBuilder:
    """Fluent Builder for Agent Configuration"""
    def __init__(self, name: str = "main"):
        self.name = name
        self.llm = ModelProvider.get_chat_model(provider="ollama", model_name="qwen3.5:9b")
        self.embd = ModelProvider.get_embeddings(provider="ollama", model_name="embeddinggemma:latest")
        self.system_prompt = None
        self.tools = list(buildin_tools)
        self.subagents = list(buildin_agents)
        self.memory_paths = ["./memories/AGENT.md"]
        self.backend = LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
        
    def with_model(self, provider: str, name: str):
        self.llm = ModelProvider.get_chat_model(provider, name)
        return self

    def with_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        return self

    def build_options(self) -> Dict[str, Any]:
        """Returns the dictionary of parameters required for create_deep_agent"""
        return {
            "name": self.name,
            "model": self.llm,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "subagents": self.subagents,
            "backend": self.backend,
            "memory": self.memory_paths,
            "middleware": [
                DebugLLMContextMiddleware(),
                ContextEngineeringMiddleware(extractor_model=self.llm)
            ],
            "context_schema": Context
        }

class DeepAgentManager:
    """Manager to handle Sync and Async Agent instances and persistence"""
    def __init__(self, builder: DeepAgentBuilder):
        self.builder = builder
        self._sync_agent = None
        self._async_agent = None
        self._async_conn: Optional[aiosqlite.Connection] = None

    # --- Synchronous Logic ---
    def get_sync_agent(self):
        if self._sync_agent is None:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            # Standard LangGraph SqliteSaver
            checkpointer = SqliteSaver(conn)
            store = SqliteStore(conn=conn)
            # Note: AsyncSqliteStore usually requires an async connection, 
            # for pure sync, you might omit 'store' or use a compatible sync store.
            self._sync_agent = create_deep_agent(
                **self.builder.build_options(),
                checkpointer=checkpointer,
                store=store
            )
        return self._sync_agent

    # --- Asynchronous Logic ---
    async def _init_async_conn(self):
        if self._async_conn is None:
            self._async_conn = await aiosqlite.connect(DB_PATH)
            await self._async_conn.execute("PRAGMA journal_mode=WAL")
            await self._async_conn.execute("PRAGMA synchronous=NORMAL")
            await self._async_conn.commit()
        return self._async_conn

    async def get_async_agent(self):
        if self._async_agent is None:
            conn = await self._init_async_conn()
            saver = AsyncSqliteSaver(conn=conn)
            store = AsyncSqliteStore(conn=conn)
            
            self._async_agent = create_deep_agent(
                **self.builder.build_options(),
                checkpointer=saver,
                store=store
            )
        return self._async_agent

    async def close(self):
        if self._async_conn:
            await self._async_conn.close()

# --- Global Manager Instance ---
agent_manager = DeepAgentManager(DeepAgentBuilder())

# --- Unified Interface Functions ---

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
        **kwargs
    )

async def stream_agent_async(state: Dict, config: Dict = None, context: Context = None, **kwargs) -> AsyncGenerator:
    agent = await agent_manager.get_async_agent()
    async for part in agent.astream(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        stream_mode=kwargs.pop("stream_mode", ["messages", "updates"]),
        version="v2",
        **kwargs
    ):
        yield part

def invoke_agent_sync(state: Dict, config: Dict = None, context: Context = None, **kwargs):
    agent = agent_manager.get_sync_agent()
    return agent.invoke(
        state,
        config=_prepare_config(config, state),
        context=_prepare_context(context, state),
        **kwargs
    )

# --- Execution Entry ---

if __name__ == '__main__':
    # Example Sync Call
    print("--- Running Sync ---")
    sync_res = invoke_agent_sync({"messages": [{"role": "user", "content": "ls"}]})
    print(f"Result: {sync_res['messages'][-1].content}")

    # Example Async Call
    async def main():
        print("\n--- Running Async ---")
        async for chunk in stream_agent_async({"messages": [{"role": "user", "content": "whoami"}]}):
            print(f"Stream Chunk: {chunk}")
        await agent_manager.close()

    asyncio.run(main())