import os
import uuid
import sqlite3
import aiosqlite
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from contextlib import asynccontextmanager

from agi.rag.retriever import MultiCollectionRAGManager
from agi.agent.models import ModelProvider
from agi.agent.middlewares import DebugLLMContextMiddleware, ContextEngineeringMiddleware,BrowserMiddleware
from agi.agent.tools import buildin_tools
from agi.agent.subagents import buildin_agents
from agi.agent.context import Context

from agi.config import OLLAMA_DEFAULT_MODE

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

# --- Constants & Config ---
DB_PATH_CHECKPOINT = "agent_checkpoint.db"
DB_PATH_STRORE = "agent_store.db"

class DeepAgentBuilder:
    """Fluent Builder for Agent Configuration"""
    def __init__(self, name: str = "main"):
        self.name = name
        self.llm = ModelProvider.get_chat_model(provider="ollama", model_name=OLLAMA_DEFAULT_MODE)
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
                BrowserMiddleware(ocr_engine=self.llm),
                ContextEngineeringMiddleware(extractor_model=self.llm),
                DebugLLMContextMiddleware()
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
    async def get_async_agent(self):
        if self._async_agent is None:
            # 1. 初始化主连接用于 Saver (检查点通常更关键)
            conn_saver = await aiosqlite.connect(DB_PATH_CHECKPOINT)
            await conn_saver.execute("PRAGMA journal_mode=WAL")
            await conn_saver.execute("PRAGMA synchronous=NORMAL")
            
            # 2. 初始化独立连接用于 Store (向量/记忆存储)
            # 指向同一个文件没问题，WAL 模式下并发安全
            conn_store = await aiosqlite.connect(DB_PATH_STRORE)
            await conn_store.execute("PRAGMA journal_mode=WAL")
            await conn_store.execute("PRAGMA synchronous=NORMAL")

            saver = AsyncSqliteSaver(conn=conn_saver)
            store = AsyncSqliteStore(conn=conn_store)

            store = AsyncSqliteStore(conn=conn_store)
            await saver.setup()  # 确保表结构已创建
            await store.setup()  # 确保表结构已创建
            
            self._async_agent = create_deep_agent(
                **self.builder.build_options(),
                checkpointer=saver,
                store=store
            )
            
            # ⚠️ 重要：你需要在程序退出时关闭这两个连接
            # 可以在 __del__ 或专门的 shutdown 方法中处理
            # self._connections_to_close = [conn_saver, conn_store]

        return self._async_agent

    async def close(self):
        if self._async_agent:
            for conn in getattr(self, "_connections_to_close", []):
                await conn.close()
            self._async_agent = None

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