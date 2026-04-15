import os
import uuid
from pathlib import Path
import sqlite3
import aiosqlite
import asyncio
import logging
import traceback
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from contextlib import asynccontextmanager
from langchain.tools import ToolRuntime

from agi.rag.retriever import MultiCollectionRAGManager
from agi.agent.models import ModelProvider
from agi.agent.middlewares import (DebugLLMContextMiddleware, 
                                   ContextEngineeringMiddleware,
                                   BrowserMiddleware,
                                   MultimodalBase64Middleware,
                                   MemoryMiddleware)
from langchain.agents.middleware import (
    ModelFallbackMiddleware
)
from agi.agent.tools import buildin_tools
from agi.agent.subagents import buildin_agents,make_backend
from agi.agent.context import Context
from deepagents.backends.protocol import BACKEND_TYPES as BACKEND_TYPES
from agi.agent.prompt import  BACKGROUD_SYSTEM_PROMPT
from agi.config import OLLAMA_DEFAULT_MODE,CACHE_DIR

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from deepagents.backends import CompositeBackend
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend,StoreBackend,FilesystemBackend,StateBackend
from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        SummarizationToolMiddleware,
    )

logger = logging.getLogger(__name__)
# --- Constants & Config ---
DB_PATH_CHECKPOINT = "agent_checkpoint.db"
DB_PATH_STRORE = "agent_store.db"

class DeepAgentBuilder:
    """Fluent Builder for Agent Configuration"""
    def __init__(self, name: str = "main"):
        self.name = name
        self.llm = ModelProvider.get_chat_model(provider="ollama", model_name=OLLAMA_DEFAULT_MODE)
        self.fallback_llm = ModelProvider.get_chat_model(provider="ollama", model_name="qwen3.5:9b")

        self.embd = ModelProvider.get_embeddings(provider="ollama", model_name="embeddinggemma:latest")
        self.system_prompt = ""
        self.tools = list(buildin_tools)
        self.subagents = list(buildin_agents)
        self.middlewares = []
        # self.memory_paths = ["/memories/AGENT.md"]
        self.backend = make_backend
    
    def clone(self):
        new = DeepAgentBuilder(self.name)
        new.llm = self.llm
        new.fallback_llm = self.fallback_llm
        new.system_prompt = self.system_prompt
        new.tools = list(self.tools)
        new.subagents = list(self.subagents)
        new.middlewares = list(self.middlewares)
        return new

    def with_model(self, provider: str, name: str):
        self.llm = ModelProvider.get_chat_model(provider, name)
        return self

    def with_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        return self
    
    def with_middleware(self, middleware):
        self.middlewares.append(middleware)
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
            # "memory": self.memory_paths,
            "middleware": [
                ContextEngineeringMiddleware(backend=make_backend),
                ModelFallbackMiddleware(self.llm,self.fallback_llm),
                DebugLLMContextMiddleware(),
                MultimodalBase64Middleware()
            ],
            "context_schema": Context
        }
    
    def build_options_for_backgroud(self) -> Dict[str, Any]:
        """Returns the dictionary of parameters required for create_deep_agent"""

        summ = SummarizationMiddleware(model=self.llm, 
                                       backend=make_backend,
                                       trigger=("tokens", 30000),
                                       keep=("messages", 10),
                                       trim_tokens_to_summarize= {
                                            # "trigger": ("messages", 20),
                                            "trigger": ("tokens", 30000),
                                            "keep": ("messages", 10),
                                            "max_length": 2000,
                                            "truncation_text": "...(truncated)",
                                        })

        return {
            "name": "backgroud",
            "model": self.fallback_llm,
            "backend": self.backend,
            "middleware": [
                SummarizationToolMiddleware(summ),
                DebugLLMContextMiddleware(),
                *self.middlewares
            ],
            "context_schema": Context
        }

class DeepAgentManager:
    """Manager to handle Sync and Async Agent instances and persistence"""
    def __init__(self, builder: DeepAgentBuilder):
        self.builder = builder
        self._sync_agent = None
        self._async_agent = None
        self._async_backgroud_agent = None
        self._async_conn: Optional[aiosqlite.Connection] = None

    # --- Synchronous Logic ---
    def get_sync_agent(self):
        if self._sync_agent is None:
            conn = sqlite3.connect(DB_PATH_CHECKPOINT, check_same_thread=False)
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
            # conn_store = await aiosqlite.connect(DB_PATH_STRORE)
            # await conn_store.execute("PRAGMA journal_mode=WAL")
            # await conn_store.execute("PRAGMA synchronous=NORMAL")

            saver = AsyncSqliteSaver(conn=conn_saver)
            # store = AsyncSqliteStore(conn=conn_store)
            # saver = MemorySaver()
            store = InMemoryStore()
            await saver.setup()  # 确保表结构已创建
            # await store.setup()  # 确保表结构已创建
            
            self._async_agent = create_deep_agent(
                **self.builder.build_options(),
                checkpointer=saver,
                store=store
            )
            
            # self._async_agent.get_graph().draw_png("agent.png")
            # ⚠️ 重要：你需要在程序退出时关闭这两个连接
            # 可以在 __del__ 或专门的 shutdown 方法中处理
            # self._connections_to_close = [conn_saver, conn_store]       

        return self._async_agent
    # 后台agent
    # 1.负责根据最新消息判断是否需要触发远期记忆提取
    # 2.负责根据最新消息判断是否需要触发消息压缩
    # 3.定期自检，判断上面任务是否执行，未执行则触发手动压缩
    async def get_backgroud_agent(self, config, interval: int = 30):
        if self._async_backgroud_agent is not None:
            return self._async_backgroud_agent   # ✅ 防重复创建

        if not self._async_agent:
            await self.get_async_agent()  # 确保主 agent 已初始化
        
        builder = self.builder.clone()
        agent_configs = (
            builder
            .with_system_prompt(BACKGROUD_SYSTEM_PROMPT)
            .with_middleware(MemoryMiddleware(backend=make_backend,
                                              checkpointer=self._async_agent.checkpointer,
                                              channels=self._async_agent.channels,
                                              config=config))
            .build_options_for_backgroud()
        )

        self._async_backgroud_agent = create_deep_agent(
            **agent_configs,
            checkpointer=self._async_agent.checkpointer,
            store=self._async_agent.store
        )

        self._bg_running = True
        logger.info("[BG] Background agent initialized. Starting loop...")
        bg_config = config or {"configurable": {"thread_id": uuid.uuid4().hex}}
        async def loop():
            while self._bg_running:
                try:
                    # 👉 关键：构造触发消息
                    trigger_input = {
                        "messages": [
                            {
                                "type": "human",
                                "content": "Background maintenance tick: analyze memory and compression needs."
                            }
                        ]
                    }

                    # 👉 关键：让 agent 自己决定做什么
                    result = await self._async_backgroud_agent.ainvoke(trigger_input,config=bg_config)
                    logger.info(f"[BG] Tick result: {result['messages'][-1].content}")

                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"[BG] error: {e}")

                await asyncio.sleep(interval)

        self._bg_task = asyncio.create_task(loop())

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
    await agent_manager.get_backgroud_agent(config)
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