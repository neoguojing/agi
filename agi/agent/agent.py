import os
import uuid
from typing import List, Optional, Dict, Any,AsyncGenerator
from agi.rag.retriever import MultiCollectionRAGManager
from agi.agent.models import ModelProvider
from agi.agent.middlewares import DebugLLMContextMiddleware,ContextEngineeringMiddleware
from agi.agent.tools import buildin_tools
from agi.agent.subagents import buildin_agents
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from agi.agent.context import Context
from functools import lru_cache
@lru_cache(maxsize=None)
def get_connection():
    return sqlite3.connect("checkpoint.db", check_same_thread=False)


class DeepAgentBuilder:
    def __init__(self, name: str = "main"):
        self.name = name
        # 模型配置
        self._llm = ModelProvider.get_chat_model(provider="ollama",model_name="qwen3.5:9b")
        self._embd = ModelProvider.get_embeddings(provider="ollama",model_name="embeddinggemma:latest")  # 你的 Embedding 初始化
        # 实例化 KM：将 Embedding 模型注入其中
        self._km = MultiCollectionRAGManager()

        self._system_prompt = "You are a helpful AI assistant."
        self._backend = LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
        
        # 向量库 (KM) 配置
        self._km_tool_collection = "agent_tools"

        # 路径与插件
        self._tools_dir = "/data/my_agent/hot_tools"
        self._skills_dir = "/data/my_agent/hot_skills"
        self._memory_paths = ["./memories/AGENT.md"]
        self._basic_tools = buildin_tools
        self._subagents = buildin_agents
        self._middleware = [
            DebugLLMContextMiddleware(),
            ContextEngineeringMiddleware(extractor_model=self._llm)
        ]
        
        # 基础设施
        self._checkpointer = SqliteSaver(get_connection())
        
        self._store = None
        self._interrupt_on = {}

    # --- 1. 模型初始化 (LLM & Embd) ---
    def set_model(self, provider: str, model_name: str):
        self._llm = ModelProvider.get_chat_model(provider, model_name)
        return self
    
    def set_embd(self, provider: str, model_name: str):
        """核心：确保你的 Embedding 初始化不丢失"""
        self._embd = ModelProvider.get_embeddings(provider, model_name)
        return self

    # --- 3. 基础能力配置 ---
    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt
        return self

    def with_hot_reload(self, tools_path: str = "./hot_tools", skills_path: str = "./hot_skills"):
        self._tools_dir = tools_path
        self._skills_dir = skills_path
        os.makedirs(tools_path, exist_ok=True)
        os.makedirs(skills_path, exist_ok=True)
        return self

    def add_basic_tools(self, tools: List):
        self._basic_tools.extend(tools)
        return self

    def add_subagents(self, subagents: List[Dict]):
        self._subagents.extend(subagents)
        return self

    # --- 4. 核心 Build 逻辑 ---
    def build(self):
        # A. 内部组装 KnowledgeManager (组合 Embd)
        if not self._embd:
            raise ValueError("Embedding model (set_embd) is required before building KM.")

        final_middleware = self._middleware
        # E. 调用 create_deep_agent (符合你提供的源码定义)
        return create_deep_agent(
            name=self.name,
            model=self._llm,
            tools=self._basic_tools,
            system_prompt=self._system_prompt,
            middleware=final_middleware,
            subagents=self._subagents,
            backend=self._backend,
            memory=self._memory_paths,
            checkpointer=self._checkpointer,
            store=self._store,
            interrupt_on=self._interrupt_on,
            context_schema=Context
        )

deep_agent = DeepAgentBuilder().build()

async def invoke_agent(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    context: Optional[Context] = None,
    **kwargs
):
    if config is None:
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4())
            }
        }

    if context is None:
        context = Context(
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id")
        )

    return await deep_agent.invoke(
        state,
        config=config,
        context=context,
        **kwargs
    )

async def stream_agent(
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    context: Optional[Context] = None,
    stream_mode: Optional[list] = None,
    **kwargs
) -> AsyncGenerator[Any, None]:

    if config is None:
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4())
            }
        }

    if context is None:
        context = Context(
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id")
        )

    if stream_mode is None:
        stream_mode = ["messages", "updates", "custom"]

    async for part in deep_agent.stream(
        state,
        config=config,
        context=context,
        stream_mode=stream_mode,
        version="v2",
        **kwargs
    ):
        yield part

if __name__ == '__main__':

    # 运行
    print("🚀 Agent 组装完毕，动态工具监控已启动...")
    result = invoke_agent(
        {"messages": [{"role": "user", "content": "执行ls 命令，并返回结果。"}]},
        context=Context(user_id="1",conversation_id="test")
    )
    
    print(result)
    print(result["messages"][-1].content)