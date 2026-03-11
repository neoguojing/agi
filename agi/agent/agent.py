import os
from typing import List, Optional, Dict, Any
from watchdog.observers import Observer
from agi.rag.retriever import KnowledgeManager
from agi.rag.tool_retreiver import ToolRegistryManager
from agi.agent.models import ModelProvider
from agi.agent.middlewares.tool import CapabilityRegistry,ToolReloaderHandler,JITOrchestratorMiddleware
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

class DeepAgentBuilder:
    def __init__(self, name: str = "main"):
        self.name = name
        # 模型配置
        self._llm = ModelProvider.get_chat_model(provider="ollama",model_name="qwen3.5:9b")
        self._embd = ModelProvider.get_embeddings(provider="ollama",model_name="bge:latest")  # 你的 Embedding 初始化
        self._system_prompt = "You are a helpful AI assistant."
        self._backend = LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
        
        # 向量库 (KM) 配置
        self._km_tool_collection = "agent_tools"

        # 路径与插件
        self._tools_dir = "/data/my_agent/hot_tools"
        self._skills_dir = "/data/my_agent/hot_skills"
        self._memory_paths = []
        self._basic_tools = []
        self._subagents = []
        self._middleware = []
        
        # 基础设施
        self._checkpointer = None
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
        
        # 实例化 KM：将 Embedding 模型注入其中
        km = KnowledgeManager(
            data_path="/data/my_agents/km/",
            embedding=self._embd
        )

        # B. 初始化 JIT 体系
        tm = ToolRegistryManager(km, collection_name=self._km_tool_collection)
        reg = CapabilityRegistry(
            tools_dir=self._tools_dir, 
            skills_dir=self._skills_dir, 
            manager=tm
        )

        # C. 启动物理监听
        observer = Observer()
        observer.schedule(ToolReloaderHandler(reg), path=self._tools_dir)
        observer.start()

        # D. 组装中间件
        jit_mw = JITOrchestratorMiddleware(reg)
        final_middleware = [jit_mw] + self._middleware

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
            interrupt_on=self._interrupt_on
        )
    
if __name__ == '__main__':
    # 使用 Builder 完成全流程组装
    agent = (
        DeepAgentBuilder()
        # 4. 配置插件与热更新
        .set_system_prompt("你是一个全栈工程师，擅长利用动态工具解决问题。")
        # 5. 构建
        .build()
    )

    # 运行
    print("🚀 Agent 组装完毕，动态工具监控已启动...")
    agent.invoke({"messages": [{"role": "user", "content": "帮我看看 custom_tools 目录里有什么能用的代码混淆工具？"}]})