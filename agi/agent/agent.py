from deepagents import create_deep_agent
from .models import ModelProvider
from langchain_core.messages import SystemMessage

# 1. 定义特化编码子 Agent 规范
coding_expert = {
    "name": "developer",
    "description": "用于执行复杂的本地代码编写、重构和测试任务",
    "system_prompt": "你是一个遵循 Claude Code 哲学的工程师。先观察，后编码，必须运行测试验证。",
    "skills": ["/skills/coding/python_best_practices.md"],
    "middleware": [
        # 为编码 Agent 绑定隔离的 Docker 后端
        FilesystemMiddleware(backend=DockerSafeBackend()) 
    ]
}

# 2. 定义多模态与搜索子 Agent
utility_worker = {
    "name": "utility",
    "description": "处理图片生成、语音识别、Web 检索和知识库查询",
    "tools": [web_search, image_gen, speech_to_text, rag_retrieval]
}

# 3. 创建主 Agent (Orchestrator)
orchestrator = create_deep_agent(
    model="anthropic:claude-3-5-sonnet-20240620",
    name="Deep-Orchestrator",
    system_prompt=SystemMessage(content="你是项目的中枢。利用黑板协调子 Agent。"),
    
    # 核心中间件
    memory=["/memory/PROJECT_CONTEXT.md"], # 长记忆
    subagents=[coding_expert, utility_worker], # 注册子 Agent
    
    # 热插拔旁路系统
    middleware=[
        HotSwapMiddleware(registry),  # 实时更新工具/技能
        DynamicToolMiddleware(retriever) # 渐进式披露工具
    ],
    
    # 人工干预
    interrupt_on={"edit_file": True, "execute": True},
    checkpointer=my_checkpointer, # 状态持久化
    store=my_blackboard_store      # 共享黑板存储
)

import os
from typing import List, Optional
from watchdog.observers import Observer

class DeepAgentBuilder:
    def __init__(self):
        self._model = ModelProvider.get_chat_model("ollama", "qwen3.5:9b")
        self._embd = ModelProvider.get_chat_model("ollama", "bge:")
        self._tools_dir = "./hot_tools"
        self._skills_dir = "./hot_skills"
        self._basic_tools = []
        self.system_prompt = ""
        self._km = None
        self._tenant = "system"
        self._hot_reload = True

    def set_model(self,provider, model_name: str):
        self._model = ModelProvider.get_chat_model(provider, model_name)
        return self
    
    def set_embd(self,provider, model_name: str):
        self._model = ModelProvider.get_embeddings(provider, model_name)
        return self

    def set_knowledge_manager(self, km):
        self._km = km
        return self
    
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        return self

    def with_hot_reload(self, tools_path: str = "./hot_tools", skills_path: str = "./hot_skills"):
        self._tools_dir = tools_path
        self._skills_dir = skills_path
        # 确保目录存在
        os.makedirs(tools_path, exist_ok=True)
        os.makedirs(skills_path, exist_ok=True)
        return self

    def add_basic_tools(self, tools: List):
        self._basic_tools.extend(tools)
        return self

    def build(self):
        """组装核心逻辑：Registry -> Manager -> Middleware -> Agent"""
        if not self._km:
            raise ValueError("KnowledgeManager is required for JIT Tool Injection.")

        # 1. 初始化向量管理器
        tm = ToolRegistryManager(self._km)

        # 2. 初始化能力注册表
        reg = CapabilityRegistry(
            tools_dir=self._tools_dir, 
            skills_dir=self._skills_dir, 
            manager=tm
        )

        # 3. 配置热重载监听 (守护线程)
        if self._hot_reload:
            observer = Observer()
            handler = ToolReloaderHandler(reg)
            observer.schedule(handler, path=self._tools_dir, recursive=False)
            observer.schedule(handler, path=self._skills_dir, recursive=False)
            observer.start()
            print(f"📡 Hot-reload active: Monitoring {self._tools_dir}")

        # 4. 组装 JIT 中间件
        jit_middleware = JITOrchestratorMiddleware(reg)

        # 5. 创建并返回 DeepAgent 实例
        return create_deep_agent(
            model=self._model,
            tools=self._basic_tools,
            middleware=[jit_middleware],
            system_prompt=self.system_prompt
            # 这里可以注入更多的全局配置
        )
    


if __name__ == '__main__':
    # 1. 准备底层的向量库封装
    my_km = KnowledgeManager(provider="qdrant", api_key="...")

    # 2. 使用 Builder 组装
    agent = (
        DeepAgentBuilder()
        .set_model("openai:gpt-4o")
        .set_knowledge_manager(my_km)
        .with_hot_reload(tools_path="./my_project/tools", skills_path="./my_project/skills")
        .add_basic_tools([web_search_tool, terminal_tool]) # 注入核心常驻工具
        .build()
    )

    # 3. 直接开始对话
    agent.invoke({"messages": [{"role": "user", "content": "用我的特化编码工具优化这段代码"}]})