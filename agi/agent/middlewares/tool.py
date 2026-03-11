from langchain.agents.middleware.types import AgentMiddleware
from typing import Any

import os
import importlib.util
from langchain_core.tools import BaseTool

class CapabilityRegistry:
    def __init__(self, tools_dir: str, skills_dir: str):
        self.tools_dir = tools_dir
        self.skills_dir = skills_dir
        self.active_tools: dict[str, BaseTool] = {}
        self.active_skills: str = ""
        self.reload()

    def reload(self):
        """扫描目录并重新加载所有工具和技能"""
        # 1. 加载工具 (.py 文件)
        new_tools = {}
        for f in os.listdir(self.tools_dir):
            if f.endswith(".py"):
                path = os.path.join(self.tools_dir, f)
                spec = importlib.util.spec_from_file_location(f, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # 假设每个文件内定义了一个名为 'tool' 的 BaseTool 对象
                if hasattr(module, "tool"):
                    new_tools[module.tool.name] = module.tool
        self.active_tools = new_tools

        # 2. 加载技能 (.md 文件)
        skills_text = []
        for f in os.listdir(self.skills_dir):
            if f.endswith(".md"):
                with open(os.path.join(self.skills_dir, f), "r") as file:
                    skills_text.append(file.read())
        self.active_skills = "\n\n".join(skills_text)
        print(f"🔄 Registry Reloaded: {len(self.active_tools)} tools, {len(skills_text)} skills.")

class DynamicToolMiddleware(AgentMiddleware):
    def __init__(self, retriever: ToolRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    async def __call__(self, state: dict[str, Any], call_next):
        # 获取最后一条用户消息的内容作为检索 query
        last_message = state["messages"][-1].content
        
        # 1. 动态检索工具
        selected_tools = self.retriever.retrieve(last_message, k=self.top_k)
        
        # 2. 注入到状态中 (假设 create_deep_agent 的底层逻辑支持从 state 读取 tools)
        # 在 LangGraph 中，我们通常修改 RunnableConfig 或传递给 LLM 的 bind_tools
        state["dynamic_tools"] = selected_tools
        
        # 3. 继续执行后续中间件
        return await call_next(state)
    

class HotSwapMiddleware(AgentMiddleware):
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    async def __call__(self, state, call_next):
        # 1. 注入动态工具
        # 注意：源码中的 create_agent 会根据 state["tools"] 重新绑定模型
        dynamic_tools = list(self.registry.active_tools.values())
        if "tools" not in state:
            state["tools"] = []
        state["tools"].extend(dynamic_tools)

        # 2. 注入动态技能到 System Prompt
        if self.registry.active_skills:
            skill_message = f"\n\n## Current Dynamic Skills:\n{self.registry.active_skills}"
            # 找到 SystemMessage 并追加内容
            for msg in state["messages"]:
                if msg.type == "system":
                    msg.content += skill_message
                    break
        
        return await call_next(state)

from typing import Callable
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class JITToolInjectionMiddleware(AgentMiddleware):
    def __init__(self, library: DynamicToolLibrary):
        super().__init__()
        self.library = library

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        核心逻辑：在模型调用前，动态注入检索到的工具
        """
        # 1. 获取当前意图（最后几条消息）
        context_query = request.messages[-1].content
        
        # 2. 语义检索：从库中找出匹配的工具
        # 这里实现了“按需发现”，Agent 甚至不知道库里有这些，但 Middleware 帮它找到了
        dynamic_tools = self.library.retrieve_relevant_tools(context_query)
        
        # 3. 混合工具集：保留预注册的基础工具（如任务管理）+ 动态注入的特化工具
        # 注意：这里我们覆盖了 request.tools，模型会认为这些是它唯一可选的工具
        new_tools = list(request.tools or []) + dynamic_tools
        
        # 4. 这里的 override 会触发模型端的 bind_tools
        updated_request = request.override(tools=new_tools)
        
        return handler(updated_request)
    
 @before_model
def discovery_hint_middleware(state: AgentState, runtime: Runtime):
    """
    Node-style 钩子：告诉 Agent 它拥有“动态发现”的能力
    """
    hint = (
        "\n[System Note]: You have access to a vast library of hidden tools. "
        "Just describe what you need to do, and the system will automatically "
        "inject the relevant specialized tools into your next turn."
    )
    # 修改系统消息，增强 Agent 的主动性
    current_sys = state["messages"][0].content
    return {"messages": [SystemMessage(content=current_sys + hint)]}   

# 1. 启动注册表
registry = CapabilityRegistry(tools_dir="./hot_tools", skills_dir="./hot_skills")

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            registry.reload()

observer = Observer()
observer.schedule(ReloadHandler(), path="./hot_tools", recursive=False)
observer.schedule(ReloadHandler(), path="./hot_skills", recursive=False)
observer.start()