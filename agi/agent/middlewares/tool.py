import os
import importlib.util
import threading
from typing import Dict, List, Any, Callable
from langchain_core.tools import BaseTool
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, before_agent
from langchain.messages import SystemMessage

from agi.rag.tool_retreiver import ToolRegistryManager 
from watchdog.events import FileSystemEventHandler

class CapabilityRegistry:
    def __init__(self, tools_dir: str, skills_dir: str, manager: ToolRegistryManager):
        self.tools_dir = tools_dir
        self.skills_dir = skills_dir
        self.manager = manager
        self.active_skills: str = ""
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        """线程安全的物理加载与向量同步"""
        with self._lock:
            new_tools: List[BaseTool] = []
            # 1. 动态加载 .py 工具
            for f in os.listdir(self.tools_dir):
                if f.endswith(".py"):
                    try:
                        path = os.path.join(self.tools_dir, f)
                        spec = importlib.util.spec_from_file_location(f[:-3], path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if hasattr(module, "tool"):
                            new_tools.append(module.tool)
                    except Exception as e:
                        print(f"❌ Failed to load tool {f}: {e}")

            # 2. 同步至向量管理器 (异步任务通常建议放入事件循环，这里简化处理)
            import asyncio
            asyncio.run(self.manager.register_tools(new_tools))

            # 3. 加载技能 .md
            skills_text = []
            for f in os.listdir(self.skills_dir):
                if f.endswith(".md"):
                    with open(os.path.join(self.skills_dir, f), "r") as file:
                        skills_text.append(file.read())
            self.active_skills = "\n\n".join(skills_text)
            print(f"🔄 Registry Synced: {len(new_tools)} tools vectorized.")


class JITOrchestratorMiddleware(AgentMiddleware):
    def __init__(self, registry: CapabilityRegistry):
        super().__init__()
        self.registry = registry
        self.manager = registry.manager

    @before_agent
    def inject_skills_and_hints(self, state, runtime):
        """Node-style: 注入动态技能和发现指南"""
        updates = []
        # 1. 注入技能
        if self.registry.active_skills:
            updates.append(f"\n\n### Specialized Skills:\n{self.registry.active_skills}")
        
        # 2. 注入发现 Hint
        updates.append("\n[System]: You have dynamic tool access. Describe your need to trigger specialized tools.")

        # 找到系统消息并追加
        messages = list(state["messages"])
        for i, m in enumerate(messages):
            if m.type == "system":
                messages[i] = SystemMessage(content=m.content + "".join(updates))
                break
        return {"messages": messages}

    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        """Wrap-style: 实现语义级工具热插拔"""
        # 1. 提取意图 (最后一条用户消息)
        user_query = request.messages[-1].content
        
        # 2. 语义检索 (同步包装异步)
        import asyncio
        dynamic_tools = asyncio.run(self.manager.retrieve_and_restore(user_query))
        
        # 3. 混合并去重
        existing_names = {t.name for t in (request.tools or [])}
        new_tools = list(request.tools or []) + [t for t in dynamic_tools if t.name not in existing_names]
        
        # 4. 覆盖请求
        return handler(request.override(tools=new_tools))
    
class ToolReloaderHandler(FileSystemEventHandler):
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def on_modified(self, event):
        if not event.is_directory and (event.src_path.endswith(".py") or event.src_path.endswith(".md")):
            print(f"✨ File {event.src_path} changed, reloading registry...")
            self.registry.reload()