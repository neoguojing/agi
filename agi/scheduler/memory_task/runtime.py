import logging
from typing import cast
from langgraph.store.base import BaseStore
from langgraph.graph.state import CompiledStateGraph
from langgraph_sdk import get_client

# 🔄 核心对齐：引入标准上下文容器与基类
from agi.scheduler.task_hub import TaskContext, BaseRuntime, hub, ExternalStateBridge, runtime_state_bridge
from agi.agent.models import ModelProvider
from agi.config import LANGGRAPH_MAIN_URL, CLOUD_MODE

logger = logging.getLogger("TaskRuntime")

class MemoryTaskRuntime(BaseRuntime):
    """
    通用的重型环境依赖容器。
    静态依赖（LLM、Client）通过构造函数直接注入；
    动态依赖（Graph、Thread_id、User_id）通过 @property 从桥接器实时拉取。
    """
    def __init__(self, state_bridge: ExternalStateBridge):
        super().__init__()
        self.llm = ModelProvider.get_falback_model()              # 🤖 静态单例依赖
        self.client = None
        if CLOUD_MODE:
            self.client = get_client(url=LANGGRAPH_MAIN_URL)        # 🔌 静态单例依赖
        self._bridge = state_bridge # 🌁 动态中转桥接器

    @property
    def graph(self) -> CompiledStateGraph:
        """动态感知外部线程传入的图实例"""
        return self._bridge.get_value("graph")

    @property
    def store(self) -> BaseStore:
        """动态感知外部线程传入的图实例"""
        return self._bridge.get_value("store")

    @property
    def thread_id(self) -> str:
        """动态感知外部线程传入的 Thread ID"""
        return self._bridge.get_value("thread_id")

    @property
    def user_id(self) -> str:
        """动态感知外部线程传入的 User ID"""
        return self._bridge.get_value("user_id")
    
memory_runtime = MemoryTaskRuntime(
    state_bridge=runtime_state_bridge
)
