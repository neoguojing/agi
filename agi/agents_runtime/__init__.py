"""新的 Agent Runtime 框架（基于 docs 设计目标 + DeepAgents）。"""

from .integration.backend_factory import BackendBuildResult, BackendFactoryError, PolicyWrapper, build_backend
from .engines.context_engine import get_context_engine
from .orchestration.harness import TodoItem, TodoManager
from .orchestration.hitl import InterruptAction, build_resume_payload, extract_interrupt_actions
from .engines.memory_engine import MemorySearchManager, create_default_memory_manager
from .engines.multimodal import Modality, MultiModalRequest, MultiModalRouter
from .orchestration.session_context import SessionContextManager, SessionState
from .core.skills import Skill, SkillRegistry
from .core.subagents import SubAgentSpec, build_default_subagents
from .core.tools import ToolRegistry, ToolSpec
from .core.types import AgentRuntimeConfig

__all__ = [
    "AgentRuntimeConfig",
    "MemorySearchManager",
    "create_default_memory_manager",
    "get_context_engine",
    "Modality",
    "MultiModalRequest",
    "MultiModalRouter",
    "ToolSpec",
    "ToolRegistry",
    "Skill",
    "SkillRegistry",
    "SubAgentSpec",
    "build_default_subagents",
    "build_backend",
    "PolicyWrapper",
    "BackendBuildResult",
    "BackendFactoryError",
    "SessionContextManager",
    "SessionState",
    "TodoManager",
    "TodoItem",
    "InterruptAction",
    "extract_interrupt_actions",
    "build_resume_payload",
]
