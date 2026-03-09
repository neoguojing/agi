"""新的 Agent Runtime 框架（基于 docs 设计目标 + DeepAgents）。"""

from .backend_factory import BackendBuildResult, BackendFactoryError, PolicyWrapper, build_backend
from .context_engine import get_context_engine
from .memory_engine import MemorySearchManager, create_default_memory_manager
from .multimodal import Modality, MultiModalRequest, MultiModalRouter
from .session_context import SessionContextManager, SessionState
from .skills import Skill, SkillRegistry
from .subagents import SubAgentSpec, build_default_subagents
from .tools import ToolRegistry, ToolSpec
from .types import AgentRuntimeConfig

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
]
