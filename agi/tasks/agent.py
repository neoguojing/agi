from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Checkpointer, interrupt

from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from agi.config import AGI_DEBUG
from agi.tasks.define import AskHuman, State
from agi.tasks.orchestration import build_main_agent, get_registered_tools


@dataclass(slots=True)
class DeepAgentOptions:
    """Options that map to deepagents `create_deep_agent` capabilities."""

    middleware: Sequence[AgentMiddleware] = field(default_factory=tuple)
    subagents: list[SubAgent | CompiledSubAgent] | None = None
    skills: list[str] | None = None
    memory: list[str] | None = None
    interrupt_on: dict[str, Any] | None = None
    backend: Any = None
    store: Any = None
    cache: Any = None
    context_schema: type[Any] | None = State
    include_external_tools: bool = True
    include_external_skills: bool = True


def _normalize_tools(tools: Union[Sequence[Union[BaseTool, Any]], ToolNode]) -> Sequence[Union[BaseTool, Any]]:
    if isinstance(tools, ToolNode):
        return list(tools.tools_by_name.values())
    return tools


def human_feedback_node(state: dict, config: RunnableConfig | None = None) -> dict:
    """Human interruption node retained for compatibility with the main graph."""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_call_id = last_message.tool_calls[0]["id"]
        ask = AskHuman.model_validate(last_message.tool_calls[0]["args"])
        feedback = interrupt(ask.question)
        return {"messages": [ToolMessage(tool_call_id=tool_call_id, content=feedback["messages"][-1].content)]}

    if isinstance(last_message, HumanMessage):
        feedback = interrupt("breaked")
        return {"messages": [AIMessage(content=feedback["messages"][-1].content)]}

    return state


async def ahuman_feedback_node(state: dict, config: RunnableConfig | None = None) -> dict:
    return human_feedback_node(state, config)


def create_react_agent(
    model: Any,
    tools: Union[Sequence[Union[BaseTool, Any]], ToolNode],
    *,
    checkpointer: Optional[Checkpointer] = None,
    response_format: Optional[Any] = None,
    debug: bool = False,
    name: Optional[str] = None,
    deepagent_options: Optional[DeepAgentOptions] = None,
    **_: Any,
) -> CompiledGraph:
    """Build main agent via deepagents with dynamic tools/skills registration."""
    options = deepagent_options or DeepAgentOptions()
    extra_tools = _normalize_tools(tools)

    merged_tools = get_registered_tools(
        include_builtin=True,
        include_external=options.include_external_tools,
        extra_tools=extra_tools,
    )

    return build_main_agent(
        model,
        tools=merged_tools,
        subagents=options.subagents,
        checkpointer=checkpointer,
        middleware=tuple(options.middleware),
        skills=options.skills,
        memory=options.memory,
        interrupt_on=options.interrupt_on if options.interrupt_on is not None else {"AskHuman": True},
        response_format=response_format,
        context_schema=options.context_schema,
        backend=options.backend,
        store=options.store,
        cache=options.cache,
        debug=debug,
        name=name,
        include_builtin_tools=False,
        include_external_tools=False,
        include_builtin_skills=True,
        include_external_skills=options.include_external_skills,
    )

