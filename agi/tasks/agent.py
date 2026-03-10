from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence, Union

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Checkpointer, interrupt

from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent

from agi.config import AGI_DEBUG
from agi.tasks.define import AskHuman, State


agent_prompt = """
You are an AI orchestrator agent.
- Delegate specialized multimodal work to subagents whenever suitable.
- Prefer using subagents for image generation/recognition, knowledge retrieval/search, and speech tasks.
- For ambiguous goals, ask for clarification using AskHuman.
- Date: {date}
- Respond only in {language}.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_prompt),
        ("placeholder", "{messages}"),
    ]
)


@dataclass(slots=True)
class DeepAgentOptions:
    """Options that directly map to deepagents `create_deep_agent` capabilities."""

    middleware: Sequence[AgentMiddleware] = field(default_factory=tuple)
    subagents: list[SubAgent | CompiledSubAgent] | None = None
    skills: list[str] | None = None
    memory: list[str] | None = None
    interrupt_on: dict[str, Any] | None = None
    backend: Any = None
    store: Any = None
    cache: Any = None
    context_schema: type[Any] | None = State


def _normalize_tools(tools: Union[Sequence[Union[BaseTool, Any]], ToolNode]) -> Sequence[Union[BaseTool, Any]]:
    if isinstance(tools, ToolNode):
        return list(tools.tools_by_name.values())
    return tools


def _build_system_prompt(language: str = "chinese") -> str:
    return agent_prompt.format(date=datetime.now(), language=language)


@tool(return_direct=True)
async def rag_search(query: str) -> Any:
    """Knowledge-base retrieval and synthesis for questions requiring indexed context."""
    from agi.tasks.rag_web import rag_as_subgraph

    state = State(messages=[HumanMessage(content=query)], user_id="subagent_rag", feature="rag")
    config = {"configurable": {"conversation_id": "subagent_rag", "thread_id": "subagent_rag"}}
    return await rag_as_subgraph.ainvoke(state, config=config)


@tool(return_direct=True)
async def web_search(query: str) -> Any:
    """Web search and synthesis when local KB is insufficient."""
    from agi.tasks.rag_web import rag_as_subgraph

    state = State(messages=[HumanMessage(content=query)], user_id="subagent_web", feature="web")
    config = {"configurable": {"conversation_id": "subagent_web", "thread_id": "subagent_web"}}
    return await rag_as_subgraph.ainvoke(state, config=config)


@tool(return_direct=True)
async def speech_to_text(audio: str) -> Any:
    """Convert audio input to text."""
    from agi.tasks.task_factory import TASK_SPEECH_TEXT, TaskFactory

    task = TaskFactory.create_task(TASK_SPEECH_TEXT)
    state = State(messages=[HumanMessage(content=audio)], user_id="subagent_speech", feature="speech")
    return await task.ainvoke(state)


@tool(return_direct=True)
async def text_to_speech(text: str) -> Any:
    """Convert text input to speech."""
    from agi.tasks.task_factory import TASK_TTS, TaskFactory

    task = TaskFactory.create_task(TASK_TTS)
    state = State(messages=[HumanMessage(content=text)], user_id="subagent_tts", feature="tts")
    return await task.ainvoke(state)


def _image_tools() -> list[Any]:
    from agi.tasks.task_factory import TASK_IMAGE_GEN, TASK_MULTI_MODEL, TaskFactory

    image_gen_tool = TaskFactory.create_task(TASK_IMAGE_GEN).as_tool(
        name="image_gen",
        description="Generate images from text description.",
    )
    image_gen_tool.return_direct = True

    image_recog_tool = TaskFactory.create_task(TASK_MULTI_MODEL).as_tool(
        name="image_recog",
        description="Recognize and describe image content.",
    )
    image_recog_tool.return_direct = True

    return [image_gen_tool, image_recog_tool]


def _build_default_subagents(model: Any) -> list[SubAgent]:
    image_tools = _image_tools()
    return [
        {
            "name": "image-specialist",
            "description": "Use this for image generation, image understanding, and multimodal image analysis.",
            "system_prompt": "You are an image specialist subagent. Prefer image tools and return concise actionable outputs.",
            "model": model,
            "tools": image_tools,
        },
        {
            "name": "knowledge-specialist",
            "description": "Use this for RAG retrieval, web search, and evidence-grounded factual answers.",
            "system_prompt": "You are a retrieval specialist. Use KB/web tools first, then synthesize with citations when possible.",
            "model": model,
            "tools": [rag_search, web_search],
        },
        {
            "name": "speech-specialist",
            "description": "Use this for speech-to-text and text-to-speech conversion workflows.",
            "system_prompt": "You are a speech specialist. Handle transcription and synthesis efficiently.",
            "model": model,
            "tools": [speech_to_text, text_to_speech],
        },
    ]


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
    """Build main agent via deepagents, with multimodal tasks delegated to subagents."""
    options = deepagent_options or DeepAgentOptions()
    base_tools = _normalize_tools(tools)

    subagents = options.subagents or _build_default_subagents(model)
    interrupt_on = options.interrupt_on if options.interrupt_on is not None else {"AskHuman": True}

    return create_deep_agent(
        model=model,
        tools=base_tools,
        system_prompt=_build_system_prompt(),
        middleware=options.middleware,
        subagents=subagents,
        skills=options.skills,
        memory=options.memory,
        response_format=response_format,
        context_schema=options.context_schema,
        checkpointer=checkpointer,
        store=options.store,
        backend=options.backend,
        interrupt_on=interrupt_on,
        debug=debug,
        name=name,
        cache=options.cache,
    )


memory = MemorySaver()


def _build_agent_executor(llm, *, checkpointer: Optional[Checkpointer] = None):
    from agi.tasks.tools import tools

    return create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        debug=AGI_DEBUG,
        name="agent",
    )


def create_react_agent_task(llm):
    return _build_agent_executor(llm, checkpointer=memory)


def create_react_agent_as_subgraph(llm):
    return _build_agent_executor(llm)
