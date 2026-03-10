from __future__ import annotations

from typing import Any, Sequence

from agi.deepagents.graph import create_deep_agent
from agi.tasks.subagents.audio_specialist import audio_subagent
from agi.tasks.subagents.image_specialist import image_subagent
from agi.tasks.subagents.rag_specialist import rag_subagent
from agi.tasks.subagents.web_research_specialist import web_subagent
from agi.tasks.simple_tools import simple_tools


def build_main_agent(
    model: Any,
    *,
    tools: Sequence[Any] | None = None,
    subagents: list[dict[str, Any]] | None = None,
    checkpointer: Any = None,
    middleware: tuple[Any, ...] = (),
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    response_format: Any = None,
    context_schema: type[Any] | None = None,
    backend: Any = None,
    store: Any = None,
    cache: Any = None,
    debug: bool = False,
    name: str | None = None,
):
    return create_deep_agent(
        model=model,
        tools=list(tools) if tools is not None else list(simple_tools),
        middleware=middleware,
        subagents=subagents or [rag_subagent, web_subagent, image_subagent, audio_subagent],
        skills=skills,
        memory=memory,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        backend=backend,
        interrupt_on=interrupt_on,
        debug=debug,
        name=name,
        cache=cache,
    )
