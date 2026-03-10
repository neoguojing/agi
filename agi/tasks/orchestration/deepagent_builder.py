from __future__ import annotations

from typing import Any, Sequence

from agi.deepagents.graph import create_deep_agent
from agi.tasks.orchestration.registry import get_registered_skills, get_registered_tools
from agi.tasks.session import resolve_session_components
from agi.tasks.subagents.audio_specialist import audio_subagent
from agi.tasks.subagents.image_specialist import image_subagent
from agi.tasks.subagents.rag_specialist import rag_subagent
from agi.tasks.subagents.web_research_specialist import web_subagent


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
    include_builtin_tools: bool = True,
    include_external_tools: bool = True,
    include_builtin_skills: bool = True,
    include_external_skills: bool = True,
    enable_long_term_memory: bool | None = None,
):
    resolved_tools = (
        get_registered_tools(
            include_builtin=include_builtin_tools,
            include_external=include_external_tools,
            extra_tools=tools,
        )
        if include_builtin_tools or include_external_tools
        else list(tools or [])
    )

    resolved_skills = get_registered_skills(
        include_builtin=include_builtin_skills,
        include_external=include_external_skills,
        extra_skills=skills,
    )

    resolved_backend, resolved_store, resolved_checkpointer = resolve_session_components(
        backend=backend,
        store=store,
        checkpointer=checkpointer,
        enable_long_term_memory=enable_long_term_memory,
    )

    return create_deep_agent(
        model=model,
        tools=resolved_tools,
        middleware=middleware,
        subagents=subagents or [rag_subagent, web_subagent, image_subagent, audio_subagent],
        skills=resolved_skills,
        memory=memory,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=resolved_checkpointer,
        store=resolved_store,
        backend=resolved_backend,
        interrupt_on=interrupt_on,
        debug=debug,
        name=name,
        cache=cache,
    )
