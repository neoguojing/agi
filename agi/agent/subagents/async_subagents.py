import contextlib

from langchain_core.runnables import RunnableConfig

from agi.agent.deep_agent import create_deep_agent
from agi.agent.middlewares.completion_notifier_middleware import build_completion_notifier
from agi.agent.models import ModelProvider
from agi.agent.subagents.general import stock_analyse_subagent


@contextlib.asynccontextmanager
async def stock_graph(config: RunnableConfig):
    """Promote the stock subagent to a standalone LangGraph assistant."""

    configurable = (config or {}).get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name="stock-analyse-asubagent",
    )
    yield create_deep_agent(
        model=ModelProvider.get_falback_model(),
        tools=stock_analyse_subagent.get("tools", []),
        system_prompt=stock_analyse_subagent.get("system_prompt", ""),
        middleware=[*stock_analyse_subagent.get("middleware", []), notifier],
        name="stock-analyse-asubagent",
    )
