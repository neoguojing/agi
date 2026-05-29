import contextlib

from langchain_core.runnables import RunnableConfig

from agi.agent.agent import build_agent_from_subagent
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
    promoted_stock_subagent = {
        **stock_analyse_subagent,
        "name": "stock-analyse-asubagent",
        "model": ModelProvider.get_falback_model(),
        "middleware": [*stock_analyse_subagent.get("middleware", []), notifier],
    }
    yield build_agent_from_subagent(promoted_stock_subagent)
