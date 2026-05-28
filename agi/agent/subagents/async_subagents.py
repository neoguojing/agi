import contextlib
from langchain_core.runnables import RunnableConfig
from agi.agent.middlewares.completion_notifier_middleware import build_completion_notifier
from agi.agent.deep_agent import create_deep_agent
from agi.agent.middlewares import BrowserMiddleware,FfmpegMiddleware,StockMiddleware,DebugLLMContextMiddleware
from agi.agent.subagents.general import make_backend
from agi.agent.models import ModelProvider

stock_middleware = StockMiddleware(backend=make_backend)

@contextlib.asynccontextmanager
async def stock_graph(config: RunnableConfig):
    """Graph factory that wires up the completion notifier from config."""
    configurable = config.get("configurable", {})
    notifier = build_completion_notifier(
        parent_thread_id=configurable.get("parent_thread_id"),
        parent_assistant_id=configurable.get("parent_assistant_id"),
        subagent_name="stock-analyse-asubagent",
    )
    yield create_deep_agent(
        model=ModelProvider.get_falback_model(),
        tools=[],
        system_prompt="",
        middleware=[stock_middleware,notifier],
        name="stock-analyse-asubagent",
    )
