# 导入具体的工具实现
from .debug_middleware import DebugLLMContextMiddleware
from .context_middleware import ContextEngineeringMiddleware
from .browser_middleware import BrowserMiddleware
from .ffmpeg_middleware import FfmpegMiddleware
from .common_middleware import MultimodalBase64Middleware
from .memory_middleware import MemoryMiddleware
from langchain.chat_models import BaseChatModel
from deepagents.backends.protocol import BACKEND_TYPES
from deepagents.middleware.summarization import SummarizationMiddleware

def create_summarization_middleware(
    model: BaseChatModel,
    backend: BACKEND_TYPES,
) -> SummarizationMiddleware:
    """Create a `SummarizationMiddleware` with model-aware defaults.

    Computes trigger, keep, and truncation settings from the model's profile
    (or uses fixed-token fallbacks) and returns a configured middleware.

    Args:
        model: Resolved chat model instance.
        backend: Backend instance or factory for persisting conversation history.

    Returns:
        Configured `SummarizationMiddleware` instance.
    """
    from langchain.chat_models import BaseChatModel as RuntimeBaseChatModel  # noqa: PLC0415

    if not isinstance(model, RuntimeBaseChatModel):
        msg = "`create_summarization_middleware` expects `model` to be a `BaseChatModel` instance."
        raise TypeError(msg)

    return SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("messages", 20),
        keep=("messages", 10),
        trim_tokens_to_summarize=None,
        truncate_args_settings={
            "trigger": ("tokens", 30000),
            "keep": ("messages", 10),
            "max_length": 2000,
            "truncation_text": "...(truncated)",
        }
    )


# 导出清单，方便其他模块调用
__all__ = [
           "create_summarization_middleware",
           "DebugLLMContextMiddleware",
           "ContextEngineeringMiddleware",
           "BrowserMiddleware",
           "FfmpegMiddleware",
           "MultimodalBase64Middleware",
           "MemoryMiddleware"
           ]