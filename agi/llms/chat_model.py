from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def create_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain chat model with standard model parameters.

    This follows the LangChain docs' recommended initialization style via
    `init_chat_model`, while remaining provider-agnostic.
    """
    init_kwargs: dict[str, Any] = {
        "model": model,
        **kwargs,
    }

    if model_provider:
        init_kwargs["model_provider"] = model_provider
    if temperature is not None:
        init_kwargs["temperature"] = temperature
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        init_kwargs["timeout"] = timeout
    if max_retries is not None:
        init_kwargs["max_retries"] = max_retries

    return init_chat_model(**init_kwargs)
