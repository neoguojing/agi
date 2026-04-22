from typing import Any, Dict
from langgraph.checkpoint.base import BaseCheckpointSaver
from collections.abc import Mapping

from langgraph.checkpoint.base import Checkpoint
from langgraph.channels.base import BaseChannel
from langgraph.managed.base import ManagedValueMapping, ManagedValueSpec

def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Get channels from a checkpoint."""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}

    if checkpoint is None:
        return (channel_specs,managed_specs)
    
    
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v
    return (
        {
            k: v.from_checkpoint(checkpoint["channel_values"].get(k, None))
            for k, v in channel_specs.items()
        },
        managed_specs,
    )

def checkpoint_to_state(
    checkpoint,
    channel_specs: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert LangGraph checkpoint -> external agent state.

    Args:
        checkpoint: checkpoint dict from checkpointer.get_tuple().checkpoint
        channel_specs: graph channel definitions (self.channels)

    Returns:
        state dict compatible with LangChain / LangGraph agent input
    """
    # 1. Rebuild channel objects from checkpoint
    channels, managed = channels_from_checkpoint(
        channel_specs,
        checkpoint
    )
    # 2. Extract messages channel (core logic)
    messages = None
    _summarization_event = None
    if "messages" in channels:
        ch = channels["messages"]

        # safe access
        if hasattr(ch, "get"):
            try:
                messages = ch.get()
            except Exception:
                messages = []

    if "_summarization_event" in channels:
        ch = channels["_summarization_event"]

        # safe access
        if hasattr(ch, "get"):
            try:
                _summarization_event = ch.get()
            except Exception:
                _summarization_event = None

    # 3. Build state object
    state = {
        "messages": messages or [],
        "_summarization_event": _summarization_event,
    }

    return state