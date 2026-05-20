"""LLM-facing memory extraction schema, prompt, and parser.

Usage:
    1. Build a prompt with `build_memory_extraction_prompt(...)`.
    2. Send that prompt to an LLM using JSON/structured output mode.
    3. Parse the LLM JSON with `parse_memory_extraction_result(...)`.
    4. Convert the typed result to patches via `result.to_patches()`.

This module deliberately does not write memory. It only shapes LLM output so
maintenance tasks can review and persist patches through `MemoryStore`.
"""

from __future__ import annotations

import logging




logger = logging.getLogger(__name__)

# System instructions provided to the LLM to guide the extraction process.
MEMORY_EXTRACTION_INSTRUCTIONS = """Extract memory candidates from the conversation and return ONLY JSON matching the provided schema.
Rules:
- Use profile_memories only for stable user identity, preferences, durable settings, and long-lived facts.
- Use episodic_memories for time-bound events, lessons, task outcomes, or experiences that can decay.
- Use semantic_memories for long-term abstract knowledge as graph-ready subject/predicate/object records.
- Reject transient, low-value, or ambiguous candidates in rejected_candidates instead of forcing a memory.
- Do not write files directly; the caller converts this structured result into MemoryPatch objects.
"""

def build_memory_extraction_prompt(*, conversation: str, existing_memory: str = "") -> str:
    """Build an LLM-facing prompt for structured memory extraction.

    Combines the system instructions, the JSON schema, the current state of
    memory (to avoid duplicates), and the conversation history.
    """
    return (
        f"{MEMORY_EXTRACTION_INSTRUCTIONS}\n\n"
        f"EXISTING_MEMORY:\n{existing_memory or '(none)'}\n\n"
        f"CONVERSATION:\n{conversation}\n"
    )

