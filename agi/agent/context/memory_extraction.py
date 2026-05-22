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
MEMORY_EXTRACTION_INSTRUCTIONS = """
Extract memory candidates from the conversation and return ONLY JSON matching the provided schema.

Requirements:
- Consolidate, normalize, and merge semantically equivalent information.
- NEVER output duplicated, overlapping, or redundant memory entries.
- If multiple messages describe the same fact, preference, project, or behavior, combine them into a single concise memory.
- Prefer generalized and stable summaries over fragmented or repetitive details.
- Remove minor wording differences and temporal duplicates.
- Keep memories atomic, non-overlapping, and maximally deduplicated.
- Do not repeat the same information across different memory items.
- The final output must contain zero redundant entries.
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

