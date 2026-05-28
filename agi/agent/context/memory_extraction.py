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

from agi.agent.context.memory_models import MemoryTarget




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

## Profile Memory
- Extract ONLY stable long-term information (preferences, skills, roles, settings)
- Use concise snake_case keys: favorite_language, job_title, preferred_editor
- Each memory must contain ONLY ONE fact — do not merge unrelated attributes
- Avoid temporary conversational details

## Episodic Memory
- Extract ONLY concrete events, milestones, decisions, or meaningful interactions
- Each memory should represent ONE atomic event with a concise factual summary
- Avoid vague summaries like "user talked about something"
- Use ISO-8601 datetime format for event_time

## Semantic Memory
- Represent knowledge as subject-predicate-object triples
- subject must be a short, canonical entity name (e.g., "Python", "OpenAI")
- predicate must be a normalized relation: works_at, likes, uses, knows, created
- object must be a concrete value or entity
- Each memory must contain ONLY ONE fact
"""

# Per-target extraction hints appended to the prompt for focused guidance.
_MEMORY_EXTRACTION_HINTS = {
    "profile": (
        "Focus: extract stable profile attributes as key-value pairs. "
        "Examples: favorite_language=Python, job_title=Software Engineer."
    ),
    "episodic": (
        "Focus: extract concrete events, milestones, and meaningful interactions. "
        "Examples: user started a new job, user completed a migration."
    ),
    "semantic": (
        "Focus: extract factual knowledge as subject-predicate-object triples. "
        "Examples: (Python, is_a, programming_language), (user, works_at, company)."
    ),
}

def build_memory_extraction_prompt(
    *,
    conversation: str,
    existing_memory: str = "",
    target: MemoryTarget | None = None,
) -> str:
    """Build an LLM-facing prompt for structured memory extraction.

    Combines the system instructions, the JSON schema, the current state of
    memory (to avoid duplicates), and the conversation history.
    """
    hint = _MEMORY_EXTRACTION_HINTS.get(target) if target else ""
    hint_section = f"\n{hint}\n\n" if hint else ""
    return (
        f"{MEMORY_EXTRACTION_INSTRUCTIONS}\n\n"
        f"EXISTING_MEMORY:\n{existing_memory or '(none)'}\n\n"
        f"CONVERSATION:\n{conversation}\n"
        f"{hint_section}"
    )

