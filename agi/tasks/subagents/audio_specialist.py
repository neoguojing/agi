from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agi.tasks.define import State
from agi.tasks.runtime.task_factory import TASK_SPEECH_TEXT, TASK_TTS, TaskFactory


@tool(return_direct=True)
async def speech_to_text(audio: str) -> Any:
    """Convert audio input to text."""
    task = TaskFactory.create_task(TASK_SPEECH_TEXT)
    state = State(messages=[HumanMessage(content=audio)], user_id="subagent_speech", feature="speech")
    return await task.ainvoke(state)


@tool(return_direct=True)
async def text_to_speech(text: str) -> Any:
    """Convert text input to speech."""
    task = TaskFactory.create_task(TASK_TTS)
    state = State(messages=[HumanMessage(content=text)], user_id="subagent_tts", feature="tts")
    return await task.ainvoke(state)


audio_subagent = {
    "name": "audio_specialist",
    "description": "Use this for transcription, synthesis, and voice chat workflows.",
    "system_prompt": "You are an audio specialist. Handle ASR/TTS workflows efficiently.",
    "tools": [speech_to_text, text_to_speech],
}
