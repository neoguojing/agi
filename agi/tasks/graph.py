from __future__ import annotations

from typing import Any, Iterator, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agi.config import log
from agi.tasks.define import Feature, State
from agi.tasks.runtime.task_factory import TASK_DEEPAGENT, TASK_SPEECH_TEXT, TASK_TTS, TaskFactory
from agi.tasks.session import SessionIdentity, to_configurable
from agi.tasks.utils import graph_print


class AgiGraph:
    """
    Thin runtime adapter over DeepAgents.

    Complex routing/planning is delegated to deepagents middleware stack
    (TodoListMiddleware, SubAgentMiddleware, Skills/Memory middleware, etc.)
    instead of maintaining a large custom state graph here.
    """

    def __init__(self):
        self.agent = TaskFactory.create_task(TASK_DEEPAGENT)
        self.speech2text = TaskFactory.create_task(TASK_SPEECH_TEXT)
        self.tts = TaskFactory.create_task(TASK_TTS)

    @staticmethod
    def _build_config(state: State) -> dict[str, Any]:
        identity = SessionIdentity.from_state(state)
        return {"configurable": to_configurable(identity, need_speech=state.get("need_speech", False))}

    async def _run_tts_if_needed(self, state: State, config: dict[str, Any]) -> State:
        if not state.get("need_speech"):
            return state

        if not state.get("messages"):
            return state

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            return state

        tts_input: State = {
            **state,
            "feature": Feature.TTS,
            "messages": [HumanMessage(content=str(last_msg.content))],
        }
        tts_output = await self.tts.ainvoke(tts_input, config=config)
        if isinstance(tts_output, dict):
            return {**state, **tts_output}
        return state

    async def invoke(self, input: State) -> State:
        config = self._build_config(input)

        # keep explicit speech-only path for backward compatibility
        if input.get("feature") == Feature.SPEECH:
            return await self.speech2text.ainvoke(input, config=config)

        output = await self.agent.ainvoke(input, config=config)
        if not isinstance(output, dict):
            return input

        output = await self._run_tts_if_needed(output, config)
        return output

    async def stream(
        self,
        input: State,
        stream_mode: list[str] | str = ["messages", "updates", "custom"],
    ) -> Iterator[Union[BaseMessage, tuple[str, Any], dict[str, Any]]]:
        config = self._build_config(input)

        mode = stream_mode
        if isinstance(stream_mode, list) and len(stream_mode) == 1:
            mode = stream_mode[0]

        try:
            async for event in self.agent.astream(input, config=config, stream_mode=mode):
                if isinstance(event, tuple) and event[0] == "messages":
                    msg = event[1][0]
                    if isinstance(msg, HumanMessage):
                        continue
                yield event
        except Exception as e:
            log.error(f"Error during streaming: {e}")
            yield {"error": str(e)}

    def display(self):
        graph_print(self.agent)
