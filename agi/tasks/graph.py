from __future__ import annotations

from typing import Any, Iterator, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agi.config import log
from agi.tasks.define import Feature, State
from agi.tasks.runtime.task_factory import  TaskFactory
from agi.tasks.session import SessionIdentity, to_configurable
from agi.tasks.utils import graph_print
from agi.tasks.agent import create_react_agent


class AgiGraph:
    """
    Thin runtime adapter over DeepAgents.

    Complex routing/planning is delegated to deepagents middleware stack
    (TodoListMiddleware, SubAgentMiddleware, Skills/Memory middleware, etc.)
    instead of maintaining a large custom state graph here.
    """

    def __init__(self):
        self.agent = create_react_agent(TaskFactory.get_llm(),[])

    @staticmethod
    def _build_config(state: State) -> dict[str, Any]:
        identity = SessionIdentity.from_state(state)
        return {"configurable": to_configurable(identity, need_speech=state.get("need_speech", False))}


    async def invoke(self, input: State) -> State:
        config = self._build_config(input)

        output = await self.agent.ainvoke(input, config=config)

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
