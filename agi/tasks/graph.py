import io
from PIL import Image as PILImage
from langgraph.graph import END, StateGraph, START
from agi.tasks import tools,TaskFactory,TASK_IMAGE_GEN,TASK_SPEECH
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import  RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.globals import set_debug
from langchain.globals import set_verbose
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from agi.tasks.task_factory import TaskFactory,TASK_AGENT,TASK_IMAGE_GEN,TASK_LLM_WITH_RAG,TASK_SPEECH_TEXT,TASK_TTS

set_verbose(True)
set_debug(True)

class State(TypedDict):
    # Append-only chat memory so the agent can try to recover from initial mistakes.
    messages: Annotated[list[AnyMessage], add_messages]
    input_type: str
    need_speech: bool
    status: str
    
class AgentGraph:
    def __init__(self):
        
        checkpointer = MemorySaver()
        # self.prompt = agent_prompt
        # self.prompt = self.prompt.partial(system_message="You should provide accurate data for the chart_generator to use.")
        # self.prompt = self.prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        # self.agent_executor = create_react_agent(self.llm, tools, state_modifier=self.prompt,checkpointer=checkpointer)
        self.builder = StateGraph(State)
        
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH_TEXT))
        self.builder.add_node("tts", TaskFactory.create_task(TASK_TTS))
        self.builder.add_node("image_gen", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("agent", TaskFactory.create_task(TASK_AGENT))
        
        self.builder.add_edge("speech2text", "agent")
        self.builder.add_conditional_edges("agent", self.agent_edge_control, {END: END, "tts": "tts"})
        self.builder.add_edge("image_gen", END)
        self.builder.add_edge("tts", END)
        
        self.builder.add_conditional_edges(START, self.routes)
        self.graph = self.builder.compile(checkpointer=checkpointer)
        
    def routes(self,state: State, config: RunnableConfig):
        msg_type = state["input_type"]
        if msg_type == "text":
            return "agent"
        elif msg_type == "image":
            return "tranlate"
        elif msg_type == "speech":
            return "speech2text"

        return END
    
    def agent_edge_control(self,state: State):
        if state["need_speech"]:
            return "text2speech"
        return END
    
    def invoke(self,input:State):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        events = self.graph.stream(input, config)
        print(events)
        for event in events:
            print(event)
            for value in event.values():
                messages = value.get("messages")
                if messages:
                    if isinstance(messages, list):
                        messages = value["messages"][-1]
                    
                    if messages.content != "":
                        print(
                            "ai:",
                            str(messages.content).replace("\n", "\\n")[:50],
                        )
                    
                    if messages.additional_kwargs.get('media') is not None:
                        print(messages.additional_kwargs.get('media'))
    
    def display(self):
        try:
               # Generate the image as a byte stream
            image_data = self.graph.get_graph().draw_mermaid_png()

            # Create a PIL Image object from the byte stream
            image = PILImage.open(io.BytesIO(image_data))

            # Save the image to a file
            image.save("graph.png")
            print("Image saved as output_image.png")
        except Exception:
            # This requires some extra dependencies and is optional
            pass


if __name__ == '__main__':
    input_example = {
        "messages":  [
            HumanMessage(
                content="画一幅太阳",
            )
        ],
        "input_type": "image",
        "need_speech": False,
        "status": "in_progress",
    }
    
    g = AgentGraph()
    # g.display()
    # resp = g.translate_chain.invoke({"text":"请画一张地球的图片"})
    # resp = g.llm_with_tools.invoke("draw a earth picture")
    # resp = g.agent_executor.invoke(input_example)e
    # resp = g.run(input_example)
    input_example = {
        "messages":  [
            HumanMessage(
                content="超人拯救了太阳",
                additional_kwargs={"image":"/win/text-generation-webui/apps/pics/output/2024_09_16/1726452758.png"}
            )
        ],
        "input_type": "image",
        "need_speech": False,
        "status": "in_progress",
    }
    # resp = g.run(input_example)
    input_example = {
        "messages":  [
            HumanMessage(
                content="俄乌战争进展",
            )
        ],
        "input_type": "text",
        "need_speech": False,
        "status": "in_progress",
    }
    resp = g.invoke(input_example)
    # print(resp)