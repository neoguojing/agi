import io
from PIL import Image as PILImage
from langgraph.graph import END, StateGraph, START
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import  RunnableConfig,Runnable,RunnablePassthrough
from langchain.globals import set_debug
from langchain.globals import set_verbose
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from agi.tasks.task_factory import TaskFactory,TASK_AGENT,TASK_IMAGE_GEN,TASK_LLM_WITH_RAG,TASK_SPEECH_TEXT,TASK_TTS

set_verbose(True)
# set_debug(True)

class State(TypedDict):
    # Append-only chat memory so the agent can try to recover from initial mistakes.
    messages: Annotated[list[AnyMessage], add_messages]
    text: str
    image: str
    audio: str
    input_type: str
    need_speech: bool
    status: str
    
class AgiGraph:
    def __init__(self):
        
        checkpointer = MemorySaver()
        # self.prompt = agent_prompt
        # self.prompt = self.prompt.partial(system_message="You should provide accurate data for the chart_generator to use.")
        # self.prompt = self.prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        # self.agent_executor = create_react_agent(self.llm, tools, state_modifier=self.prompt,checkpointer=checkpointer)
        self.builder = StateGraph(State)
        print("**********",type(TaskFactory.create_task(TASK_SPEECH_TEXT)))
        self.builder.add_node("speech2text", self.node_wrapper(TaskFactory.create_task(TASK_SPEECH_TEXT)))
        self.builder.add_node("tts", self.node_wrapper(TaskFactory.create_task(TASK_TTS)))
        self.builder.add_node("image_gen", self.node_wrapper(TaskFactory.create_task(TASK_IMAGE_GEN)))
        self.builder.add_node("agent", self.node_wrapper(TaskFactory.create_task(TASK_AGENT)))
        
        self.builder.add_edge("speech2text", "agent")
        self.builder.add_conditional_edges("agent", self.agent_edge_control, {END: END, "tts": "tts"})
        self.builder.add_edge("image_gen", END)
        self.builder.add_edge("tts", END)
        
        self.builder.add_conditional_edges(START, self.routes)
        self.graph = self.builder.compile(
            checkpointer=checkpointer,
            # interrupt_before=["tools"],
            # interrupt_after=["tools"]
            )
    def node_wrapper(self,node: Runnable):
        return RunnablePassthrough.assign(messages=node)
        
    def routes(self,state: State, config: RunnableConfig):
        msg_type = state["input_type"]
        if msg_type == "text":
            return "agent"
        elif msg_type == "image":
            return "image_gen"
        elif msg_type == "audio":
            return "speech2text"

        return END
    
    def agent_edge_control(self,state: State):
        if state["need_speech"]:
            return "tts"
        return END
    
    def invoke(self,input:State):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        events = self.graph.invoke(input, config)
        print(events)
    
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
