import io
from PIL import Image as PILImage
from langgraph.graph import END, StateGraph, START
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import  RunnableConfig,Runnable,RunnablePassthrough
from langchain.globals import set_debug
from langchain.globals import set_verbose
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_AGENT,
    TASK_IMAGE_GEN,
    TASK_LLM_WITH_RAG,
    TASK_SPEECH_TEXT,
    TASK_TTS,
    TASK_WEB_SEARCH,
    TASK_CUSTOM_RAG,
    )
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Dict, Any, Iterator,Union
from langchain_core.messages import BaseMessage
set_verbose(True)
set_debug(True)

class State(AgentState):
    input_type: str
    need_speech: bool
    status: str
    user_id: str
    conversation_id: str
    feature: str  # 支持的特性，1.agent，2.web 3.rag，默认为agent


    
class AgiGraph:
    def __init__(self):
        # TODO 
        checkpointer = MemorySaver()
        # self.prompt = agent_prompt
        # self.prompt = self.prompt.partial(system_message="You should provide accurate data for the chart_generator to use.")
        # self.prompt = self.prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        # self.agent_executor = create_react_agent(self.llm, tools, state_modifier=self.prompt,checkpointer=checkpointer)
        self.builder = StateGraph(State)
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH_TEXT,graph=True))
        self.builder.add_node("tts", TaskFactory.create_task(TASK_TTS,graph=True))
        self.builder.add_node("image_gen", TaskFactory.create_task(TASK_IMAGE_GEN,graph=True))
        self.builder.add_node("rag", TaskFactory.create_task(TASK_CUSTOM_RAG,graph=True))
        self.builder.add_node("web", TaskFactory.create_task(TASK_WEB_SEARCH,graph=True))
        self.builder.add_node("agent", TaskFactory.create_task(TASK_AGENT))
    
        self.builder.add_conditional_edges("speech2text",self.speech_edge_control)
        self.builder.add_conditional_edges("agent", self.llm_edge_control, {END: END, "tts": "tts"})
        self.builder.add_conditional_edges("rag", self.llm_edge_control, {END: END, "tts": "tts"})
        self.builder.add_conditional_edges("web", self.llm_edge_control, {END: END, "tts": "tts"})
        self.builder.add_edge("image_gen", END)
        self.builder.add_edge("tts", END)
        
        self.builder.add_conditional_edges(START, self.routes)
        self.graph = self.builder.compile(
            checkpointer=checkpointer,
            # interrupt_before=["tools"],
            # interrupt_after=["tools"]
            )
    # 通过用户指定input_type，来决定使用哪个分支
    def routes(self,state: State, config: RunnableConfig):
        msg_type = state.get("input_type")
        if msg_type == "text":
            return self.speech_edge_control(state)
        elif msg_type == "image":
            return "image_gen"
        elif msg_type == "audio":
            return "speech2text"

        return END
    
    def speech_edge_control(self,state: State):
        feature = state.get("feature","agent")
        if feature == "agent":
            return "agent"
        elif feature == "rag":
            return "rag"
        elif feature == "web":
            return "web"

        return END
    
    def llm_edge_control(self,state: State):
        if state["need_speech"]:
            return "tts"
        return END
    
    def invoke(self,input:State) -> State:
        config={"configurable": {"user_id": input.get("user_id",""), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": str(uuid.uuid4())}}
        events = self.graph.invoke(input, config)
        return events

    def stream(self, input: State) -> Iterator[Union[BaseMessage, Dict[str, Any]]]:
        config={"configurable": {"user_id": input.get("user_id",""), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": str(uuid.uuid4())}}
        events = self.graph.stream(input, config, stream_mode="values")

        try:
            for event in events:
                print(event)
                if "messages" in event and event["messages"]:
                    for message in event["messages"]:
                        yield message  # 返回当前事件
                else:
                    print(f"Event missing messages: {event}")
                    yield event # 返回当前事件
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield {"error": str(e)}  # 返回错误信息
    
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
