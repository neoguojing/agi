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
    TASK_DOC_CHAT,
    )
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Dict, Any, Iterator,Union
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage
from agi.tasks.agent import State
import traceback
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# set_verbose(True)
# set_debug(True)

# TODO
# 1. 语音：需要支持直接转换为文本和转文本之后问答 done
# 2. tts： 需要支持将直接转文本和问答之后转文本 done
# 3. rag 流程独立，支持输出检索结果和转向llm done
# 4. web检索 流程独立，支持输出检索结果和转向llm done
# 5. 加入人工check环节，返回结果，提示用户输入
# 6.模型可以绑定工具 
    
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
        self.builder.add_node("doc_chat", TaskFactory.create_task(TASK_DOC_CHAT,graph=True))
        self.builder.add_node("agent", TaskFactory.create_task(TASK_AGENT))
        self.builder.add_node("result_fix", self.result_fix)
    
        self.builder.add_conditional_edges("speech2text",self.feature_control)
        self.builder.add_conditional_edges("agent", self.tts_control)
        self.builder.add_conditional_edges("doc_chat", self.tts_control)

        # 有上下文的请求支持平行处理
        self.builder.add_edge("rag", "doc_chat")
        self.builder.add_edge("web", "doc_chat")
        # 输出rag和查询的结果，根据测试，流式结果会输出工具的查询过程，无需并行返回
        # self.builder.add_edge("rag", END)
        # self.builder.add_edge("web", END)

        self.builder.add_edge("image_gen", END)
        self.builder.add_edge("tts", END)
        self.builder.add_edge("result_fix", END)
        
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
            return self.feature_control(state)
        elif msg_type == "image":
            return "image_gen"
        elif msg_type == "audio":
            return "speech2text"

        return "result_fix"
    # 结果修正,作为流程返回的唯一回归点
    def result_fix(self,state: State, config: RunnableConfig):
        try:
            if isinstance(state.get("messages")[-1],HumanMessage):
                # 若speech直接输出，则需要将usermessage 转换为aimessage
                user_msg = state.get("messages")[-1]
                ai_msg = AIMessage(content=user_msg.content)
                state.get("messages").append(ai_msg)
        except Exception as e:
            log.error(f"{e}")
            print(traceback.format_exc())

        return state
    
    # 处理推理模型返回
    def split_think_content(self,content):
        think_content = ""
        other_content = content
        try:
            if isinstance(content,list):
                content = content[0].get("text","")
                
            import re
            match = re.search(r"(<think>\s*.*?\s*</think>)\s*(.*)", content, re.DOTALL)

            if match:
                think_content = match.group(1).replace("\n", " ").strip()  # 保留 <think> 标签，并去掉换行
                other_content = match.group(2).replace("\n", " ").strip()  # 去掉换行

        except Exception as e:
            log.error(e)

        return think_content,other_content

    def feature_control(self,state: State):
        feature = state.get("feature","agent")
        if feature == "agent":
            return "agent"
        elif feature == "rag":
            return "rag"
        elif feature == "web":
            return "web"
        elif feature == "speech" and state.get("input_type") == "audio": #仅语音转文本
            return "result_fix"
        elif feature == "tts" and state.get("input_type") == "text":    #仅文本转语音
            return "tts"
        return END
    
    def tts_control(self,state: State):
        if state["need_speech"]:
            return "tts"
        return END
    
    def invoke(self,input:State) -> State:
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": input.get("user_id",None) or str(uuid.uuid4())}}
        snapshot = self.graph.get_state(config)
        if snapshot:
            snapshot.next
            if "messages" in snapshot.values:
                existing_message = snapshot.values["messages"][-1]
                existing_message.pretty_print()
    
        events = self.graph.invoke(input, config)
        return events

    def stream(self, input: State) -> Iterator[Union[BaseMessage, Dict[str, Any]]]:
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": input.get("user_id",None) or str(uuid.uuid4())}}
        
        events = None
        # 处于打断状态的graph实例
        snapshot = self.graph.get_state(config)
        if snapshot:
            snapshot.next
            if "messages" in snapshot.values:
                existing_message = snapshot.values["messages"][-1]
                existing_message.pretty_print()
            
        events = self.graph.stream(input, config, stream_mode="values")
        try:
            for event in events:
                log.debug(event)
                if "messages" in event and event["messages"]:
                    # 这段代码，返回重复值给客户端
                    # for message in event["messages"]:
                    #     yield message  # 返回当前事件
                    # 仅返回最后一条消息
                    log.info(f"last state message:{event['messages'][-1]}")
                    last_message = event['messages'][-1]
                    # 处理推理场景，目前适配qwq
                    think_content,other_content = self.split_think_content(last_message.content)
                    if think_content != "":
                        think_message = ToolMessage(content=think_content,tool_call_id="thinking")
                        yield think_message
                        last_message.content = other_content
                        
                    yield last_message
                else:
                    log.error(f"Event missing messages: {event}")
                    yield event # 返回当前事件
        except Exception as e:
            log.error(f"Error during streaming: {e}")
            yield {"error": str(e)}  # 返回错误信息
    
    def display(self):
        try:
               # Generate the image as a byte stream
            image_data = self.graph.get_graph().draw_mermaid_png()

            # Create a PIL Image object from the byte stream
            image = PILImage.open(io.BytesIO(image_data))

            # Save the image to a file
            image.save("graph.png")
            log.debug("Image saved as output_image.png")
        except Exception:
            # This requires some extra dependencies and is optional
            pass
