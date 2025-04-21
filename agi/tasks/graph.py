import io
from PIL import Image as PILImage
from langgraph.graph import END, StateGraph, START
import uuid
from langgraph.types import Command, interrupt
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import StreamWriter
from langchain_core.runnables import  RunnableConfig,Runnable,RunnablePassthrough
from langchain.globals import set_debug
from langchain.globals import set_verbose
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_AGENT,
    TASK_IMAGE_GEN,
    TASK_SPEECH_TEXT,
    TASK_TTS,
    TASK_WEB_SEARCH,
    TASK_RAG,
    TASK_DOC_CHAT,
    TASK_LLM,
    TASK_MULTI_MODEL,
    TASK_LLM_WITH_HISTORY
    )
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Dict, Any, Iterator,Union
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage,AIMessageChunk
from agi.tasks.agent import State,Feature,InputType
from agi.tasks.prompt import decide_modify_state_messages_runnable
from agi.tasks.utils import split_think_content
import traceback
import threading
from agi.config import log
from agi.tasks.tools import AskHuman
# set_verbose(True)
# set_debug(True)

# TODO
# 1. 语音：需要支持直接转换为文本和转文本之后问答 done
# 2. tts： 需要支持将直接转文本和问答之后转文本 done
# 3. rag 流程独立，支持输出检索结果和转向llm done
# 4. web检索 流程独立，支持输出检索结果和转向llm done
# 5. 加入人工check环节，返回结果，提示用户输入
# 6.模型可以绑定工具 
# 7.rag没有检索到合适的内容，则转换到agent模式，解决图片的问题
    
class AgiGraph:
    def __init__(self):
        checkpointer = MemorySaver()

        self.builder = StateGraph(State)
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH_TEXT))
        self.builder.add_node("tts", TaskFactory.create_task(TASK_TTS))
        self.builder.add_node("image_gen", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("rag", TaskFactory.create_task(TASK_RAG))
        self.builder.add_node("web", TaskFactory.create_task(TASK_WEB_SEARCH))
        # self.builder.add_node("doc_chat", TaskFactory.create_task(TASK_DOC_CHAT))
        self.builder.add_node("doc_chat", self.doc_chat_node)
        self.builder.add_node("agent", TaskFactory.create_task(TASK_AGENT))
        self.builder.add_node("multi_modal", TaskFactory.create_task(TASK_MULTI_MODEL))
        # 用于处理非agent的请求:1.标题生成等用户自定义提示请求；2.作为决策节点，判定用户意图
        self.builder.add_node("llm", TaskFactory.create_task(TASK_LLM))
        #1.图像识别等 image2text 请求；2.正常的用户对话等
        self.builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
        
        self.builder.add_node("human_feedback", self.human_feedback)
        
        self.builder.add_conditional_edges("human_feedback", self.human_feedback_control)
        self.builder.add_conditional_edges("agent", self.agent_control)
        self.builder.add_conditional_edges("llm", self.output_control)
        self.builder.add_conditional_edges("llm_with_history", self.output_control)
        self.builder.add_conditional_edges("doc_chat", self.output_control)

        # 有上下文的请求支持平行处理
        self.builder.add_conditional_edges("rag", self.context_control)
        self.builder.add_conditional_edges("web", self.context_control)
   
        

        self.builder.add_edge("multi_modal", END)
        self.builder.add_edge("image_gen", END)
        self.builder.add_edge("tts", END)
        
        
        
        self.builder.add_conditional_edges("speech2text",self.text_feature_control)
        self.builder.add_conditional_edges(START, self.routes)
        self.graph = self.builder.compile(
            checkpointer=checkpointer,
            # interrupt_before=["tools"],
            # interrupt_after=["tools"]
            )
        
        # 定义状态机chain
        self.decider_chain = decide_modify_state_messages_runnable | TaskFactory.get_llm() | StrOutputParser()
        self.node_list = ["image_gen","llm_with_history","agent"]

    # 通过用户指定input_type，来决定使用哪个分支
    def routes(self,state: State, config: RunnableConfig):
        msg_type = state.get("input_type")
        # 状态初始化
        state["context"] = None
        state["docs"] = None
        state["citations"] = None

        if msg_type == InputType.TEXT:
            return self.text_feature_control(state)
        elif msg_type == InputType.IMAGE:
            return self.image_feature_control(state)
        elif msg_type == InputType.AUDIO:
            return self.audio_feature_control(state)
        elif msg_type == InputType.VIDEO:
            return self.video_feature_control(state)

        return END
    
    # 文档对话
    def doc_chat_node(self,state: State,config: RunnableConfig,writer: StreamWriter):
        chain = TaskFactory.create_task(TASK_DOC_CHAT)
        if state["citations"]:
            writer({"citations":state["citations"],"docs":state["docs"]})
        return chain.invoke(state,config=config)
    
    def auto_state_machine(self,state: State):
        config={"configurable": {"user_id": "tools", "conversation_id": "",
                                 "thread_id": "tools"}}
        next_step = self.decider_chain.invoke(state,config=config)
        # 去除think标签
        _,next_step = split_think_content(next_step)
        log.info(f"auto_state_machine: {next_step}")
        # 判断返回是否在决策列表里
        if next_step in self.node_list:
            state["auto_decide_reuslt"] = next_step
            return next_step
        # 若大模型返回的值不标准，则看是否包含节点
        match = next((option for option in self.node_list if option in next_step), None)
        if match:
            state["auto_decide_reuslt"] = match
            return match
        state["auto_decide_reuslt"] = "llm_with_history"
        return "llm_with_history"
    
    # 控制图像输入决策
    def image_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.IMAGE2TEXT:    #图片转文字
            return "llm_with_history"
        elif feature == Feature.IMAGE2IMAGE:    #图片转图片
            return "image_gen"
        else:
            return self.auto_state_machine(state)
    
    # 语音输出决策
    def audio_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.VOICECHAT:  #语音对话
            return "multi_modal"
        return "speech2text"
    
    # 视频输入决策
    def video_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.VIDEOPARSE:  #视频内容解析
            return "multi_modal"
        return "multi_modal"

    # 文本输入决策
    def text_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.AGENT:
            return "agent"
        elif feature == Feature.RAG:
            return "rag"
        elif feature == Feature.WEB:
            return "web"
        elif feature == Feature.TTS:   #文字转语音
            return "tts"
        elif feature == Feature.SPEECH:  #语音转文字，直接输出
            return END
        elif feature == Feature.LLM:  #处理任务类相关请求，如自动标题、tag、提示完成等
            return "llm"
        elif feature == Feature.HUMAN: #人工介入，用于测试
            return "human_feedback"
        else: #通用任务处理：如标题生成、tag生成等 或者 自主决策
            return self.auto_state_machine(state)
    
    def agent_control(self,state: State):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return self.output_control(state=state)
        # If tool call is asking Human, we return that node
        # You could also add logic here to let some system know that there's something that requires Human input
        # For example, send a slack message, etc
        elif last_message.tool_calls[0]["name"] == "AskHuman":
            state["step"].append("agent")
            return "human_feedback"
        # Otherwise if there is, we continue
        # else:
        #     return "action"
    
    def human_feedback_control(self,state: State):
        if state["step"]:
            return state["step"][-1]
        
        return END
    
    def output_control(self,state: State):
        if state["need_speech"]:
            return "tts"
        # 处理agent返回冗余信息的问题，从倒数第一个humanmessage开始输出
        # for i in range(len(state["messages"])-1, -1, -1):
        #     if isinstance(state["messages"][i],HumanMessage):
        #         state["messages"] = state["messages"][i:]
        #         break
        return END
    
    # 适用于web 和 rag的情况，当无法获取有效的上下文信息时，
    # 1.重置feature特性
    # 2.交给agent处理
    def context_control(self,state: State):
        docs = state.get("docs")
        if docs:
            return "doc_chat"
        return "agent"
    
    def human_feedback(self,state: State):
        messages = []
        # agent的场景,需要使用到AskHuman
        if isinstance(state["messages"][-1],ToolMessage):
            ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
            # feedback的类型是State
            feedback = interrupt(ask.question)
            return feedback
        elif isinstance(state["messages"][-1],HumanMessage): #用于测试
            feedback = interrupt("breaked")
            # TODO 此处并没有返回
            messages = [AIMessage(content=feedback["messages"][-1].content)]
            state["messages"] = messages
            return state
        
        return state
        
    def invoke(self,input:State) -> State:
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": input.get("user_id",None) or str(uuid.uuid4())}}
        state = self.graph.get_state(config)
        # Print the pending tasks
        log.debug(state.tasks)
        events = None
        # TODO tasks只有在非空情况下一定是打断吗
        if state.tasks:
            events = self.graph.invoke(Command(resume=input), config)
        else:
            input["step"] = []
            events = self.graph.invoke(input, config)
        return events

    def stream(self, input: State,stream_mode=["messages","updates", "custom"]) -> Iterator[Union[BaseMessage, Dict[str, Any]]]:
        thread_id =  input.get("user_id",None) or str(uuid.uuid4())
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id":thread_id}}
        
        events = None        
        try:
            state = self.graph.get_state(config)
            log.debug(state)
            if state.tasks:
                events = self.graph.invoke(Command(resume=input), config)
            else:
                input["step"] = []
                events = self.graph.stream(input, config=config, stream_mode=stream_mode)

            for event in events:
                log.debug(event)
                # 返回非HumanMessage
                if "values" in stream_mode:
                    # event是 State类型
                    if "messages" in event and event["messages"]:
                        log.info(f"last state message:{event['messages'][-1]}")
                        last_message = event['messages'][-1]
                        # 部返回最后一个
                        if isinstance(last_message,HumanMessage):
                            continue
                        # 处理推理场景，目前适配qwq
                        think_content,other_content = split_think_content(last_message.content)
                        if think_content != "":
                            think_message = ToolMessage(content=think_content,tool_call_id="thinking")
                            yield think_message
                            last_message.content = other_content
                        yield last_message
                    else:
                        log.error(f"Event missing messages: {event}")
                elif "updates" in stream_mode and event[0] == "updates":
                    # 可以拿到每个节点的信息
                    # event 类型： {"agent":State} {"web":State} {'doc_chat': None}
                    # ('updates', {'__interrupt__': (Interrupt(value='Please provide feedback:', resumable=True, ns=['human_feedback:0a8efb87-cce4-22f2-f8e6-23744b8946b7'], when='during'),)})
                    # 仅返回__interrupt__消息
                    if event[1].get("__interrupt__"):
                        yield event
                elif "custom" in stream_mode and event[0] == "custom":
                    # 用户自定义消息
                    # ("custom":())
                    yield event
                elif "messages" in stream_mode and event[0] == "messages": 
                    # turple 类型的消息 0是AIMessageChunk，2是一个字典
                    # (AIMessageChunk,{}) (ToolMessage,{}) (HumanMessage,{})
                    '''
                    ("messages",(
                        AIMessageChunk(content='0', additional_kwargs={}, response_metadata={}, id='run-63b5b380-baa6-4e72-a13d-653d9148585d'),
                        {'user_id': 'default_tenant', 'conversation_id': '', 'thread_id': 'ebbdc908-a785-4036-900a-7298aac68cb0', 'langgraph_step': 1, 'langgraph_node': 'web', 'langgraph_triggers': ['branch:__start__:routes:web'], 'langgraph_path': ('__pregel_pull', 'web'), 'langgraph_checkpoint_ns': 'web:fdb11f55-aa59-ab33-4fde-ec903ab4ef98', 'checkpoint_ns': 'web:fdb11f55-aa59-ab33-4fde-ec903ab4ef98', 'ls_provider': 'openai', 'ls_model_name': 'qwen2.5:14b', 'ls_model_type': 'chat', 'ls_temperature': 0.7}
                    ))
                    '''
                    # 仅返回AIMessageChunk以及content不能为空,过滤ToolMessage和HumaMessage
                    # 多模态场景下,会返回AIMessage
                    # TODO decide chain 和 tranlate chain 以及 web search chain会输出中间结果,需要想办法处理
                    if (isinstance(event[1][0],AIMessage)) and event[1][0].content:
                        meta = event[1][1]
                        if meta.get("langgraph_node") in ["web","__start__","rag"]:
                            pass
                        else:
                            yield event
                elif stream_mode == "debug":
                    pass
        except Exception as e:
            log.error(f"Error during streaming: {e}")
            print(traceback.format_exc())
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
