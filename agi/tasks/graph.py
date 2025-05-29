from langgraph.graph import END, StateGraph, START
import uuid
from langgraph.types import Command, interrupt
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import  RunnableConfig
from agi.tasks.image import image_as_graph
from agi.tasks.rag_web import rag_as_subgraph

from agi.tasks.task_factory import (
    TaskFactory,
    TASK_AGENT,
    TASK_SPEECH_TEXT,
    TASK_TTS,
    TASK_LLM,
    TASK_MULTI_MODEL,
    TASK_LLM_WITH_HISTORY
    )
from typing import Dict, Any, Iterator,Union
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage
from agi.tasks.define import State,Feature,InputType
from agi.tasks.prompt import (
    decide_modify_state_messages_runnable
)
from agi.tasks.utils import split_think_content,graph_print,refine_human_message
from agi.tasks.agent import create_react_agent_as_subgraph,ahuman_feedback_node
import traceback
from agi.config import (
    log
)
from agi.tasks.tools import AskHuman

# TODO
# 1. 语音：需要支持直接转换为文本和转文本之后问答 done
# 2. tts： 需要支持将直接转文本和问答之后转文本 done
# 3. rag 流程独立，支持输出检索结果和转向llm done
# 4. web检索 流程独立，支持输出检索结果和转向llm done
# 5. 加入人工check环节，返回结果，提示用户输入 done
# 7.rag没有检索到合适的内容，则转换到agent模式，解决图片的问题 done
    
class AgiGraph:
    def __init__(self):
        checkpointer = MemorySaver()

        self.builder = StateGraph(State)
        self.builder.add_node("image", image_as_graph)
        self.builder.add_node("rag", rag_as_subgraph)
        self.builder.add_node("agent", create_react_agent_as_subgraph(TaskFactory.get_llm()))

        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH_TEXT))
        self.builder.add_node("tts", TaskFactory.create_task(TASK_TTS))
        
        self.builder.add_node("multi_modal", TaskFactory.create_task(TASK_MULTI_MODEL))
        # 用于处理非agent的请求:1.标题生成等用户自定义提示请求；2.作为决策节点，判定用户意图;3.图像识别等 image2text 请求；该请求base64，对上下文影响较大
        self.builder.add_node("llm", TaskFactory.create_task(TASK_LLM))
        #2.正常的用户对话等
        self.builder.add_node("llm_with_history", TaskFactory.create_task(TASK_LLM_WITH_HISTORY))
        
        self.builder.add_node("human_feedback", ahuman_feedback_node)

        self.builder.add_conditional_edges("human_feedback", self.human_feedback_control)
        self.builder.add_conditional_edges("agent", self.output_control)
        self.builder.add_conditional_edges("llm_with_history", self.output_control)
        self.builder.add_conditional_edges("rag", self.output_control)

        self.builder.add_edge("multi_modal", END)
        self.builder.add_edge("image", END)
        self.builder.add_edge("tts", END)
        self.builder.add_edge("llm", END)

        self.builder.add_conditional_edges("speech2text",self.text_feature_control)
        self.builder.add_conditional_edges(START, self.routes)
        self.graph = self.builder.compile(
            checkpointer=checkpointer,
            # interrupt_before=["tools"],
            # interrupt_after=["tools"],
            name="main"
            )

    # 通过用户指定input_type，来决定使用哪个分支
    async def routes(self,state: State, config: RunnableConfig):
        msg_type = state.get("input_type")

        if msg_type == InputType.TEXT:
            return await self.text_feature_control(state)
        elif msg_type == InputType.IMAGE:
            return await self.image_feature_control(state)
        elif msg_type == InputType.AUDIO:
            return await self.audio_feature_control(state)
        elif msg_type == InputType.VIDEO:
            return await self.video_feature_control(state)

        return END
    
        
    async def auto_state_machine(self,state: State):
        config={"configurable": {"user_id": "tools", "conversation_id": "",
                                 "thread_id": "tools"}}
        # 定义状态机chain
        decider_chain = decide_modify_state_messages_runnable | TaskFactory.get_llm() | StrOutputParser()
        node_list = ["image","llm_with_history","agent","llm",END]
        next_step = await decider_chain.ainvoke(state,config=config)
        # 去除think标签
        _,next_step = split_think_content(next_step)
        log.info(f"auto_state_machine: {next_step}")
        
        result = "llm_with_history"
        # 判断返回是否在决策列表里
        if next_step in node_list:
            result = next_step
        else: # 若大模型返回的值不标准，则看是否包含节点
            match = next((option for option in node_list if option in next_step), None)
            if match:
                result = match
        state["auto_decide_result"] = result

        return result
    
    # 控制图像输入决策
    async def image_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.IMAGE2TEXT:    #图片转文字,涉及到使用base64，对上下文影响较大，不能进入上下文
            return "llm"
        elif feature == Feature.IMAGE2IMAGE:    #图片转图片
            return "image"
        else:
            return await self.auto_state_machine(state)
    
    # 语音输出决策
    async def audio_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.VOICECHAT:  #语音对话
            return "multi_modal"
        return "speech2text"
    
    # 视频输入决策
    async def video_feature_control(self,state: State):
        feature = state.get("feature","")
        if feature == Feature.VIDEOPARSE:  #视频内容解析
            return "multi_modal"
        return "multi_modal"

    # 文本输入决策
    async def text_feature_control(self,state: State):
        if state.get("user_id") == "raspberrypi":
            refine_human_message(state,lambda x:f"{x} /no_think")
        feature = state.get("feature","")
        if feature == Feature.AGENT:
            return "agent"
        elif feature == Feature.RAG:
            return "rag"
        elif feature == Feature.WEB:
            return "rag"
        elif feature == Feature.TTS:   #文字转语音
            return "tts"
        elif feature == Feature.SPEECH:  #语音转文字，直接输出
            return END
        elif feature == Feature.LLM:  #处理任务类相关请求，如自动标题、tag、提示完成等
            return "llm"
        elif feature == Feature.HUMAN: #人工介入，用于测试
            return "human_feedback"
        else: #通用任务处理：如标题生成、tag生成等 或者 自主决策
            return await self.auto_state_machine(state)
    
    async def human_feedback_control(self,state: State):
        if state["step"]:
            return state["step"][-1]
        
        return await self.output_control(state)
    
    async def output_control(self,state: State):
        if state["need_speech"]:
            log.info(f"to tts:{state['messages'][-1]}")
            return "tts"
        
        return END
    
        
    async def invoke(self,input:State) -> State:
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id": input.get("conversation_id",None) or str(uuid.uuid4()),
                                 "need_speech":input.get("need_speech",False)}}
        state = self.graph.get_state(config)
        # Print the pending tasks
        log.debug(state.tasks)
        events = None
        # TODO tasks只有在非空情况下一定是打断吗
        if state.tasks and state.tasks[0].interrupts:
            events = await self.graph.ainvoke(Command(resume=input), config)
        else:
            input["step"] = []
            events = await self.graph.ainvoke(input, config)
        return events

    async def stream(self, input: State,stream_mode=["messages","updates", "custom"]) -> Iterator[Union[BaseMessage, Dict[str, Any]]]:
        thread_id =  input.get("conversation_id",None) or str(uuid.uuid4())
        config={"configurable": {"user_id": input.get("user_id","default_tenant"), "conversation_id": input.get("conversation_id",""),
                                 "thread_id":thread_id,"need_speech":input.get("need_speech",False)}}
        
        events = None        
        try:
            state = self.graph.get_state(config)
            log.debug(state)
            if state.tasks and state.tasks[0].interrupts:
                # events = self.graph.astream(Command(resume=input),config=config,stream_mode=stream_mode,subgraphs=True)
                events = self.graph.astream(Command(resume=input),config=config,stream_mode=stream_mode)
            else:
                input["step"] = []
                # events = self.graph.astream(input, config=config, stream_mode=stream_mode,subgraphs=True)
                events = self.graph.astream(input, config=config, stream_mode=stream_mode)

            async for event in events:
                log.debug(f"stream-event:{event}")
                if not isinstance(event,tuple):
                    continue
                # 返回非HumanMessage
                if "values" in stream_mode:
                    # event是 State类型
                    if "messages" in event and event["messages"]:
                        log.debug(f"last state message:{event['messages'][-1]}")
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
                    else:
                        continue
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
                        log.debug(f"stream-event-message:{event}")
                        if meta.get("langgraph_node") in ["web","__start__","rag",'user_understand',"compress","intend"]:
                            continue
                        else:
                            # 某些场景下，如agent，返回消息非流式返回，整体作为一个返回：
                            # 1.finish_reason一定等于stop
                            # 2.在包含think的场景下，think的内容一起返回，导致出现问题
                            # 3.此处将该类消息拆为两条,分别发送
                            last_message = event[1][0]
                            if last_message.response_metadata.get("finish_reason","") in ["stop","tool_calls"] and last_message.content:
                                think_content,other_content = split_think_content(last_message.content)
                                if think_content:
                                    last_message.content = think_content
                                    last_message.response_metadata["finish_reason"] = None
                                    yield event
                                    last_message.content = other_content
                                    last_message.response_metadata["finish_reason"] = "stop"
                            yield event
                    else:
                        continue
                elif stream_mode == "debug":
                    continue
                else:
                    continue
        except Exception as e:
            log.error(f"Error during streaming: {e}")
            print(traceback.format_exc())
            yield {"error": str(e)}  # 返回错误信息
    
    def display(self):
        graph_print(self.graph)
