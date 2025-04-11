from langchain_core.messages import AIMessage, HumanMessage,BaseMessage,ToolMessage
from typing import Any, List, Mapping, Optional, Union
from langchain_core.runnables import (
    RunnableLambda
)
from langgraph.prebuilt.chat_agent_executor import AgentState
from agi.config import log,AGI_DEBUG
import inspect

# 处理推理模型返回
def split_think_content(content):
    think_content = ""
    other_content = content
    try:
        if isinstance(content,list):
            content = content[0].get("text","")
            
        import re
        match = re.search(r"(<think>\s*.*?\s*</think>)\s*(.*)", content, re.DOTALL)

        if match:
            think_content = match.group(1).strip()  # 保留 <think> 标签，并去掉换行
            other_content = match.group(2).strip()  # 去掉换行

    except Exception as e:
        log.error(e)

    return think_content,other_content

def get_last_message_text(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,str):
            return last_message.content
        elif isinstance(last_message.content,list):
            for item in last_message.content:
                if item["type"] == "text":
                    return item["text"]
    return ""

# 修复最后一条AI消息的text内容,去除特定标签内容
def refine_last_message_text(message :Union[AIMessage,ToolMessage,list[BaseMessage]]):
    last_message = message
    if isinstance(message,list):
        last_message = message[-1]

    if not isinstance(last_message,HumanMessage):
        if isinstance(last_message.content,str):
            _,last_message.content = split_think_content(last_message.content)
        elif isinstance(last_message.content,list):
            for item in last_message.content:
                if item["type"] == "text":
                    _,item["text"] = split_think_content(item["text"])
    return last_message


refine_last_message_runnable = RunnableLambda(refine_last_message_text)

def graph_response_format(message :Union[AIMessage,ToolMessage,list[BaseMessage]]):
    
    refine_last_message_text(message)
    if isinstance(message,list):
        return {"messages": message}
    
    return {"messages": [message]} 

graph_response_format_runnable = RunnableLambda(graph_response_format)

# TODO parent_name不是实际的函数
def debug_info(x : Any):
    if AGI_DEBUG:
        parent_name = ""
        stack = inspect.stack()
        if len(stack) > 2:  # stack[0] 是 get_parent_function_name，stack[1] 是调用它的函数
            parent_name = stack[2].function  # stack[2] 是再往上的函数，即父函数
        
        log.info(f"type:{parent_name}\nmessage:{x}")

    return x

debug_tool = RunnableLambda(debug_info)