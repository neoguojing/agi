from langchain_core.messages import AIMessage, HumanMessage,BaseMessage,ToolMessage
from typing import Any, List, Mapping, Optional, Union
from langchain_core.runnables import (
    RunnableLambda
)
from langchain_core.messages import (
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from agi.tasks.define import AgentState
from langgraph.graph.message import Messages
from agi.config import log,AGI_DEBUG
import inspect
import json
import traceback
import uuid
import hashlib

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

def compute_content_hash(content: any) -> str:
    """
    计算 content 的哈希值。
    如果 content 是列表或字典，则将其序列化为 JSON 字符串（sort_keys=True确保稳定顺序）。
    对于其他类型，则转换为字符串。
    """
    if isinstance(content, (dict, list)):
        # 序列化为 JSON 字符串，并保证键排序以获得一致的结果
        serialized = json.dumps(content, sort_keys=True)
    else:
        serialized = str(content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def add_messages(left: Messages, right: Messages) -> Messages:
    """
    合并两个消息列表，依据消息的 content 哈希值和消息类型进行更新或删除。

    如果右侧消息与左侧已有消息的 content（基于哈希）和消息类型相同，
    则用右侧消息替换左侧消息；如果右侧消息为 RemoveMessage，
    则删除所有匹配的消息。

    Args:
        left: 基础消息列表（或者单条消息）
        right: 要合并的消息列表（或者单条消息）

    Returns:
        合并后的消息列表
    """
    try:
        # 转换为列表
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        log.debug(f"add_messages--begin--{left} \n {right}")
        left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
        right = [message_chunk_to_message(m) for m in convert_to_messages(right)]
        log.debug(f"add_messages--after--{left} \n {right}")
        # 为缺失 id 的消息分配唯一 ID
        for m in left:
            if m.id is None:
                m.id = str(uuid.uuid4())
        for m in right:
            if m.id is None:
                m.id = str(uuid.uuid4())

        # 使用 content 的哈希值构建 key，同时结合消息类型
        def make_key(m):
            content_hash = compute_content_hash(m.content)
            return (content_hash, m.__class__.__name__)

        left_idx_by_key = {make_key(m): i for i, m in enumerate(left)}
        merged = left.copy()
        keys_to_remove = set()

        for m in right:
            key = make_key(m)
            if key in left_idx_by_key:
                if isinstance(m, RemoveMessage):
                    keys_to_remove.add(key)
                else:
                    merged[left_idx_by_key[key]] = m
            else:
                if isinstance(m, RemoveMessage):
                    raise ValueError(
                        f"Attempting to delete a message with content (hash) and type that doesn't exist: {key}"
                    )
                merged.append(m)

        merged = [m for m in merged if make_key(m) not in keys_to_remove]
        log.debug(f"--------{merged}")
        return merged
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())
        