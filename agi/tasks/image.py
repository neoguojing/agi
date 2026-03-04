from agi.config import (
    log,
    BASE_URL,
    FILE_STORAGE_PATH
)
from agi.tasks.utils import split_think_content,graph_print
from agi.tasks.define import State
import json
import os
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_IMAGE_GEN,
    TASK_LLM_WITH_HISTORY
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig
)
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.pregel import RetryPolicy
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.tasks.define import State,Feature,InputType
from langgraph.types import Command

import traceback

intend_understand_prompt = """
You are an assistant whose only task is to rewrite user requests into a strict JSON object.

Rules:
1. Detect references to prior content (text or image).
   - If the user refers to a past image, include its URL or data in "image".
   - Otherwise, set "image" to an empty string "".

2. Rephrase the user's request concisely and clearly into English, and put it in "text".

3. Output format:
   - Output only a single valid JSON object.
   - Do not add explanations, prefixes, suffixes, or markdown.
   - The JSON object must always have exactly two fields: "text" and "image".

Examples:

User: "I want to change the last picture you made for me."
(Last picture: oil painting of a cat, URL: http://localhost:8000/v1/files/1745247442.png)

Output:
{{"text": "Modify the last generated image (an oil painting of a cat).", "image": "http://localhost:8000/v1/files/1745247442.png"}}

User: "Can you tell me more about the last project?"
(Last project: Project Chimera - a research initiative on AI ethics.)

Output:
{{"text": "More details about the last project, 'Project Chimera - a research initiative on AI ethics.'", "image": ""}}

User: "The previously generated image is blurry and difficult to see, please redraw it."
(Previous request: oil painting of a landscape.)

Output:
{{"text": "Redraw an oil painting of a landscape.", "image": ""}}
"""


intend_understand_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            intend_understand_prompt
        ),
        ("placeholder", "{messages}")
    ]
)

async def intend_understand_modify_state_messages(state: State):
    # 可能会存在重复的系统消息需要去掉
    filter_messages = []
    for message in state["messages"]:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
             message.content = json.dumps(message.content)
        filter_messages.append(message)
    return intend_understand_template.invoke({"messages": filter_messages}).to_messages()


intend_understand__modify_state_messages_runnable = RunnableLambda(intend_understand_modify_state_messages)

intend_understand_chain = intend_understand__modify_state_messages_runnable | TaskFactory.get_llm() 

# 理解用户意图，并生成结构化的输入
async def intend_understand_node(state: State,config: RunnableConfig):
    if state["input_type"] == InputType.IMAGE:
        return state
    
    try:
        ai = await intend_understand_chain.ainvoke(state)
        log.info(f"intend_understand_node:{ai}\n{state}")
        # think 标签过滤
        _, result = split_think_content(ai.content)
        log.debug(result)
        obj = json.loads(result)
        text = obj.get("text")
        image = obj.get("image")
        last_message = state["messages"][-1]
        if text and last_message:
            last_message.content = [{"type":"text","text":text}]
            if image:
                if image.startswith(BASE_URL):
                    image = os.path.join(FILE_STORAGE_PATH,os.path.basename(image))
                last_message.content.append({"type":"image","image":image})     
        else:
            raise ValueError("text or last_message is missing")
        log.info(f"user_understand end:{state}")
        return state
    except Exception as e:
        log.error(f"Error during user_understand output_parser: {e}")
        print(traceback.format_exc())
        raise
  
# graph
checkpointer = MemorySaver()

image_graph_builder = StateGraph(State)

image_graph_builder.add_node("intend", intend_understand_node,retry=RetryPolicy(retry_on=[json.JSONDecodeError, TypeError,ValueError]))
image_graph_builder.add_node("image_gen", TaskFactory.create_task(TASK_IMAGE_GEN))

image_graph_builder.add_edge(START, "intend")
image_graph_builder.add_edge("intend", "image_gen")
image_graph_builder.add_edge("image_gen", END)

image_graph = image_graph_builder.compile(checkpointer=checkpointer,name="image")
image_as_graph = image_graph_builder.compile(checkpointer=checkpointer,name="image")

graph_print(image_graph)

