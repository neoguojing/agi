from agi.config import (
    log,
    BASE_URL,
    IMAGE_FILE_SAVE_PATH
)
from agi.tasks.utils import split_think_content
from agi.tasks.define import State
import json
import os
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_IMAGE_GEN
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnableConfig
)
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from agi.tasks.define import State,Feature,InputType


import traceback

intend_understand_prompt = '''
    You are an assistant whose job is to clarify and refine user requests. When you receive a user’s query, follow these steps:

    1.Detect references to prior content

        - If the user mentions a past interaction (text or image), identify and reference that specific content.

        - For images, include the URL or data if available.

    2.Rephrase for clarity and precision

        - Rewrite the user’s question so it’s concise, unambiguous, and faithful to their original intent.

    3.Output format

        - Always respond in English.

        - Return a JSON object with two fields:

            "text": the rewritten question.

            "image": the URL or data of the referenced image, or an empty string if none.
            
    Example 1:

    User: "I want to change the last picture you made for me." (Assume the last picture was an oil painting of a cat, URL: `http://localhost:8000/v1/files/1745247442.png`)

    Response:
    {{
        "text": "Modify the last generated image (an oil painting of a cat).",
        "image": "http://localhost:8000/v1/files/1745247442.png"
    }}
    
    Example 2:

    User: "Can you tell me more about the last project?" (Assume the last project was "Project Chimera - a research initiative on AI ethics.")

    Response:
    {{
        "text": "More details about the last project, 'Project Chimera - a research initiative on AI ethics.'",
        "image": ""
    }}
    
    Example 3 (addressing the redraw scenario):

    User: "The previously generated image is blurry and difficult to see, please redraw it." (Assume the previous request was for an oil painting of a landscape)

    Response:
    {{
        "text": "Redraw an oil painting of a landscape).",
        "image": ""
    }}
'''

intend_understand_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            intend_understand_prompt
        ),
        ("placeholder", "{messages}")
    ]
)

def intend_understand_modify_state_messages(state: State):
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
def intend_understand_node(state: State,config: RunnableConfig):
    if state["input_type"] == InputType.IMAGE:
        return state
    
    try:
        ai = intend_understand_chain.invoke(state)
        log.info(f"user_understand:{ai}\n{state}")
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
                    image = os.path.join(IMAGE_FILE_SAVE_PATH,os.path.basename(image))
                last_message.content.append({"type":"image","image":image})     
        log.info(f"user_understand end:{state}")
        return state
    except Exception as e:
        log.error(f"Error during user_understand output_parser: {e}")
        print(traceback.format_exc())
        return state
    
# graph
checkpointer = MemorySaver()

image_graph_builder = StateGraph(State)

image_graph_builder.add_node("intend", intend_understand_node)
image_graph_builder.add_node("image_gen", TaskFactory.create_task(TASK_IMAGE_GEN))

image_graph_builder.add_edge(START, "intend")
image_graph_builder.add_edge("intend", "image_gen")
image_graph_builder.add_edge("image_gen", END)



image_graph = image_graph_builder.compile(checkpointer=checkpointer)
