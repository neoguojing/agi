from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import (
    RunnableLambda
)
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from langchain_core.output_parsers import BaseOutputParser
from agi.tasks.agi_prompt import MultiModalChatPromptTemplate
from agi.tasks.define import AgentState
from agi.tasks.utils import get_last_message_text,get_text_from_message
from agi.config import log
import json
from datetime import datetime


class YesNoOutputParser(BaseOutputParser[bool]):
    def parse(self, text: str) -> bool:
        normalized = text.strip().upper()
        return normalized.startswith("YES")


english_traslate_template = ChatPromptTemplate.from_messages([
    ("human", "Translate the following into English and only return the translation result: {text}"),
])

# 使用默认系统模版的，消息修改器
def traslate_modify_state_messages(state: AgentState):
    text = get_last_message_text(state)
    return english_traslate_template.invoke({"text": text}).to_messages()


traslate_modify_state_messages_runnable = RunnableLambda(traslate_modify_state_messages)

agent_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
# 用于路由决策的promt，分析用户意图，提供决策路径
decider_prompt = (
    "You are an AI assistant tasked with determining a single command based on the user's input type,dialogue context, and their question. The input consists of two parts:"
    '1. Input Type: {input_type}'
    '2. Dialogue Context: {context}'
    '3. User Question: {text}'

    'Please follow these decision rules:'

    '- If the input type is "image":'
    '    - If the question requests extracting information from the image (e.g., reading text, describing image content), output: "multi_modal".'
    '    - If the question requires modifying the image, transforming its style, or generating a new image based on the input, output: "image".'

    '- If the input type is "text":'
    '    - If the question indicates a request to generate or create an image (e.g., "Draw a cat", "Generate a futuristic cityscape"), output: "image". '
    '    - If the question requires latest news, real-time data, or factual verification from the internet, output: "web_search".'
    '    - If a request involves reasoning, decision making, tool use, or multi-step interaction, output "agent".'
    '    - If the input is nonsensical, meaningless, or just gibberish, output: "llm_with_history"'
    '    - Otherwise, for typical text-based inquiries that do not require external data retrieval, output: "llm_with_history".'

    'Your output should be a single command chosen from: "image", "agent","multi_modal","web_search","web_scrape" or "llm_with_history". Do not include any additional explanation or details.'

    'Examples:'
    '1. Input Type: "image"; Question: "Can you read the text in this photo?" '
    '-> Output: "multi_modal"'

    '2. Input Type: "image"; Question: "Please convert this image into a watercolor painting." '
    '-> Output: "image"'

    '3. Input Type: "text"; Question: "What is the latest news ?"' 
    '-> Output: "web_search"'

    '4. Input Type: "text"; Question: "Summarize the key points from Wikipedia about quantum computing."' 
    '-> Output: "agent"'

    '5. Input Type: "text"; Question: "Tell me about the history of the Eiffel Tower." '
    '-> Output: "llm_with_history"'
    
    '6. Input Type: "text"; Question: "这个安装什么东西来着安装那个啊也出现自在它与晏斗是数学天文学" '
    '-> Output: "llm_with_history"'

    '7. Input Type: "text"; Question: "Provide one or more URLs to scrape content from."'
    '-> Output: "web_scrape"'
)

decide_template = ChatPromptTemplate.from_messages(
    [
        ("human", decider_prompt)
    ]
)
def decide_modify_state_messages(state: AgentState):
    # 过滤掉非法的消息类型
    state["messages"] = list(filter(lambda x: not isinstance(x.content, dict), state["messages"]))
    text = get_last_message_text(state)
    context = context = state["messages"][:-1]
    if len(state["messages"]) >= 5:
        context = state["messages"][-5:-1]

    return decide_template.invoke({"input_type": state["input_type"],"context":context,"text":text}).to_messages()

decide_modify_state_messages_runnable = RunnableLambda(decide_modify_state_messages)

system_prompt = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
    "Please respond in {language}."
)

# 给有历史的llm使用的提示
default_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),
        ("placeholder", "{messages}")
    ]
)

# 使用默认系统模版的，消息修改器
def default_modify_state_messages(state: AgentState):
    # 过滤掉非法的消息类型
    state["messages"] = list(filter(lambda x: not isinstance(x.content, dict), state["messages"]))
    # 可能会存在重复的系统消息需要去掉
    # 
    filter_messages = []
    for message in state["messages"]:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
            # 请求消息处理
            if isinstance(message,HumanMessage) and isinstance(message.content,list):
                # ollama的图片请求协议,转换为内部协议
                for item in message.content:
                    if item.get("type") == "image_url":
                        item["type"] = "image"
                        del item["image_url"]
            
            message.content = json.dumps(message.content)
        filter_messages.append(message)
    return default_template.invoke({"messages": filter_messages,"language":"chinese"}).to_messages()


default_modify_state_messages_runnable = RunnableLambda(default_modify_state_messages)

# 支持collection_names作为系统参数
custome_rag_system_prompt = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
    "\n\n"
    "{collection_names}"
    "\n\n"
    "Please respond in {language}."
)

cumstom_rag_default_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            custome_rag_system_prompt
        ),
        MessagesPlaceholder(variable_name="chat_history",optional=True),
        ("human", "{text}")
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    "Please respond in {language}."
)

contextualize_q_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt
        ),
        MessagesPlaceholder("chat_history",optional=True),
        ("human", "{text}")
    ]
)

doc_qa_prompt = (
    "Current time is {date}."
    "Answer the question using ONLY the context below. "
    "Do not assume the current date or time; always rely on the temporal information explicitly provided in the question or context."
    "If the answer is not explicitly in the context, respond 'I don't know'. "
    "Do not use external knowledge or assumptions. Keep the answer concise and preserve indentation for code or structured data.\n\n"
    "Use Markdown formatting to make your answer clear and readable. "
    "You can use headings, bullet points, numbered lists, bold, italics, or code blocks as appropriate, "
    "but do not add any information not in the context.\n\n"
    "===== CONTEXT START =====\n"
    "{context}\n"
    "===== CONTEXT END =====\n\n"
    "Self-check before responding:\n"
    "1. Ensure all answer content is directly supported by the context.\n"
    "2. If any part is unsupported, output exactly: I don't know.\n"
    "3. Do not include the self-check steps in the answer.\n\n"
    "4. Never invent or assume a current date/time.\n\n"
    "Respond in {language} using Markdown format."
)



doc_qa_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            doc_qa_prompt
        ),
        ("placeholder", "{messages}"),
    ]
)

def docqa_modify_state_messages(state: AgentState):
    messages = doc_qa_template.invoke({"messages": [state["messages"][-1]],"context":state["context"],
                                       "language":"chinese","date":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).to_messages()
    log.debug(f"docqa_modify_state_messages:{messages}")
    return messages

docqa_modify_state_messages_runnable = RunnableLambda(docqa_modify_state_messages)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["text","date","results_num"],
    template="""You are an assistant tasked with improving Google search results.\
The current date is {date}.\
Generate {results_num} Google search queries that are similar to this question using {language}.\
For any time-sensitive words, calculate the corresponding actual dates based on the specific date provided, and replace them accordingly..\
The output should be a numbered list of questions and each should have a question mark at the end:\
{text}""",
)


rag_filter_prompt = """Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant :"""

rag_filter_template = PromptTemplate(
    template=rag_filter_prompt,
    input_variables=["question", "context"],
    output_parser=YesNoOutputParser(),
)


# 用于llm模块，多模态消息的渲染
multimodal_input_template = MultiModalChatPromptTemplate(
    [
        (
            "human", [
                {"type": "text", "text": "{text}"},
                {"type": "image", "image": "{image}"},
                {"type": "audio", "audio": "{audio}"},
                {"type": "video", "video": "{video}"},
            ]
        )
    ],
    optional_variables=["text","image","audio","video"]
)


tts_format_prompt = """
You are a text-to-speech assistant. Convert the input text into a clean, easy-to-read format for TTS. 

Rules:
1. Remove extra spaces, invisible characters, and duplicate punctuation.
2. Normalize numbers, dates, currencies, and percentages into fully readable words.
3. Replace symbols that hinder pronunciation with readable alternatives:
   - - → "dash" / or space
   - / → "to" / "or"
   - & → "and" / "和"
   - @ → "at" / "艾特"
   - # → "number" / "编号"
   - % → "percent" / "百分之"
4. Add spaces between Chinese, English, and numbers where needed.
5. Split very long sentences if necessary.
6. Preserve meaning, do not add new info.
"""

tts_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            tts_format_prompt
        ),
        ("human","{text}"),
    ]
)

def tts_modify_state_messages(state: AgentState):
    text = get_text_from_message(state["messages"][-1])
    messages = tts_template.invoke({"text": text}).to_messages()
    log.debug(f"tts_modify_state_messages:{messages}")
    return messages

tts_modify_state_messages_runnable = RunnableLambda(tts_modify_state_messages)