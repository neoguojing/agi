from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import (
    RunnableLambda
)
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from langchain.output_parsers.boolean import BooleanOutputParser
from agi.tasks.agi_prompt import MultiModalChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from agi.tasks.utils import get_last_message_text
import json

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
    '    - If the question requests extracting information from the image (e.g., reading text, describing image content), output: "llm_with_history".'
    '    - If the question requires modifying the image, transforming its style, or generating a new image based on the input, output: "image_gen".'

    '- If the input type is "text":'
    '    - If the question indicates a request to generate or create an image (e.g., "Draw a cat", "Generate a futuristic cityscape"), output: "image_gen". '
    '    - If the question requires current or external information (e.g., latest news, real-time data, factual verification, Wikipedia, Wikidata, Python code execution, arXiv papers, weather, or stock market data), output: "agent".'
    '    - Otherwise, for typical text-based inquiries that do not require external data retrieval, output: "llm_with_history".'

    'Your output should be a single command chosen from: "image_gen", "agent", or "llm_with_history". Do not include any additional explanation or details.'

    'Examples:'
    '1. Input Type: "image"; Question: "Can you read the text in this photo?" '
    '-> Output: "llm_with_history"'

    '2. Input Type: "image"; Question: "Please convert this image into a watercolor painting." '
    '-> Output: "image_gen"'

    '3. Input Type: "text"; Question: "What is the latest update on the stock market?" '
    '-> Output: "agent"'

    '4. Input Type: "text"; Question: "Tell me about the history of the Eiffel Tower." '
    '-> Output: "llm_with_history"'
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
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "Please format your response in Markdown," 
    "including appropriate headers, lists (ordered and unordered),"
    " emphasis (using underscores or asterisks for italics and bold text), "
    "links, images, blockquotes, tables if necessary. "
    "Pay careful attention to indentation for proper rendering of these constructs."
    "\n\n"
    "{context}"
    "\n\n"
    "Please respond in {language}."
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
    # 过滤掉非法的消息类型
    state["messages"] = list(filter(lambda x: not isinstance(x.content, dict), state["messages"]))
    return doc_qa_template.invoke({"messages": state["messages"],"context":state["context"],"language":"chinese"}).to_messages()

docqa_modify_state_messages_runnable = RunnableLambda(docqa_modify_state_messages)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["text","date","results_num"],
    template="""You are an assistant tasked with improving Google search results.\
The current date is {date}.\
Generate {results_num} Google search queries that are similar to this question using English or Chinese.\
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
    output_parser=BooleanOutputParser(),
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


user_understand_prompt = '''
    You are tasked with understanding and refining the user's inquiry. Your goal is to rephrase their question to make it clearer and more precise. If the user refers to past content (such as previous messages or images), identify and reference that historical content.

    Rules:

    1.  If the user mentions or asks for something related to a previous interaction (such as a previously generated image, text, etc.), identify and reference that historical content. For image references, prepare to include the actual image URL or data in the output if it was the subject of the user's request and you have access to it.
    2.  Reconstruct their question for clarity and precision. The rephrased question must align with their original intent. 
    3.  Make sure the response is in English.
    4.  If the request is to modify a previously generated image, obtain the URL of the previous image and include it in the `image` field.
    5.  Provide your final response in JSON format, with two fields:
        * `text`: The rephrased question or request.
        * `image`: If the original query refers to a specific previous image and you have access to its URL or data, include it here. Otherwise, set this field to an empty string.

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

user_understand_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            user_understand_prompt
        ),
        ("placeholder", "{messages}")
    ]
)

def user_understand_modify_state_messages(state: AgentState):
    # 可能会存在重复的系统消息需要去掉
    filter_messages = []
    for message in state["messages"]:
        if isinstance(message,SystemMessage):
            continue
        # 修正请求的类型，否则openapi会报错
        if not isinstance(message.content,str):
             message.content = json.dumps(message.content)
        filter_messages.append(message)
    return user_understand_template.invoke({"messages": filter_messages}).to_messages()


user_understand__modify_state_messages_runnable = RunnableLambda(user_understand_modify_state_messages)