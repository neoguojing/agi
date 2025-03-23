from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate

from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from langchain.output_parsers.boolean import BooleanOutputParser

from agi.tasks.agi_prompt import MultiModalChatPromptTemplate

english_traslate_template = ChatPromptTemplate.from_messages([
        ("human", "Translate the following into English and only return the translation result: {text}"),
    ])

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

system_prompt = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
    "Please use {language} as default language."
)

default_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{text}")
    ]
)

# 支持collection_names作为系统参数
custome_rag_system_prompt = (
    "You are a helpful assistant. Answer all questions to the best of your ability."
    "\n\n"
    "{collection_names}"
    "\n\n"
    "Please use {language} as default language."
)

cumstom_rag_default_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            custome_rag_system_prompt
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{text}")
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    "Please use {language} as default language."
)

contextualize_q_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt
        ),
        MessagesPlaceholder("chat_history"),
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
    "Please use {language} as default language."
)

doc_qa_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            doc_qa_prompt
        ),
        # MessagesPlaceholder("chat_history"),
        ("human", "{text}")
    ]
)

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


def stock_code_prompt(input_text):
    template = """Stock Symbol or Ticker Symbol of {text}"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(input=input_text)


# multimodal_input_template = PromptTemplate(
#     template='{"type":"{{type}}","data":"{{data}}","text":"{{text}}"}',
#     partial_variables={"text":None,"type":None,"data":None},
#     template_format="mustache"
# )

multimodal_input_template = MultiModalChatPromptTemplate(
    [
        (
            "human", [
                {"type": "text", "text": "{text}"},
                {"type": "image", "image": "{image}"},
                {"type": "audio", "audio": "{audio}"},
            ]
        )
    ],
    optional_variables=["text","image","audio"]
)
