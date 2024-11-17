from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from langchain.output_parsers.boolean import BooleanOutputParser


english_traslate_template = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="Translate the following into English and only return the translation result: {text}")
    ]
)

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
        ("human", "{input}")
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
        ("human", "{input}")
    ]
)

doc_qa_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {input}""",
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
    template = """Stock Symbol or Ticker Symbol of {input}"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(input=input_text)

image_input_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a weather description assistant."),
        ("user", [
            {
                "type": "text",
                "text": "describe the weather in this image"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        ]),
    ]
)
