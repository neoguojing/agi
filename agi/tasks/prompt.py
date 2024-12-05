from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate,PromptTemplate,BaseChatPromptTemplate
from langchain.prompts.chat import MessageLikeRepresentation,BaseMessagePromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from typing import (
    Any,
    List,
    Union,
    Sequence,
    cast
)

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
        ("human", "{text}")
    ]
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["input"],
    template="""You are an assistant tasked with improving Google search \
results. Generate {results_num} Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {text}""",
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


# class MultiModalPromptTemplate(ChatPromptTemplate):
#     prompt: Union[
#         StringPromptTemplate, list[Union[StringPromptTemplate, ImagePromptTemplate]]
#     ]
    
#     # 来自于ChatPromptTemplate TODO
#     def __init__(
#         self,
#         messages: Sequence[MessageLikeRepresentation],
#         *,
#         template_format: PromptTemplateFormat = "f-string",
#         **kwargs: Any,
#     ) -> None:
#         """Create a chat prompt template from a variety of message formats.

#         Args:
#             messages: sequence of message representations.
#                   A message can be represented using the following formats:
#                   (1) BaseMessagePromptTemplate, (2) BaseMessage, (3) 2-tuple of
#                   (message type, template); e.g., ("human", "{user_input}"),
#                   (4) 2-tuple of (message class, template), (5) a string which is
#                   shorthand for ("human", template); e.g., "{user_input}".
#             template_format: format of the template. Defaults to "f-string".
#             input_variables: A list of the names of the variables whose values are
#                 required as inputs to the prompt.
#             optional_variables: A list of the names of the variables for placeholder
#             or MessagePlaceholder that are optional. These variables are auto inferred
#             from the prompt and user need not provide them.
#             partial_variables: A dictionary of the partial variables the prompt
#                 template carries. Partial variables populate the template so that you
#                 don't need to pass them in every time you call the prompt.
#             validate_template: Whether to validate the template.
#             input_types: A dictionary of the types of the variables the prompt template
#                 expects. If not provided, all variables are assumed to be strings.

#         Returns:
#             A chat prompt template.

#         Examples:

#             Instantiation from a list of message templates:

#             .. code-block:: python

#                 template = ChatPromptTemplate([
#                     ("human", "Hello, how are you?"),
#                     ("ai", "I'm doing well, thanks!"),
#                     ("human", "That's good to hear."),
#                 ])

#             Instantiation from mixed message formats:

#             .. code-block:: python

#                 template = ChatPromptTemplate([
#                     SystemMessage(content="hello"),
#                     ("human", "Hello, how are you?"),
#                 ])

#         """
#         _messages = [
#             _convert_to_message(message, template_format) for message in messages
#         ]

#         # Automatically infer input variables from messages
#         input_vars: set[str] = set()
#         optional_variables: set[str] = set()
#         partial_vars: dict[str, Any] = {}
#         for _message in _messages:
#             if isinstance(_message, MessagesPlaceholder) and _message.optional:
#                 partial_vars[_message.variable_name] = []
#                 optional_variables.add(_message.variable_name)
#             elif isinstance(
#                 _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
#             ):
#                 input_vars.update(_message.input_variables)

#         kwargs = {
#             "input_variables": sorted(input_vars),
#             "optional_variables": sorted(optional_variables),
#             "partial_variables": partial_vars,
#             **kwargs,
#         }
#         cast(type[ChatPromptTemplate], super()).__init__(messages=_messages, **kwargs)
    
    
#     # 基于 ChatPromptTemplate
#     def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
#         """Format the chat template into a list of finalized messages.

#         Args:
#             **kwargs: keyword arguments to use for filling in template variables
#                       in all the template messages in this chat template.

#         Returns:
#             list of formatted messages.
#         """
#         kwargs = self._merge_partial_and_user_variables(**kwargs)
#         result = []
#         for message_template in self.messages:
#             if isinstance(message_template, BaseMessage):
#                 result.extend([message_template])
#             elif isinstance(
#                 message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
#             ):
#                 message = message_template.format_messages(**kwargs)
#                 result.extend(message)
#             else:
#                 msg = f"Unexpected input: {message_template}"
#                 raise ValueError(msg)
#         return result
    
#     def _format(self, **kwargs: Any) -> ChatPromptValue:
#         content: List[dict] = kwargs.get("content", [])
#         messages = []  # 初始化为 ChatPromptValue 类型

#         for item in content:
#             media_type = item.get("type")
#             if media_type == "text":
#                 # 处理文本输入
#                 text_content = item.get("text")
#                 messages.append(HumanMessage(content=text_content))
#             elif media_type == "image":
#                 # 处理图像输入
#                 image_data = item.get("image_data")
#                 if isinstance(image_data, bytes):
#                     # 如果是二进制数据，转换为base64
#                     encoded_image = base64.b64encode(image_data).decode('utf-8')
#                     messages.append(HumanMessage(content=f"![Image](data:image/jpeg;base64,{encoded_image})"))
#                 elif isinstance(image_data, str) and os.path.isfile(image_data):
#                     # 如果是文件路径，读取文件并转换为base64
#                     with open(image_data, "rb") as img_file:
#                         encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
#                         messages.append(HumanMessage(content=f"![Image](data:image/jpeg;base64,{encoded_image})"))
#                 else:
#                     messages.append(HumanMessage(content="Unsupported image format."))
#             elif media_type == "audio":
#                 # 处理语音输入
#                 audio_data = item.get("audio_data")
#                 if isinstance(audio_data, bytes):
#                     # 如果是二进制数据，转换为base64
#                     encoded_audio = base64.b64encode(audio_data).decode('utf-8')
#                     messages.append(HumanMessage(content=f"[Audio](data:audio/wav;base64,{encoded_audio})"))
#                 elif isinstance(audio_data, str) and os.path.isfile(audio_data):
#                     # 如果是文件路径，读取文件并转换为base64
#                     with open(audio_data, "rb") as audio_file:
#                         encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
#                         messages.append(HumanMessage(content=f"[Audio](data:audio/wav;base64,{encoded_audio})"))
#                 else:
#                     messages.append(HumanMessage(content="Unsupported audio format."))
#             else:
#                 # 处理未知类型
#                 messages.append(HumanMessage(content="Unsupported media type."))

#         # 返回合并后的消息，确保是 ChatPromptValue 类型
#         return messages
    
#     def _format(self, **kwargs: Any) -> BaseMessage:
#         """Format the prompt template.

#         Args:
#             **kwargs: Keyword arguments to use for formatting.

#         Returns:
#             Formatted message.
#         """
              
#         content: list = []
#         for prompt in self.prompt:
#             inputs = {var: kwargs[var] for var in prompt.input_variables}
#             if isinstance(prompt, StringPromptTemplate):
#                 formatted: Union[str, ImageURL] = prompt.format(**inputs)
#                 content.append({"type": "text", "text": formatted})
#             if isinstance(prompt, StringPromptTemplate):
#                 formatted: Union[str, ImageURL] = prompt.format(**inputs)
#                 content.append({"media_type": "text", "media_data": formatted})
#         return HumanMessage(
#             content=content, additional_kwargs=self.additional_kwargs
#         )
            
#     def _create_template_from_message_type(self,
#         message_type: str,
#         template: Union[str, list],
#         template_format: PromptTemplateFormat = "f-string",
#     ) -> BaseMessagePromptTemplate:
#         """Create a message prompt template from a message type and template string.

#         Args:
#             message_type: str the type of the message template (e.g., "human", "ai", etc.)
#             template: str the template string.
#             template_format: format of the template. Defaults to "f-string".

#         Returns:
#             a message prompt template of the appropriate type.

#         Raises:
#             ValueError: If unexpected message type.
#         """
#         if message_type in ("human", "user"):
#             message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
#                 template, template_format=template_format
#             )
#             for tmpl in template:
#                 if isinstance(tmpl, dict) :
#                     if "text" in tmpl:
#                         text = tmpl["text"]
#                         prompt = PromptTemplate.from_template(
#                             text, template_format=template_format
#                         )
#                     if "media_type" in tmpl:
#                         media_type = tmpl["media_type"]
#                         prompt = PromptTemplate.from_template(
#                             media_type, template_format=template_format
#                         )
#                     if "media_data" in tmpl:
#                         media_data = tmpl["media_data"]
#                         prompt = PromptTemplate.from_template(
#                             media_data, template_format=template_format
#                         )
#                     self.prompt.append(prompt)
            
#         else:
#             msg = (
#                 f"Unexpected message type: {message_type}. Use one of 'human',"
#                 f" 'user', 'ai', 'assistant', or 'system'."
#             )
#             raise ValueError(msg)
#         return message

#     def _convert_to_message(
#         self,
#         message: MessageLikeRepresentation,
#         template_format: PromptTemplateFormat = "f-string",
#     ) -> Union[BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate]:
#         """Instantiate a message from a variety of message formats.

#         The message format can be one of the following:

#         - BaseMessagePromptTemplate
#         - BaseMessage
#         - 2-tuple of (role string, template); e.g., ("human", "{user_input}")
#         - 2-tuple of (message class, template)
#         - string: shorthand for ("human", template); e.g., "{user_input}"

#         Args:
#             message: a representation of a message in one of the supported formats.
#             template_format: format of the template. Defaults to "f-string".

#         Returns:
#             an instance of a message or a message template.

#         Raises:
#             ValueError: If unexpected message type.
#             ValueError: If 2-tuple does not have 2 elements.
#         """
#         if isinstance(message, tuple):
#             if len(message) != 2:
#                 msg = f"Expected 2-tuple of (role, template), got {message}"
#                 raise ValueError(msg)
#             message_type_str, template = message
#             if isinstance(message_type_str, str):
#                 _message = self._create_template_from_message_type(
#                     message_type_str, template, template_format=template_format
#                 )
#             else:
#                 _message = message_type_str(
#                     prompt=PromptTemplate.from_template(
#                         cast(str, template), template_format=template_format
#                     )
#                 )
#         else:
#             msg = f"Unsupported message type: {type(message)}"
#             raise NotImplementedError(msg)

#         return _message

multimodal_input_template = PromptTemplate(
    template='{"media":{"url":"{{url}}","path":"{{path}}","data":{{data}}},"text":"{{text}}"}',
    partial_variables={"text":None,"url":None,"path":None,"data":"null"},
    template_format="mustache"
)