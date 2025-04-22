from typing import (
    Any,
    List,
    Union,
    Sequence,
    cast,
    TypeVar,
    Optional,
    TypedDict
)

from langchain.prompts.chat import (
    MessageLikeRepresentation,
    BaseMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from langchain_core.prompt_values import ChatPromptValue, ImageURL, PromptValue
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage
from pydantic import (
    Field,
    PositiveInt,
    SkipValidation,
    model_validator,
)
from pathlib import Path
from langchain_core.prompts.chat import MessagesPlaceholder,ChatPromptTemplate,BaseChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompts.image import ImagePromptTemplate

MultiModalMessagePromptTemplateT = TypeVar(
    "MultiModalMessagePromptTemplateT", bound="MultiModalMessagePromptTemplate"
)

class _TextTemplateParam(TypedDict, total=False):
    text: Union[str, dict]

class _ImageTemplateParam(TypedDict, total=False):
    image: Union[str, dict]
    
class _AudioTemplateParam(TypedDict, total=False):
    audio: Union[str, dict]

class _VideoTemplateParam(TypedDict, total=False):
    video: Union[str, dict]
    
class MultiModalMessagePromptTemplate(BaseMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    prompt: Union[
        StringPromptTemplate, list[Union[StringPromptTemplate, ImagePromptTemplate]]
    ]
    """Prompt template."""
    additional_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the prompt template."""

    _msg_class: type[BaseMessage]  = HumanMessage

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    @classmethod
    def from_template(
        cls: type[MultiModalMessagePromptTemplateT],
        template: Union[str, list[Union[str, _TextTemplateParam, _ImageTemplateParam,_AudioTemplateParam,_VideoTemplateParam]]],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MultiModalMessagePromptTemplateT:
        """Create a class from a string template.

        Args:
            template: a template.
            template_format: format of the template.
                Options are: 'f-string', 'mustache', 'jinja2'. Defaults to "f-string".
            partial_variables: A dictionary of variables that can be used too partially.
                Defaults to None.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.

        Raises:
            ValueError: If the template is not a string or list of strings.
        """
        if isinstance(template, str):
            prompt: Union[StringPromptTemplate, list] = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, **kwargs)
        elif isinstance(template, list):
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "Partial variables are not supported for list of templates."
                raise ValueError(msg)
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or isinstance(tmpl, dict) and "text" in tmpl:
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_TextTemplateParam, tmpl)["text"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif isinstance(tmpl, str) or isinstance(tmpl, dict) and "image" in tmpl:
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_ImageTemplateParam, tmpl)["image"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif isinstance(tmpl, str) or isinstance(tmpl, dict) and "audio" in tmpl:
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_AudioTemplateParam, tmpl)["audio"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif isinstance(tmpl, str) or isinstance(tmpl, dict) and "video" in tmpl:
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast(_VideoTemplateParam, tmpl)["video"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                else:
                    msg = f"Invalid template: {tmpl}"
                    raise ValueError(msg)
            return cls(prompt=prompt, **kwargs)
        else:
            msg = f"Invalid template: {template}"
            raise ValueError(msg)

    @classmethod
    def from_template_file(
        cls: type[MultiModalMessagePromptTemplateT],
        template_file: Union[str, Path],
        input_variables: list[str],
        **kwargs: Any,
    ) -> MultiModalMessagePromptTemplateT:
        """Create a class from a template file.

        Args:
            template_file: path to a template file. String or Path.
            input_variables: list of input variables.
            **kwargs: keyword arguments to pass to the constructor.

        Returns:
            A new instance of this class.
        """
        with open(str(template_file)) as f:
            template = f.read()
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessages.
        """
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """
        Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        input_variables = [iv for prompt in prompts for iv in prompt.input_variables]
        return input_variables

    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        else:
            content: list = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = prompt.format(**inputs)
                    if "text" in inputs:
                        content.append({"type": "text", "text": formatted})
                    elif "image" in inputs:
                        content.append({"type": "image", "image": formatted})
                    elif "audio" in inputs:
                        content.append({"type": "audio", "audio": formatted})
                    elif "video" in inputs:
                        content.append({"type": "video", "video": formatted})
            return self._msg_class(
                content=content, additional_kwargs=self.additional_kwargs
            )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )
        else:
            content: list = []
            for prompt in self.prompt:
                inputs = {var: kwargs[var] for var in prompt.input_variables}
                if isinstance(prompt, StringPromptTemplate):
                    formatted: Union[str, ImageURL] = await prompt.aformat(**inputs)
                    content.append({"type": "text", "text": formatted})
                elif "image" in inputs:
                        content.append({"type": "image", "image": formatted})
                elif "audio" in inputs:
                    content.append({"type": "audio", "audio": formatted})
                elif "video" in inputs:
                    content.append({"type": "video", "video": formatted})
            return self._msg_class(
                content=content, additional_kwargs=self.additional_kwargs
            )

    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        # TODO: Handle partials
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"
    
def _create_template_from_message_type(
    message_type: str,
    template: Union[str, list],
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessagePromptTemplate:

    if message_type in ("human", "user"):
        message: BaseMessagePromptTemplate = MultiModalMessagePromptTemplate.from_template(
            template, template_format=template_format
        )
    elif message_type in ("ai", "assistant"):
        message = AIMessagePromptTemplate.from_template(
            cast(str, template), template_format=template_format
        )
    elif message_type == "system":
        message = SystemMessagePromptTemplate.from_template(
            cast(str, template), template_format=template_format
        )
    elif message_type == "placeholder":
        if isinstance(template, str):
            if template[0] != "{" or template[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {template}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        elif len(template) == 2 and isinstance(template[1], bool):
            var_name_wrapped, is_optional = template
            if not isinstance(var_name_wrapped, str):
                msg = (
                    "Expected variable name to be a string." f" Got: {var_name_wrapped}"
                )
                raise ValueError(msg)
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                msg = (
                    f"Invalid placeholder template: {var_name_wrapped}."
                    " Expected a variable name surrounded by curly braces."
                )
                raise ValueError(msg)
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
        else:
            msg = (
                "Unexpected arguments for placeholder message type."
                " Expected either a single string variable name"
                " or a list of [variable_name: str, is_optional: bool]."
                f" Got: {template}"
            )
            raise ValueError(msg)
    else:
        msg = (
            f"Unexpected message type: {message_type}. Use one of 'human',"
            f" 'user', 'ai', 'assistant', or 'system'."
        )
        raise ValueError(msg)
    return message


def _convert_to_message(
    message: MessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> Union[BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate]:
   
    if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        _message: Union[
            BaseMessage, BaseMessagePromptTemplate, BaseChatPromptTemplate
        ] = message
    elif isinstance(message, BaseMessage):
        _message = message
    elif isinstance(message, str):
        _message = _create_template_from_message_type(
            "human", message, template_format=template_format
        )
    elif isinstance(message, tuple):
        if len(message) != 2:
            msg = f"Expected 2-tuple of (role, template), got {message}"
            raise ValueError(msg)
        message_type_str, template = message
        if isinstance(message_type_str, str):
            _message = _create_template_from_message_type(
                message_type_str, template, template_format=template_format
            )
        else:
            _message = message_type_str(
                prompt=PromptTemplate.from_template(
                    cast(str, template), template_format=template_format
                )
            )
    else:
        msg = f"Unsupported message type: {type(message)}"
        raise NotImplementedError(msg)

    return _message


class MultiModalChatPromptTemplate(ChatPromptTemplate):
    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        _messages = [
            _convert_to_message(message, template_format) for message in messages
        ]

        # Automatically infer input variables from messages
        input_vars: set[str] = set()
        optional_variables: set[str] = set()
        partial_vars: dict[str, Any] = {}
        for _message in _messages:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
                optional_variables.add(_message.variable_name)
            elif isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        kwargs = {
            "input_variables": sorted(input_vars),
            "optional_variables": sorted(optional_variables),
            "partial_variables": partial_vars,
            **kwargs,
        }
        cast(type[ChatPromptTemplate], super()).__init__(messages=_messages, **kwargs)
