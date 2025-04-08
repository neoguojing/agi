from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,format_document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage,ToolMessage,trim_messages
from langchain_core.runnables.utils import AddableDict
from langchain_core.runnables.base import Runnable,RunnableConfig
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langgraph.prebuilt.chat_agent_executor import AgentState

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)
from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput
)
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    DEFAULT_DOCUMENT_PROMPT,
    _validate_prompt,
)
from langchain_core.output_parsers import StrOutputParser,BaseOutputParser
from agi.tasks.prompt import get_last_message_text,doc_qa_template,docqa_modify_state_messages_runnable,default_modify_state_messages_runnable
from agi.tasks.retriever import KnowledgeManager
from agi.tasks.utils import graph_response_format_runnable
import json
from datetime import datetime,timezone
from langchain.globals import set_debug
from collections import defaultdict
import validators
from agi.config import (
    LANGCHAIN_DB_PATH,
    AGI_DEBUG,
)
from typing import (
    Any,
    Dict,
    Optional,
    Union
)

def is_valid_url(url):
    return validators.url(url)
import traceback
import logging
import inspect
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

set_debug(False)

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

# 裁剪历史消息
trimmer = trim_messages(
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    token_counter=len,
    # When token_counter=len, each message
    # will be counted as a single token.
    # Remember to adjust for your use case
    # 保存30条
    max_tokens=30,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    start_on="human",
    # Most chat models expect that chat history ends with either:
    # (1) a HumanMessage or
    # (2) a ToolMessage
    # end_on=("human", "tool"),
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
)

# TODO 历史数据压缩 数据实体抽取
# TODO xpected `str` but got `dict` with value `{'type': 'text', 'text': ...], 'distances': [1.0]}]}` - serialized value may not be as expected
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", LANGCHAIN_DB_PATH)

# dict_input = True时，只能输入dict
# Input: List[BaseMessage]
# Output: AIMessage
# 该工具不能单独使用
def create_llm_with_history(runnable,dict_input=False):
    # 支持历史消息裁剪
    runnable = debug_tool | trimmer | runnable

    input_key = "text"
    history_key = "chat_history"
    if not dict_input:
        input_key = None
        history_key = None
        
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="",
                is_shared=True,
            )
        ],
    )

# Input：AgentState
# Output: AgentState
# llm转换为基于历史的对话模式
# 若context不存在，则直接转到chat
# context存在，则转到 doc chain
def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate = doc_qa_template,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
) -> Runnable[Dict[str, Any], Any]:
    
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: AgentState) -> str:
        # 1. 从输入数据中获取需要处理的文档列表
        documents = inputs["docs"]

        # 2. 创建一个空列表用于存储格式化后的文档内容
        formatted_documents = []

        # 3. 遍历每个文档进行格式化处理
        for doc in documents:
            # 对单个文档应用格式化模板
            formatted_doc = format_document(doc, _document_prompt)
            # 将处理后的文档添加到列表
            formatted_documents.append(formatted_doc)

        # 4. 用指定的分隔符连接所有格式化后的文档
        combined_documents = document_separator.join(formatted_documents)

        # 最终结果保存在 combined_documents 变量中
        return combined_documents
    
    llm_with_history = create_llm_with_history(runnable=llm,dict_input=False)

    doc_chain = (
        RunnablePassthrough.assign(context=format_docs).with_config(run_name="format_inputs")
        | docqa_modify_state_messages_runnable
        | llm_with_history
    ).with_config(run_name="stuff_documents_chain")

    target_chain = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("docs", False),
            # If no docs, then we just pass input to llm
            default_modify_state_messages_runnable | llm_with_history
        ),
        # If docs, then we pass inputs to tag chain
        doc_chain
    ).with_config(run_name="stuff_documents_with_branch")

    return debug_tool | target_chain | graph_response_format_runnable



# 独立的web检索chain
# Input: AgentState
# Output: AgentState
def create_websearch(km: KnowledgeManager):
    def web_search(input: AgentState,config: RunnableConfig):
        tenant = config.get("configurable", {}).get("user_id", None)
        text = get_last_message_text(input)
        _,_,_,raw_docs = km.web_search(text,tenant=tenant)
        log.debug(f"web_search---{raw_docs}")
        return raw_docs
    
    web_search_runable = RunnableLambda(web_search)
    web_search_chain = RunnablePassthrough.assign(
            docs=web_search_runable.with_config(run_name="web_search_runable"),
    ).assign(citations=build_citations)
    
    return debug_tool | web_search_chain

# 独立的文档检索chain
# Input: AgentState
# Output: AgentState
# TODO 在未获取到知识库，或者未检索到相关文档的情况下，直接交给大模型型回答
def create_rag(km: KnowledgeManager):
    def query_docs(input: AgentState,config: RunnableConfig):
        log.debug(f"query_docs----{input}")
        # collection_names 位None，则默认使用 all进行检索
        collection_names = input.get("collection_names",None)        
        collections = "all"
        if isinstance(collection_names,str):
            collections = json.loads(collection_names)
        elif isinstance(collection_names,list):
            collections = collection_names
            
        tenant = config.get("configurable", {}).get("user_id", None)
        retriever = km.get_retriever(collection_names=collections,tenant=tenant)
        if retriever:
            text = get_last_message_text(input)
            docs = retriever.invoke(text)
            log.info(f"relative docks:{docs}")
            return docs
        return []
    
    retrieval_docs = RunnableLambda(query_docs)
    rag_runable = RunnablePassthrough.assign(
            docs=retrieval_docs.with_config(run_name="retrieval_docs"),
        ).assign(citations=build_citations)
    
    return debug_tool | rag_runable

# Input: AgentState
# Output: AgentState
# chat without history
def create_chat(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{text}")
        ]
    )

    def modify_state_messages(state: AgentState):
        text = get_last_message_text(state)
        return prompt.invoke({"text": text}).to_messages()
    
    input_format = RunnableLambda(modify_state_messages)
    
    chat = debug_tool | input_format | llm | graph_response_format_runnable

    return chat

'''
{
            "source": {
                "id": "3242bbd4-4a09-47d0-a704-dcbd5d665774",
                "user_id": "50e5febb-e4d7-4caa-9965-751160245ab6",
                "name": "test",
                "description": "个人文档",
                "meta": null,
                "access_control": null,
                "created_at": 1741763827,
                "updated_at": 1741775216,
                "user": {
                    "id": "50e5febb-e4d7-4caa-9965-751160245ab6",
                },
                "files": [
                    {
                        "id": "eb495463-9977-45d6-abd5-50a6365cacac",
                        "meta": {
                            "name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                            "content_type": "application/pdf",
                            "size": 83703,
                            "data": {},
                            "collection_name": "3242bbd4-4a09-47d0-a704-dcbd5d665774"
                        },
                        "created_at": 1741775216,
                        "updated_at": 1741775216
                    }
                ],
                "type": "collection"
            },
            "document": [""            ],
            "metadata": [
                {
                    "author": "China Tax",
                    "created_by": "50e5febb-e4d7-4caa-9965-751160245ab6",
                    "creationdate": "D:20240912212111",
                    "creator": "Suwell",
                    "embedding_config": "{\"engine\": \"\", \"model\": \"sentence-transformers/all-MiniLM-L6-v2\"}",
                    "file_id": "eb495463-9977-45d6-abd5-50a6365cacac",
                    "hash": "3e11c6cbdad114f4807614b25e6e94a83378905f091130c277c68178e0e327df",
                    "moddate": "D:20240912212111",
                    "name": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                    "ofd2pdflib": "ofd2pdflib/2.2.24.0111.1407",
                    "page": 0,
                    "page_label": "1",
                    "producer": "Suwell OFD convertor",
                    "source": "上海椒客多餐饮服务有限公司_发票金额357.00元.pdf",
                    "start_index": 0,
                    "total_pages": 1
                }
            ],
            "distances": [
                0.6697336820511759
            ]
        }
'''
def build_citations(inputs: dict):
    citations = []
    # 使用 defaultdict 来根据 source 聚合文档
    source_dict = defaultdict(list)
    try:
        # 将文档按 source 聚合
        for doc in inputs["docs"]:
            # 使用llm extract 提取内容时，与输入无关会返回NO_OUTPUT
            if "NO_OUTPUT" in doc.page_content:
                continue
            
            source_type = ""
            if doc.metadata.get('filename'):
                source_type = "file"
            elif doc.metadata.get('link'):
                source_type = "web"
            elif doc.metadata.get('source'):
                source_type = "collection"

            source = doc.metadata.get('filename') or doc.metadata.get('link') or doc.metadata.get('source')
            source_dict[source].append(doc)

        # 对每个 source 下的文档进行排序，并整理成需要的格式
        for source, docs in source_dict.items():
            # 按照 start_index 排序（假设页面顺序可以通过 start_index 排序）
            sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get('page', 0))
            
            # 提取排序后的 document 内容
            document_contents = [doc.page_content for doc in sorted_docs]
            
            # 提取 metadata
            metadata = [doc.metadata for doc in sorted_docs]
            distances = [1.0] * len(sorted_docs)
            citations.append({
                "source": {"id":source,"name":source},
                "document": document_contents,
                "metadata": metadata,
                "distances": distances
            })
        log.debug(f"build_citations----{citations}")
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())
        
    return citations

