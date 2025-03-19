from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,format_document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage,ToolMessage
from langchain_core.runnables.utils import AddableDict
from langchain_core.runnables.base import Runnable,RunnableConfig
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
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
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser,BaseOutputParser
from agi.tasks.prompt import default_template,contextualize_q_template,doc_qa_template,cumstom_rag_default_template
from agi.tasks.retriever import KnowledgeManager
from agi.tasks.common import graph_parser,graph_input_format
import json
from datetime import datetime,timezone
from langchain.globals import set_debug
from collections import defaultdict
import validators
from agi.config import (
    LANGCHAIN_DB_PATH
)
from agi.tasks.retriever import create_retriever
from typing import (
    Any,
    Dict,
    Optional,
    Union
)
from langgraph.prebuilt.chat_agent_executor import AgentState
def is_valid_url(url):
    return validators.url(url)
import traceback
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

set_debug(False)

def debug_info(x : Any):
    return f"type:{type(x)}\nmessage:{x}"
    
# TODO 历史数据压缩 数据实体抽取
# TODO xpected `str` but got `dict` with value `{'type': 'text', 'text': ...], 'distances': [1.0]}]}` - serialized value may not be as expected
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", LANGCHAIN_DB_PATH)

# dict_input = True时，只能输入dict
# Input: dict or List[BaseMessage]
# Output: dict
def create_llm_with_history(runnable,debug=False,dict_input=True):
    def debug_print(x: Any) :
        log.debug(f"create_llm_with_history\n:{debug_info(x)}")
        return x

    debug_tool = RunnableLambda(debug_print)
    
    if debug:
        runnable = debug_tool | runnable
    
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

# 记录历史的聊天，单纯添加了模板
def create_chat_with_history(llm,debug=False):
    def debug_print(x: Any) :
        log.debug(f"create_chat_with_history\n:{debug_info(x)}")
        return x

    debug_tool = RunnableLambda(debug_print)
    
    runnable = (default_template | llm )
    if debug:
        runnable = (default_template | debug_tool | llm )
    
    return create_llm_with_history(runnable=runnable)

def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate=contextualize_q_template,
    debug=False
) -> RetrieverOutputLike:
    # The Runnable output is a list of Documents
    def debug_print(x: Any) :
        log.debug(f"create_history_aware_retriever\n:{debug_info(x)}")
        return x

    debug_tool = RunnableLambda(debug_print)
    
    if "text" not in prompt.input_variables:
        raise ValueError(
            "Expected `input` to be a prompt variable, "
            f"but got {prompt.input_variables}"
        )
    chain = (prompt | llm | StrOutputParser() | retriever)
    
    if debug:
        chain = (debug_tool | prompt | llm | StrOutputParser() | retriever)
    
    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["text"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        chain
    ).with_config(run_name="chat_retriever_chain")
    
    return retrieve_documents

# Input：dict
# Output: str
def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate = doc_qa_template,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
    debug=False
) -> Runnable[Dict[str, Any], Any]:
    
    def debug_print(x: Any) :
        log.debug(f"create_stuff_documents_chain\n:{debug_info(x)}")
        return x

    debug_tool = RunnableLambda(debug_print)
    
    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt)
            for doc in inputs[document_variable_name]
        )
    # 经过prompt 之后，变为了ChatPromptValue，需要转换为List[BaseMessage]
    def format_history_chain_input(x: Any):
        log.debug(f"format_history_chain_input:{x}")
        messages = x.to_messages()
        log.debug(f"format_history_chain_input out:{messages}")
        return messages
    
    chain = (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | format_history_chain_input
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")
    
    if debug:
        return debug_tool | chain
    return chain

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

    # 将文档按 source 聚合
    for doc in inputs["context"]:
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
    return citations

def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], str],
    debug=False
) -> Runnable:
    
    def debug_print(x: Any) :
        log.debug(f"create_retrieval_chain\n:{debug_info(x)}")
        return x

    debug_tool = RunnableLambda(debug_print)
    
    if not isinstance(retriever, BaseRetriever):
        retrieval_docs: Runnable[dict, RetrieverOutput] = retriever
    else:
        retrieval_docs = (lambda x: x["text"]) | retriever

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
            
        ).assign(answer=combine_docs_chain,citations=build_citations)
    ).with_config(run_name="retrieval_chain")

    if debug:
        return debug_tool | retrieval_chain
    return retrieval_chain

# chain:RunnableWithMessageHistory
# chain:insert_history
# chain:RunnableParallel<chat_history>
# chain:load_history
# chain:check_sync_or_async 
# chain:retrieval_chain 
# chain:RunnableAssign<answer> 
# chain:RunnableParallel<answer>
def create_chat_with_rag(km: KnowledgeManager,llm,debug=False,**kwargs):
    retrievers = create_retriever(km,**kwargs)
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retrievers, contextualize_q_template,debug=debug
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=debug)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain,debug=debug)
    return create_llm_with_history(runnable=retrieval_chain,debug=debug)

# 知识库名称（列表），可以作为参数传入
# TODO 无法和llm history 融合
def create_chat_with_custom_rag(
        km: KnowledgeManager,llm,
        debug=True,
        graph: bool = False
):
    # 1.引入新的prompt
    # 2.构建创建retrever的runnable,返回docs
    # 3.组合chain

    def debug_print(x: Any) :
        log.debug(f"create_chat_with_custom_rag\n:{debug_info(x)}")
        return x
    
    debug_tool = RunnableLambda(debug_print)
    
    def query_docs(inputs: dict) :
        log.debug(f"query_docs----{inputs}")
        collection_names = inputs.get("collection_names",None)
       
        collections = "all"
        if isinstance(collection_names,str):
            collections = json.loads(collection_names)
        retriever = create_retriever(km,collection_names=collections)
        return retriever.invoke(inputs.get("text",""))
    
    llm = create_llm_with_history(runnable=llm,debug=debug,dict_input=False)
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=debug)
     # 获取正确的输入格式
    begin = RunnableLambda(message_to_dict)
    # 获取合理的输出格式
    output = RunnableLambda(dict_to_ai_message)
    retrieval_docs = RunnableLambda(query_docs)
    
    retrieval_chain = (
        begin
        | RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
            
        ).assign(answer=combine_docs_chain,citations=build_citations)
        | output
    ).with_config(run_name="custom_rag_chain")
   
    if debug:
        retrieval_chain = debug_tool | retrieval_chain
    
    # 转换graph格式
    if graph:
        return retrieval_chain | graph_parser
    
    
    return retrieval_chain

# 支持网页检索的对答 chain
# DONE 无法和llm history 融合
def create_chat_with_websearch(km: KnowledgeManager,llm,debug=True,graph: bool = False):
    
    def web_search(inputs: dict) :
        log.debug(f"web_search----{inputs}")
        _,_,_,raw_docs = km.web_search(inputs.get("text",""))
        return raw_docs
    
    # 期望combine_docs_chain 能够存储历史
    llm = create_llm_with_history(runnable=llm,debug=debug,dict_input=False)
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=debug)
    # 获取正确的输入格式
    begin = RunnableLambda(message_to_dict)
    # 获取合理的输出格式
    output = RunnableLambda(dict_to_ai_message)

    web_search_runable = RunnableLambda(web_search)
    web_search_chain = (
        begin
        | RunnablePassthrough.assign(
            context=web_search_runable.with_config(run_name="web_search_runable"),
        )
        .assign(answer=combine_docs_chain,citations=build_citations)
        | output
    ).with_config(run_name="custom_rag_chain")
    

    # 转换graph格式
    if graph:
        return web_search_chain | graph_parser
    
    return web_search_chain

# 独立的web检索chain
# Input: AgentState
# Output: AgentState
def create_websearch_for_graph(km: KnowledgeManager):
    def web_search(input: dict,config: RunnableConfig) :
        tenant = config.get("configurable", {}).get("user_id", None)
        _,_,_,raw_docs = km.web_search(input.get("text"),tenant=tenant)
        log.debug(f"web_search---{raw_docs}")
        return raw_docs
    
    web_search_runable = RunnableLambda(web_search)
    web_search_chain = (
        start_runnable
        | RunnablePassthrough.assign(
            context=web_search_runable.with_config(run_name="web_search_runable"),
        ).assign(citations=build_citations)
        | tool_output_runnable
        | graph_parser
    ).with_config(run_name="web_search_chain")
    
    return web_search_chain

# 独立的文档检索chain
# Input: AgentState
# Output: AgentState
def create_rag_for_graph(km: KnowledgeManager):
    def query_docs(inputs: dict,config: RunnableConfig) :
    # def query_docs(inputs: dict) :
        log.debug(f"query_docs----{inputs}")
        collection_names = inputs.get("collection_names",None)        
        collections = "all"
        if isinstance(collection_names,str):
            collections = json.loads(collection_names)
        tenant = config.get("configurable", {}).get("user_id", None)
        retriever = km.get_retriever(collection_names=collections,tenant=tenant)
        docs = retriever.invoke(inputs.get("text",""))
        log.debug(f"query_docs----{docs}")
        return docs
    
    retrieval_docs = RunnableLambda(query_docs)
    rag_runable = (
        start_runnable
        | RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieval_docs"),
        ).assign(citations=build_citations)
        | tool_output_runnable
        | graph_parser
    ).with_config(run_name="rag_runable")
    
    return rag_runable

# 原子化的文档聊天chain
# Input: AgentState
# OutPut: AgentState
def create_docchain_for_graph(llm):
    llm = create_llm_with_history(runnable=llm,debug=False,dict_input=False)
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=False)

    combine_docs_chain = (
        start_runnable
        | RunnablePassthrough.assign(answer=combine_docs_chain)
        | ai_output_runnable
        | graph_parser
    )
    return combine_docs_chain

 # 将输出的字典格式转换为BaseMessage 或者 graph的格式
def dict_to_ai_message(output: dict):
    content = output.get('answer', '')
    if isinstance(content,dict):
        content = [content]
    log.debug(f"dict_to_ai_message---{output}")
    ai = AIMessage(
        content=content,
        additional_kwargs={
            'context': output.get('context', ''),
            'citations': output.get('citations', [])
        }
    )
    log.debug(f"dict_to_ai_message---{ai}")
    return ai
# 获取合理的输出格式
ai_output_runnable = RunnableLambda(dict_to_ai_message)

def dict_to_tool_message(output: dict):
    log.debug(f"dict_to_tool_message---{output}")
    ai = ToolMessage(
        tool_call_id = "web or rag",
        content=output.get('text', ''),
        additional_kwargs={
            'context': output.get('context', ''),
            'citations': output.get('citations', [])
        }
    )
    log.debug(f"dict_to_tool_message-ret--{ai}")
    return ai
# 获取合理的输出格式
tool_output_runnable = RunnableLambda(dict_to_tool_message)

# 用于将各种格式的输入，转换为dict格式，供chain使用
# 支持将AgentState 等消息转换为dict
# TODO 为什么有的消息的content被改为了dict？
def message_to_dict(message: Union[list,HumanMessage,ToolMessage,dict,AgentState]):
    # 若是graph，则从state中抽取消息
    # AgentState 是typedict ，不支持类型检查
    def correct_message_content(message):
        if isinstance(message.content,dict):
            message.content = [message.content]
        return message
    try:
        
        log.debug(f"message_to_dict--message---{message}")
        if "messages" in message:
            message = graph_input_format(message)
            # last_message = correct_message_content(message[-1])
            last_message = message[-1]
            log.debug(f"message_to_dict--last_message---{last_message}")
            if isinstance(last_message,HumanMessage) or isinstance(last_message,ToolMessage):
                return {
                    "text": last_message.content,
                    "language": "chinese",
                    "collection_names": last_message.additional_kwargs.get("collection_names",None),
                    "context": last_message.additional_kwargs.get("context",None),
                    "citations": last_message.additional_kwargs.get("citations",None),
                }
        elif isinstance(message,dict):
            return message
        elif isinstance(message,HumanMessage) or isinstance(message,ToolMessage):
            message.additional_kwargs.get("collection_names",None)
            message = correct_message_content(message)
            return {
                "text": message.content,
                "language": "chinese",
                "collection_names": message.additional_kwargs.get("collection_names",None),
                "context": message.additional_kwargs.get("context",None),
                "citations": message.additional_kwargs.get("citations",None),
            }
        elif isinstance(message,list) and len(message) > 0:
            last_message = correct_message_content(message[-1])
            if isinstance(last_message,HumanMessage) or isinstance(message,ToolMessage):
                return {
                    "text": last_message.content,
                    "language": "chinese",
                    "collection_names": last_message.additional_kwargs.get("collection_names",None),
                    "context": message.additional_kwargs.get("context",None),
                    "citations": message.additional_kwargs.get("citations",None),
                }
    except Exception as e:
        log.error(e)
        print(traceback.format_exc())
    
    return {
        "text": "user dont say anything",
        "language": "chinese",
    } 
        
# 获取正确的输入格式
start_runnable = RunnableLambda(message_to_dict)

class LangchainApp:
    
    llm: BaseChatModel
    runnable: Runnable
    with_message_history: RunnableWithMessageHistory
    db_path: str
    retrievers: EnsembleRetriever

    def __init__(self,llm,db_path="sqlite:///langchain.db",retrievers=None):
        self.db_path = db_path
        self.llm =llm
        
        self.retrievers = retrievers
        self.history_aware_retriever = None
        
        if retrievers is not None:
            
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retrievers, contextualize_q_template
            )
           
            question_answer_chain = create_stuff_documents_chain(self.llm, doc_qa_template)

            # self.runnable = question_answer_chain
            self.runnable = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        else:
            self.runnable = default_template | self.llm 

        self.with_message_history = LangchainApp.create_llm_with_history(self.runnable)
        
    @staticmethod
    def create_llm_with_history(runnable=None):
        
        return RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
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
        
    def get_session_history(self,user_id: str, conversation_id: str):
        return SQLChatMessageHistory(f"{user_id}--{conversation_id}", self.db_path)
    
    def query_doc(self,input):
        return self.history_aware_retriever.invoke({"input":input})
        
    def stream(self,input: str,language="chinese",user_id="",conversation_id=""):
        if conversation_id == "":
            import uuid
            conversation_id = str(uuid.uuid4())
            
        input_template = {"language": language, "input": input}
        config = {"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        context = None
        if self.history_aware_retriever:
            context = self.history_aware_retriever.invoke({"input": input})
            log.info("qury context: %s",context)
            input_template = {"language": language, "input": input,"context":context}
        
        response = self.with_message_history.stream(input_template,config)
        return self._process_stream(context,response)
        
    
    def _process_stream(self, context,response):
        if context:
            yield context
        if response:
            for item in response:
                # log.debug("_process_stream:",item)
                if item is not None:
                    if isinstance(item, AddableDict):
                        content = item.get('answer')
                                                
                        if content is not None:
                            processed_item = AIMessage(content=content)
                            if content == "":
                                processed_item.response_metadata = {'finish_reason': "stop"}
                            yield processed_item  # Yield only if processed_item is valid
                            
                    elif isinstance(item, AIMessage):
                        processed_item = item
                        yield processed_item
                    elif isinstance(item, str):
                        processed_item = AIMessage(content=item)
                        if item == "":
                            processed_item.response_metadata = {'finish_reason': "stop"}
                        yield processed_item  # Yield only if processed_item is valid
                            
                    
    def invoke(self,input: str,language="chinese",user_id="",conversation_id=""):
        if conversation_id == "":
            import uuid
            conversation_id = str(uuid.uuid4())

        input_template = {"language": language, "input": input}
        config = {"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
        if self.history_aware_retriever:
            context = self.history_aware_retriever.invoke({"input": input})
            input_template = {"language": language, "input": input,"context":context}

        response = self.with_message_history.invoke(input_template,config)
        log.debug(f"invoke:{response}")
        if isinstance(response,dict):
            content = response['answer']
            context = response.get('context')
            response = AIMessage(content=content)
            response.response_metadata = {"context":context}
        return response
    

    def citations(self, relevant_docs, contexts):
        # 初始化 citations 和 citamap
        citations = {"citations": []}
        citamap = defaultdict(lambda: defaultdict(dict))  # 使用 defaultdict 来自动初始化嵌套字典
        log.debug(f"context:{contexts}")
        
        def build_citations(context,doc,filename):
            # 如果匹配，更新 citamap
            citation = citamap[context['source'].get("collection_name")].get(filename)
            if citation:  # 如果已经有值，则追加
                citation["document"].append(doc.page_content)
                citation["metadata"].append(doc.metadata)
            else:  # 否则初始化
                context['source']["name"] = filename
                citamap[context.get("collection_name")][filename] = {
                    "source": context['source'],
                    "document": [doc.page_content],
                    "metadata": [doc.metadata],
                }
        # 遍历每个文档和上下文
        for doc in relevant_docs:
            for c in contexts:
                try:
                    if doc.metadata:
                        # 匹配文档和上下文的 collection_name 和 filename/source
                        doc_filename = doc.metadata.get("filename") or doc.metadata.get("source")
                        if not is_valid_url(doc_filename):
                            uid,doc_filename = doc_filename.split('_', 1)
                        if c['source'].get("type") == "collection":
                            for collection_name in c['source'].get("collection_names"):
                                if collection_name == doc.metadata.get("collection_name"):
                                    build_citations(c,doc,doc_filename)
                        else:
                            if doc.metadata.get("collection_name") == c['source'].get("collection_name"):
                                context_filename = c['source'].get("filename") 
                                urls = c['source'].get("urls") 
                                if doc_filename == context_filename or doc_filename in urls:
                                    build_citations(c,doc,doc_filename)
                except Exception as e:
                    log.debug(f"Error processing document {doc.metadata}: {e}")

        # 构建 citations 列表
        for collection, files in citamap.items():
            for filename, file_data in files.items():
                citations["citations"].append(file_data)

        return citations
                
    def wrap_citation(self,item):
        return f"{item}\n"
        
    def ollama(self,input: str,user_id="",conversation_id="",**kwargs):
        response = self.stream(input=input,user_id=user_id,conversation_id=conversation_id)
        content = None
        message_data = None
 
        is_done = False
        finish_reason = None
        for item in response:
            # log.debug(item,type(item))
            if isinstance(item,list):
                if item:
                    item = self.citations(item,kwargs.get("contexts"))
                    yield self.wrap_citation(json.dumps(item))
            
            elif isinstance(item, AIMessage):
            # 从每个 item 中提取 'content'
                content = item.content
                if item.response_metadata:
                    is_done = True
                    finish_reason = item.response_metadata['finish_reason']
                
                utc_now = datetime.now(timezone.utc)
                utc_now_str = utc_now.isoformat() + 'Z'
                message_data = {
                    "model": self.model,
                    "created_at": utc_now_str,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "done": is_done,
                    "done_reason": finish_reason,
                    
                }
                
                yield json.dumps(message_data) + "\n"  # 添加换行符
    
    def __call__(self,input: str,user_id="",conversation_id=""):
        response = self.invoke(input=input,user_id=user_id,conversation_id=conversation_id)
        return response

# if __name__ == "__main__":
#     app = LangchainApp()
#     stream_generator = app.ollama("hello")
#     # 遍历生成器
#     for response in stream_generator:
#         log.debug(response)
#     # ret = app.embed_query("我爱北京天安门")
#     # log.debug(ret)