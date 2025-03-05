from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,format_document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langchain_core.runnables.utils import AddableDict
from langchain_core.runnables.base import Runnable
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
def is_valid_url(url):
    return validators.url(url)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

set_debug(False)

def debug_info(x : Any):
    return f"type:{type(x)}\nmessage:{x}"
    
# TODO 历史数据压缩 数据实体抽取
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", LANGCHAIN_DB_PATH)

def create_llm_with_history(runnable,debug=False):
    def debug_print(x: Any) :
        print("create_llm_with_history\n:",debug_info(x))
        return x

    debug_tool = RunnableLambda(debug_print)
    
    if debug:
        runnable = debug_tool | runnable
        
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="text",
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
        
def create_chat_with_history(llm,debug=False):
    def debug_print(x: Any) :
        print("create_chat_with_history\n:",debug_info(x))
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
        print("create_history_aware_retriever\n:",debug_info(x))
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
        print("create_stuff_documents_chain\n:",debug_info(x))
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
            
    chain = (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")
    
    if debug:
        return debug_tool | chain
    return chain

def build_citations(inputs: dict):
    citations = []
    # 使用 defaultdict 来根据 source 聚合文档
    source_dict = defaultdict(list)

    # 将文档按 source 聚合
    for doc in inputs["context"]:
        source = doc.metadata.get('filename') or doc.metadata.get('link') or doc.metadata.get('source')
        source_dict[source].append(doc)

    # 对每个 source 下的文档进行排序，并整理成需要的格式
    for source, docs in source_dict.items():
        # 按照 start_index 排序（假设页面顺序可以通过 start_index 排序）
        sorted_docs = sorted(docs, key=lambda doc: doc.metadata.get('page', 0))
        
        # 聚合 metadata 中的 page 字段，去重后存为列表
        pages = list({doc.metadata.get('page') for doc in sorted_docs})  # 去重并转换成列表

        # 提取排序后的 document 内容
        document_contents = [doc.page_content for doc in sorted_docs]
        
        # 提取 metadata，假设只取第一个文档的 metadata 信息
        metadata = sorted_docs[0].metadata if sorted_docs else {}
        metadata['pages'] = pages
        citations.append({
            "source": source,
            "document": document_contents,
            "metadata": metadata,
        })
    
    return citations

def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], str],
    debug=False
) -> Runnable:
    
    def debug_print(x: Any) :
        print("create_retrieval_chain\n:",debug_info(x))
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

# 1. 知识库名称（列表），可以作为参数传入
def create_chat_with_custom_rag(
        km: KnowledgeManager,llm,
        prompt: BasePromptTemplate=cumstom_rag_default_template,
        debug=True
):
    # 1.引入新的prompt
    # 2.构建创建retrever的runnable,返回docs
    # 3.组合chain

    def debug_print(x: Any) :
        print("create_chat_with_custom_rag\n:",debug_info(x))
        return x
    
    debug_tool = RunnableLambda(debug_print)
    
    def query_docs(inputs: dict) :
        print("query_docs----",inputs)
        collection_names = inputs.get("collection_names",None)
        if collection_names is None:
            return []
        
        collections = "all"
        if isinstance(collection_names,str):
            collections = json.loads(collection_names)
        retriever = create_retriever(km,collection_names=collections)
        return retriever.invoke(inputs.get("text",""))
    
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=debug)

    retrieval_docs = RunnableLambda(query_docs)
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
            
        ).assign(answer=combine_docs_chain,citations=build_citations)
    ).with_config(run_name="custom_rag_chain")
    # retrieval_chain = prompt | retrieval_chain
    if debug:
        retrieval_chain = debug_tool | retrieval_chain
    
    return create_llm_with_history(runnable=retrieval_chain,debug=debug)

 # 创建将字典转换为 AIMessage 的函数
def dict_to_ai_message(output: dict) -> AIMessage:
    return AIMessage(
        content=output.get('answer', ''),
        additional_kwargs={
            'context': output.get('context', ''),
            'citations': output.get('citations', [])
        }
    )
    
def message_to_dict(message: Union[list,HumanMessage,dict]):
    if isinstance(message,dict):
        return message
    elif isinstance(message,HumanMessage):
        return {
            "text": message.content,
            "language": "chinese",
        }
    else:
        last_message = message[-1]
        return {
            "text": last_message.content,
            "language": "chinese",
        }
# 支持网页检索的对答 chain
def create_chat_with_websearch(km: KnowledgeManager,llm,debug=False):
    
    def web_search(inputs: dict) :
        _,_,_,raw_docs = km.web_search(inputs.get("text",""))
        return raw_docs
    
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template,debug=debug)

    web_search_runable = RunnableLambda(web_search)
    web_search_chain = (
        RunnablePassthrough.assign(
            context=web_search_runable.with_config(run_name="web_search_runable"),
        )
        .assign(answer=combine_docs_chain,citations=build_citations)
    ).with_config(run_name="custom_rag_chain")
    
    begin = RunnableLambda(message_to_dict)
    output = RunnableLambda(dict_to_ai_message)
    runnable = begin | web_search_chain | output
    return create_llm_with_history(runnable=runnable,debug=debug)

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
                # print("_process_stream:",item)
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
        print("invoke:",response)
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
        print("relevant_docs num:", len(relevant_docs))
        print("context:", contexts)
        
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
                    print(f"Error processing document {doc.metadata}: {e}")

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
            # print(item,type(item))
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
#         print(response)
#     # ret = app.embed_query("我爱北京天安门")
#     # print(ret)