from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,format_document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages.ai import AIMessage,AIMessageChunk
from langchain_core.runnables.utils import AddableDict
from langchain_core.runnables.base import Runnable
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch
)
from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput
)

from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from agi.tasks.prompt import default_template,contextualize_q_template,doc_qa_template
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

def is_valid_url(url):
    return validators.url(url)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

set_debug(False)

# TODO 历史数据压缩 数据实体抽取
def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", LANGCHAIN_DB_PATH)

def create_llm_with_history(runnable):
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
        
def create_chat_with_history(llm):
    runnable = default_template | llm 
    return create_llm_with_history(runnable=runnable)

def create_history_aware_retriever(
    llm: LanguageModelLike,
    retriever: RetrieverLike,
    prompt: BasePromptTemplate,
) -> RetrieverOutputLike:
    # The Runnable output is a list of Documents
    
    if "text" not in prompt.input_variables:
        raise ValueError(
            "Expected `input` to be a prompt variable, "
            f"but got {prompt.input_variables}"
        )

    retrieve_documents: RetrieverOutputLike = RunnableBranch(
        (
            # Both empty string and empty list evaluate to False
            lambda x: not x.get("chat_history", False),
            # If no chat history, then we just pass input to retriever
            (lambda x: x["text"]) | retriever,
        ),
        # If chat history, then we pass inputs to LLM chain, then to retriever
        prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")
    
    return retrieve_documents
    
def create_chat_with_rag(km: KnowledgeManager,llm,**kwargs):
    retrievers = create_retriever(km,**kwargs)
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retrievers, contextualize_q_template
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm, doc_qa_template)
    
    retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    # retrieval_docs = (lambda x: x["text"]) | history_aware_retriever
    # retrieval_chain = (
    #     RunnablePassthrough.assign(
    #         context=retrieval_docs.with_config(run_name="retrieve_documents"),
    #     ).assign(answer=combine_docs_chain)
    # ).with_config(run_name="retrieval_chain")

    # return retrieval_chain
    return create_llm_with_history(runnable=retrieval_chain)

    
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