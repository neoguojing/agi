from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    OutlookMessageLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredXMLLoader,
    JSONLoader,
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from typing import Any,List,Dict,Iterator, Optional, Sequence, Union
import validators
import socket
import urllib.parse
from langchain.retrievers.document_compressors import DocumentCompressorPipeline,EmbeddingsFilter,LLMListwiseRerank,LLMChainFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from agi.tasks.vectore_store import CollectionManager
from agi.tasks.prompt import DEFAULT_SEARCH_PROMPT,rag_filter_template
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper,SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import AsyncChromiumLoader,AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.retrievers.web_research import QuestionListOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
import time
from uuid import uuid4
from agi.config import (
    CACHE_DIR
)
from datetime import datetime
from exa_py import Exa
from agi.config import EXA_API_KEY
import traceback

from agi.config import log


from enum import Enum

class SourceType(Enum):
    YOUTUBE = "youtube"
    WEB = "web"
    FILE = "file"
    SEARCH_RESULT = "search_result"

class FilterType(Enum):
    LLM_FILTER = "llm_chain_filter"
    LLM_RERANK = "llm_listwise_rerank"
    RELEVANT_FILTER = "embeddings_filter"
    LLM_EXTRACT = "llm_extract"

class SimAlgoType(Enum):
    MMR = "mmr"
    SST = "similarity_score_threshold"


class KnowledgeManager:
    def __init__(self, data_path,llm,embedding):
        self.embedding = embedding

        self.llm =llm
        self.search_chain = DEFAULT_SEARCH_PROMPT | self.llm | QuestionListOutputParser()
        self.search_engines = {}

        # Only add Exa if EXA_API_KEY is not empty
        if EXA_API_KEY:
            self.search_engines["Exa"] = Exa(EXA_API_KEY)

        # Always add DuckDuckGoSearch
        self.search_engines["DuckDuckGoSearch"] = DuckDuckGoSearchAPIWrapper(region="wt-wt", safesearch="moderate", time="d", max_results=3, source="news")
                
        self.collection_manager = CollectionManager(data_path,embedding)

    def list_documets(self,collection_name, tenant=None):
        return self.collection_manager.get_documents(collection_name,tenant=tenant)
    
    def list_collections(self, tenant=None):
        return self.collection_manager.list_collections(tenant=tenant)
    
    def store(self,collection_name: str, source: Union[str, List[str],List[dict]],tenant=None, source_type: SourceType=SourceType.FILE, **kwargs):
        """
        存储 URL 或文件，支持单个或多个 source。
        
        参数:
        - collection_name: 存储的集合名称
        - source: 如果是 URL，传入字符串或列表；如果是文件，传入文件路径或文件路径列表
        - source_type: 数据源类型，支持 SourceType.YOUTUBE, SourceType.WEB, SourceType.FILE
        - file_info: 一个包含文件相关信息的字典，只有当 source_type 是 SourceType.FILE 时使用。
                    字典中可以包含:
                    - "file_name": 文件名
                    - "content_type": 文件的内容类型
        """
         # 处理单个或多个 source
        if isinstance(source, str):
            sources = [source]  # 如果是字符串，转为列表
        elif isinstance(source, list):
            sources = source  # 如果是列表，直接使用
        else:
            raise ValueError("Source must be a string or a list of strings.")
    
        loader = None
        known_type = None
        raw_docs = []
        try:
            exist_sources = self.collection_manager.get_sources(collection_name,tenant=tenant)
            store = self.collection_manager.get_vector_store(collection_name,tenant=tenant)
            for source in sources:
                if source in exist_sources:
                    continue
                
                if source_type == SourceType.YOUTUBE:
                    loader = self.get_youtube_loader(source)
                elif source_type == SourceType.WEB:
                    loader = self.get_web_loader(source)
                elif source_type == SourceType.SEARCH_RESULT:
                    raw_docs.append(
                        Document(
                            page_content=f"{source.get('date')}\n{source.get('title')}\n{source.get('snippet')}",
                            metadata={"source":source.get("source"),"link":source.get("link")},
                        )
                    )

                else:
                    file_name = kwargs.get('filename')
                    content_type = kwargs.get('content_type')
                    if file_name is None:
                        raise ValueError("File name is required for file storage.")
                    loader,known_type = self.get_loader(file_name,source,content_type)
                
                log.info("start load file---------")
                if loader:
                    raw_docs = loader.load()

                for doc in raw_docs:
                    # doc.page_content = doc.page_content.replace("\n", " ")
                    # doc.page_content = doc.page_content.replace("\t", "")
                    doc.metadata["collection_name"] = collection_name
                    doc.metadata["type"] = source_type.value
                    doc.metadata["timestamp"] = str(time.time())
                    if doc.metadata.get("source") is None:
                        doc.metadata["source"] = source
                    doc.metadata = {**doc.metadata, **kwargs}

                log.info(f"loader file count:{len(raw_docs)}")
                docs = self.split_documents(raw_docs)
                log.info(f"splited documents count:{len(docs)}")
                uuids = [str(uuid4()) for _ in range(len(docs))]
                if len(docs) > 0:
                    store.add_documents(docs,ids=uuids)
            log.info("add documents done------")
            return collection_name,known_type,raw_docs
        except Exception as e:
            if e.__class__.__name__ == "UniqueConstraintError":
                return True
            log.exception(e)
            log.error(e)
            return collection_name,known_type,raw_docs

    def get_compress_retriever(self,retriever,filter_type:FilterType):
        relevant_filter = None
        # 关联性检查
        if filter_type == FilterType.LLM_FILTER:
            relevant_filter = LLMChainFilter.from_llm(self.llm,prompt=rag_filter_template)
        # 结果重排
        elif filter_type == FilterType.LLM_RERANK:
            relevant_filter = LLMListwiseRerank.from_llm(self.llm, top_n=1)
        # 相似度检查
        elif filter_type == FilterType.RELEVANT_FILTER:
            relevant_filter = EmbeddingsFilter(embeddings=self.embedding, similarity_threshold=0.76)
        elif filter_type == FilterType.LLM_EXTRACT:
            relevant_filter = LLMChainExtractor.from_llm(self.llm)

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding)
        
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, relevant_filter]
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=retriever
        )

        return compression_retriever
    
    def bm25_retriever(self,docs:List[Document],k=1):
        try:
            import jieba
            bm25_retriever = BM25Retriever.from_documents(documents=docs,preprocess_func=jieba.lcut)
            bm25_retriever.k = k
            return bm25_retriever
        except Exception as e:
            log.error(f"{e} {type(docs)}")
            print(traceback.format_exc())
    
    def get_retriever(self,collection_names="all",tenant=None,k: int=3,bm25: bool=False,filter_type=FilterType.LLM_EXTRACT,
                      sim_algo:SimAlgoType = SimAlgoType.SST):
        retriever = None
        try:
            # all的情况下，获取用户自己的collection_names
            if collection_names == "all":
                collections = self.collection_manager.list_collections(tenant=tenant)
                collection_names = [c.name for c in collections]
                
            if isinstance(collection_names, str):
                collection_names = [collection_names]  # 如果是字符串，转为列表
            log.info(f"retriever use {collection_names} for tenant {tenant}")
            retrievers = []
            docs = []
            for collection_name in collection_names:
                if sim_algo == SimAlgoType.MMR:
                    chroma_retriever = self.collection_manager.get_vector_store(collection_name,tenant=tenant).as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': k, 'lambda_mult': 0.5}
                    )
                elif sim_algo == SimAlgoType.SST:
                    # chroma_retriever = self.collection_manager.get_vector_store(collection_name,tenant=tenant).as_retriever(
                    #     search_type="similarity_score_threshold",
                    #     search_kwargs={"score_threshold": 0.5}
                    # )
                    chroma_retriever = self.collection_manager.get_vector_store(collection_name,tenant=tenant).as_retriever(
                        search_kwargs={"k": k}
                    )
                retrievers.append(chroma_retriever)
                if bm25:
                    docs.extend(self.collection_manager.get_documents(collection_name,tenant=tenant))
                
            if bm25 and len(docs) > 0:
                retrievers.append(self.bm25_retriever(docs,k))
                
            retriever = EnsembleRetriever(
                retrievers=retrievers
            )
            
            # retriever = RunnableParallel(input=RunnablePassthrough(), docs=retriever)

            if filter_type is not None:
                retriever = self.get_compress_retriever(retriever,filter_type)
            
            # 生成3个问题，增加检索的多样性
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=self.llm
            )
        except Exception as e:
            log.error(e)
        return retriever_from_llm
    
    
    def query_doc(self,
        collection_name: Union[str, List[str]],
        query: str,
        tenant: str = None,
        k: int = 3,
        bm25: bool = False,
        filter_type: FilterType = FilterType.LLM_FILTER,
        to_dict: bool = False
    ):
        collection_names = []
        if isinstance(collection_name, str):
            collection_names = [collection_name]  # 如果是字符串，转为列表
        elif isinstance(collection_name, list):
            collection_names = collection_name  # 如果是列表，直接使用
        else:
            raise ValueError("Source must be a string or a list of strings.")
        
        try:
            retriever = self.get_retriever(collection_names,tenant=tenant,k=k,bm25=bm25,filter_type=filter_type)
            if retriever is None:
                return None
            docs = retriever.invoke(query)
            docs = [d for d in docs if d.page_content and d.page_content.strip() not in ["NO_OUTPUT","NO_ OUTPUT","NO_RESPONSE"]]
            if to_dict:
                docs = {
                    "distances": [[d.metadata.get("score") for d in docs]],
                    "documents": [[d.page_content for d in docs]],
                    "metadatas": [[d.metadata for d in docs]],
                }
            return docs
        except Exception as e:
            raise e
    
    def resolve_hostname(self,hostname):
        # Get address information
        addr_info = socket.getaddrinfo(hostname, None)

        # Extract IP addresses from address information
        ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
        ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]

        return ipv4_addresses, ipv6_addresses
    
    def validate_url(self,url: Union[str, Sequence[str]]):
        if isinstance(url, str):
            if isinstance(validators.url(url), validators.ValidationError):
                raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
            # Local web fetch is disabled, filter out any URLs that resolve to private IP addresses
            parsed_url = urllib.parse.urlparse(url)
            # Get IPv4 and IPv6 addresses
            ipv4_addresses, ipv6_addresses = self.resolve_hostname(parsed_url.hostname)
            # Check if any of the resolved addresses are private
            # This is technically still vulnerable to DNS rebinding attacks, as we don't control WebBaseLoader
            for ip in ipv4_addresses:
                if validators.ipv4(ip, private=True):
                    raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
            for ip in ipv6_addresses:
                if validators.ipv6(ip, private=True):
                    raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
            return True
        elif isinstance(url, Sequence):
            return all(self.validate_url(u) for u in url)
        else:
            return False
        
    def get_web_loader(self,url: Union[str, Sequence[str]], verify_ssl: bool = True):
        # Check if the URL is valid
        if not self.validate_url(url):
            raise ValueError("Oops! The URL you provided is invalid. Please double-check and try again.")
        return SafeWebBaseLoader(
            url,
            verify_ssl=verify_ssl,
            requests_per_second=10,
            continue_on_failure=True,
        )
        # return AsyncChromiumLoader(url)
    
    def get_youtube_loader(self,url: Union[str, Sequence[str]]):
        loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language='en',
                translation=None
            )
        return loader

    def get_loader(self,filename: str, file_path: str, file_content_type: str=None):
        loader = None
        file_ext = filename.split(".")[-1].lower()
        known_type = True

        known_source_ext = [
            "go",
            "py",
            "java",
            "sh",
            "bat",
            "ps1",
            "cmd",
            "js",
            "ts",
            "css",
            "cpp",
            "hpp",
            "h",
            "c",
            "cs",
            "sql",
            "log",
            "ini",
            "pl",
            "pm",
            "r",
            "dart",
            "dockerfile",
            "env",
            "php",
            "hs",
            "hsc",
            "lua",
            "nginxconf",
            "conf",
            "m",
            "mm",
            "plsql",
            "perl",
            "rb",
            "rs",
            "db2",
            "scala",
            "bash",
            "swift",
            "vue",
            "svelte",
            "msg",
            "ex",
            "exs",
            "erl",
            "tsx",
            "jsx",
            "hs",
            "lhs",
        ]

        if file_ext == "pdf":
            loader = PyPDFLoader(
                file_path, extract_images=False
            )
        elif file_ext == "csv":
            loader = CSVLoader(file_path)
        elif file_ext == "rst":
            loader = UnstructuredRSTLoader(file_path, mode="elements")
        elif file_ext == "xml":
            loader = UnstructuredXMLLoader(file_path)
        elif file_ext in ["htm", "html"]:
            loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
        elif file_ext == "md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_content_type == "application/epub+zip":
            loader = UnstructuredEPubLoader(file_path)
        elif (
            file_content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or file_ext in ["doc", "docx"]
        ):
            loader = Docx2txtLoader(file_path)
        elif file_content_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ] or file_ext in ["xls", "xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_content_type in [
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ] or file_ext in ["ppt", "pptx"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext == "msg":
            loader = OutlookMessageLoader(file_path)
        elif file_ext ==".json":
            loader = JSONLoader(file_path)
        elif file_ext in known_source_ext or (
            file_content_type and file_content_type.find("text/") >= 0
        ):
            loader = TextLoader(file_path, autodetect_encoding=True)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)
            known_type = False

        return loader, known_type
    # decrease
    def web_parser(self,urls,tenant=None,metadata=None,collection_name=None):
        bs_transformer = BeautifulSoupTransformer()
        vector_store = None
        if collection_name:
            vector_store = self.collection_manager.get_vector_store(collection_name,tenant=tenant)
        docs = None
        if urls:
            loader = AsyncChromiumLoader(urls)
            log.info("Indexing new urls...")
            docs = loader.load()
            log.info(f"load docs:{len(docs)}")
            docs_transformed = bs_transformer.transform_documents(
                docs, tags_to_extract=["span"]
            )
            docs = list(docs_transformed)
            log.info(f"transform docs:{len(docs)}")
            docs = self.split_documents(docs)
            log.info(f"split docs:{len(docs)}")
            uuids = [str(uuid4()) for _ in range(len(docs))]
            if metadata:
                for doc in docs:
                    doc.metadata=metadata[doc.metadata['source']]
            if vector_store:
                vector_store.add_documents(docs,ids=uuids)
        return docs
        
    def web_search(self,query,tenant=None,collection_name="web",max_results=3):
        if query is None:
            return 
    
        raw_results = []
        raw_docs = []
        try:
            questions = self.search_chain.invoke({"date":datetime.now().date(),"text":query,"results_num":max_results})
            log.info(f"questions:{questions}")
            # Relevant urls
            urls,raw_results = self.do_search(questions,max_results=max_results)        
            known_type = None
            # TODO 执行网页爬虫 效果很差
            # collection_name,known_type,raw_docs = self.store(collection_name,list(urls),source_type=SourceType.WEB,tenant=tenant)
            # log.info(f"scrach results: {raw_docs}")
            # 未爬到信息，则使用检索结果拼装
            if len(raw_docs) == 0:
                collection_name,known_type,raw_docs = self.store(collection_name,raw_results,source_type=SourceType.SEARCH_RESULT,tenant=tenant)
            # 使用bm25算法重排
            if raw_docs and len(raw_docs) > 1:
                raw_docs= self.bm25_retriever(raw_docs,k=1).invoke(query)
                log.info(f"bm25 results: {raw_docs}")
            # docs = self.web_parser(urls,url_meta_map,collection_name)
            return collection_name,known_type,raw_results,raw_docs
        except Exception as e:
            log.error(f"Error search: {e}")
            print(traceback.format_exc())
            return "", False,raw_results,[]
         

    def split_documents(self, documents,chunk_size=4000,chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(separators=[
                                                    "\n\n",
                                                    "\n",
                                                    " ",
                                                    ".",
                                                    ",",
                                                    "\u200b",  # Zero-width space
                                                    "\uff0c",  # Fullwidth comma
                                                    "\u3001",  # Ideographic comma
                                                    "\uff0e",  # Fullwidth full stop
                                                    "\u3002",  # Ideographic full stop
                                                ],
                                                chunk_size=chunk_size, chunk_overlap=chunk_overlap,add_start_index=True)
        return text_splitter.split_documents(documents)

    def do_search(self, questions, max_results=2, max_retries=3):
        import random
        urls_to_look = []
        raw_results = []
        
        try:
            log.info("Searching for relevant urls...")
            for q in questions:
                retries = 0
                success = False
                
                # 尝试最大次数
                while retries < max_retries and not success:
                    try:
                        # 随机选择一个搜索引擎
                        random_key = random.choice(list(self.search_engines.keys()))
                        search = self.search_engines[random_key]
                        
                        # 获取搜索结果
                        search_results = []
                        if isinstance(search,DuckDuckGoSearchAPIWrapper):
                            search_results = search.results(q, max_results=max_results)
                        elif isinstance(search,Exa):
                            resp = search.search_and_contents(
                                q,
                                type="auto",
                                num_results=max_results,
                                text=True,
                            )
                            search_results = []
                            for r in resp.results:
                                search_results.append({
                                    "snippet": r.text,
                                    "title": r.title,
                                    "link": r.url,
                                    "date": r.published_date,
                                    "source": r.url,
                                })
                            
                        for res in search_results:
                            if res.get("link", None):
                                urls_to_look.append(res["link"])
                        
                        raw_results.extend(search_results)
                        
                        log.info(f"Search results for query '{q}': {search_results}")
                        success = True  # 如果成功，就跳出重试循环
                    
                    except Exception as e:
                        retries += 1
                        log.error(f"Error with search engine {random_key}, retrying {retries}/{max_retries}...")
                        log.error(e)
                        if retries >= max_retries:
                            log.error("Max retries reached, skipping this query.")
                        else:
                            continue  # 如果还没有达到最大重试次数，则继续尝试其他引擎
            
            log.info(f"Final search results: {raw_results}")
        
        except Exception as e:
            log.error(e)
            print(traceback.format_exc())
            
        return set(urls_to_look),raw_results
    
class SafeWebBaseLoader(WebBaseLoader):
    """WebBaseLoader with enhanced error handling for URLs."""

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path with error handling."""
        for path in self.web_paths:
            try:
                soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
                text = soup.get_text(**self.bs_get_text_kwargs)
                # Build metadata
                metadata = {"source": path}
                if title := soup.find("title"):
                    metadata["title"] = title.get_text()
                if description := soup.find("meta", attrs={"name": "description"}):
                    metadata["description"] = description.get(
                        "content", "No description found."
                    )
                if html := soup.find("html"):
                    metadata["language"] = html.get("lang", "No language found.")

                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                # Log the error and continue with the next URL
                log.error(f"Error loading {path}: {e}")
