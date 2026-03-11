import asyncio
import traceback
from datetime import datetime
from typing import List, Union, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
import os
import hashlib
# 核心内部依赖
from agi.rag.file_loader import LoaderFactory
from agi.rag.spliter import CustomDocumentSplitter
from agi.rag.chroma import CollectionManager, ChromaClientFactory,CollectionHandle

from agi.config import log

class KnowledgeManager:
    def __init__(self, data_path, embedding):
        self.embedding = embedding
        self.factory = ChromaClientFactory(data_path)
        self.collection_manager = CollectionManager(self.factory, embedding)
        # 统一分块策略
        self.splitter = CustomDocumentSplitter(chunk_size=1000, chunk_overlap=150)

    # --- 1. 存储模块：默认处理文件，支持幂等 ---

    async def store(
        self, 
        collection_name: str, 
        content: Union[str, List[str], Document, List[Document]], 
        tenant: Optional[str] = None,
        source_name: Optional[str] = None, # 如果是纯文本，建议提供一个标识名
        **kwargs
    ):
        """
        统一入库接口，支持文件路径、纯字符串和 LangChain Document。
        """
        handle = self.collection_manager.get_handle(collection_name, tenant=tenant)
        
        # 1. 统一转化为 List[Document]
        docs = self._to_documents(content, source_name)
        
        # 2. 幂等性过滤：根据 metadata 中的 source 检查是否已存在
        new_docs = []
        seen_sources = set()
        for doc in docs:
            src = doc.metadata.get("source")
            if src not in seen_sources and not self._is_indexed(handle, src):
                new_docs.append(doc)
                seen_sources.add(src)

        if not new_docs:
            log.info(f"All provided content already indexed in {collection_name}.")
            return collection_name

        # 3. 并行处理切分与入库
        tasks = [self._process_and_upsert(doc, handle, collection_name, tenant) for doc in new_docs]
        await asyncio.gather(*tasks)
        
        log.info(f"Successfully indexed {len(new_docs)} sources into {collection_name}")
        return collection_name

    def _to_documents(self, content, source_name) -> List[Document]:
        """将各种输入格式归一化为 Document 对象"""
        if isinstance(content, Document):
            return [content]
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], Document):
            return content

        # 处理字符串或字符串列表
        raw_list = [content] if isinstance(content, str) else content
        docs = []
        for i, text in enumerate(raw_list):
            # 如果是路径则保留路径，否则生成 Hash 标识
            is_file = os.path.exists(text) if isinstance(text, str) and len(text) < 255 else False
            
            if is_file:
                # 暂时只记录路径，由 _process_and_upsert 中的 Loader 处理
                docs.append(Document(page_content="", metadata={"source": text, "is_file": True}))
            else:
                # 纯文本输入
                # 如果没有提供 source_name，则根据内容生成哈希作为唯一标识
                uid = source_name or f"txt_{hashlib.md5(text.encode()).hexdigest()[:12]}"
                docs.append(Document(page_content=text, metadata={"source": uid, "is_file": False}))
        return docs

    async def _process_and_upsert(self, doc, handle, collection_name, tenant):
        """处理 Document（包括文件加载和文本切分）"""
        try:
            source = doc.metadata["source"]
            
            # 如果是文件路径，需要加载内容
            if doc.metadata.get("is_file"):
                loader = LoaderFactory.get_loader(source)
                raw_docs = await asyncio.to_thread(loader.load)
            else:
                raw_docs = [doc]

            final_splits = []
            for d in raw_docs:
                # 补充基础元数据
                d.metadata.update({
                    "source": source,
                    "tenant": tenant,
                    "collection_name": collection_name,
                    "ingest_at": datetime.now().isoformat()
                })
                
                # 执行切分
                file_ext = source.split(".")[-1].lower() if "." in source else "txt"
                splits = self.splitter.split_text(
                    text=d.page_content,
                    file_type=file_ext,
                    file_name=source,
                    metadata=d.metadata
                )
                final_splits.extend(splits)

            if final_splits:
                await handle.upsert(final_splits)
                
        except Exception as e:
            log.error(f"Failed to process source {doc.metadata.get('source')}: {e}")

    def _is_indexed(self, handle, source_id: str) -> bool:
        """检查标识是否已存在"""
        try:
            res = handle.collection.get(where={"source": source_id}, include=[], limit=1)
            return len(res['ids']) > 0
        except Exception:
            return False

    # --- 2. 检索模块：纯算法混合检索 ---

    def get_retriever(self, collection_names="all", tenant=None, k=4):
        """
        构造高性能检索器：向量(60%) + 关键词(40%)
        """
        import jieba
        
        if collection_names == "all":
            collection_names = self.collection_manager.list_collections(tenant=tenant)
        
        names = [collection_names] if isinstance(collection_names, str) else collection_names
        v_retrievers = []
        all_docs = []

        for name in names:
            handle = self.collection_manager.get_handle(name, tenant=tenant)
            
            # 使用 LangChain Chroma 包装底层 Client
            vectorstore = Chroma(
                client=self.factory.get_client(tenant=tenant),
                collection_name=name,
                embedding_function=self.embedding
            )
            v_retrievers.append(vectorstore.as_retriever(search_kwargs={"k": k}))
            
            # 收集文档用于 BM25 (限制最大拉取量以保证速度)
            all_docs.extend(handle.get_documents(limit=5000))

        # 构造融合检索
        if all_docs:
            bm25_r = BM25Retriever.from_documents(all_docs, preprocess_func=jieba.lcut, k=k)
            base_retriever = EnsembleRetriever(
                retrievers=v_retrievers + [bm25_r],
                weights=[0.6 / len(v_retrievers)] * len(v_retrievers) + [0.4]
            )
        else:
            base_retriever = EnsembleRetriever(retrievers=v_retrievers)

        # 纯数学过滤链 (无 LLM 介入)
        pipeline = DocumentCompressorPipeline(transformers=[
            EmbeddingsRedundantFilter(embeddings=self.embedding),
            EmbeddingsFilter(embeddings=self.embedding, similarity_threshold=0.65)
        ])
        
        return ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=base_retriever)

    async def query_doc(self, collection_name, query, tenant=None, k=4):
        """极速异步检索接口"""
        if not query.strip(): return []
        retriever = self.get_retriever(collection_name, tenant=tenant, k=k)
        return await retriever.ainvoke(query)