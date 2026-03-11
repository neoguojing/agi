import os
import uuid
import hashlib
from typing import List, Optional, Dict
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language
)
from langchain_core.documents import Document

class CustomDocumentSplitter:
    """
    增强版文档切分器：自动注入溯源 ID、块序号与结构化标题
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 基础递归切分器：add_start_index=True 能够自动记录块在原文中的偏移量
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True 
        )

    def split_text(
        self, 
        text: str, 
        file_type: Optional[str] = None, 
        file_name: Optional[str] = "unknown",
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        统一入口：自动识别类型并注入增强元数据
        """
        # 1. 初始化基础元数据
        base_meta = metadata or {}
        base_meta.update({
            "source": file_name,
            "doc_id": str(uuid.uuid4())[:8], # 原始文档 ID
            "content_hash": hashlib.md5(text.encode()).hexdigest()[:12] # 内容指纹，用于去重
        })

        dtype = file_type.lower() if file_type else self._guess_content_type(text)
        
        # 2. 路由切分逻辑
        if dtype in ["md", "markdown"]:
            docs = self._split_markdown(text, base_meta)
        elif dtype in ["html", "htm"]:
            docs = self._split_html(text, base_meta)
        elif dtype in ["py", "python", "js", "ts", "java", "cpp"]:
            docs = self._split_by_language(text, dtype, base_meta)
        else:
            docs = self.base_splitter.create_documents([text], metadatas=[base_meta])
        
        # 3. 二次增强：注入块序号和全局唯一块 ID
        return self._enrich_chunk_metadata(docs)

    def _enrich_chunk_metadata(self, docs: List[Document]) -> List[Document]:
        """注入块级别的详细信息"""
        for i, doc in enumerate(docs):
            # 块序号
            doc.metadata["chunk_index"] = i
            # 全局唯一 Chunk ID (文档ID + 序号)
            doc.metadata["chunk_id"] = f"{doc.metadata.get('doc_id')}-c{i}"
            # 拼接一个可读的简短摘要或标题作为 reference_title
            if not doc.metadata.get("title"):
                # 尝试从 Markdown/HTML 提取的 Header 拼接
                headers = [doc.metadata.get(k) for k in ["H1", "H2", "H3"] if doc.metadata.get(k)]
                doc.metadata["breadcrumb"] = " > ".join(headers) if headers else doc.metadata.get("source")
        return docs

    def _guess_content_type(self, text: str) -> str:
        s = text.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return "json"
        if "<html" in s.lower() or "<!doctype" in s.lower():
            return "html"
        if s.startswith("# ") or "\n# " in s:
            return "md"
        if "def " in s and ":" in s:
            return "py"
        return "txt"

    def _split_markdown(self, text: str, meta: dict) -> List[Document]:
        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        docs = splitter.split_text(text)
        return self._finalize_docs(docs, meta)

    def _split_html(self, text: str, meta: dict) -> List[Document]:
        headers = [("h1", "H1"), ("h2", "H2"), ("h3", "H3")]
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
        docs = splitter.split_text(text)
        return self._finalize_docs(docs, meta)

    def _split_by_language(self, text: str, dtype: str, meta: dict) -> List[Document]:
        lang_map = {
            "py": Language.PYTHON, "python": Language.PYTHON,
            "js": Language.JS, "ts": Language.JS,
            "java": Language.JAVA, "cpp": Language.CPP
        }
        c_splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_map.get(dtype, Language.PYTHON), 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        return c_splitter.create_documents([text], metadatas=[meta])

    def _finalize_docs(self, docs: List[Document], base_meta: dict) -> List[Document]:
        """补充元数据并物理降维"""
        for d in docs:
            d.metadata.update(base_meta)
        return self.base_splitter.split_documents(docs)