import asyncio
from typing import List, Optional
import httpx
from langchain_core.documents import Document
from agi.config import EMBEDDING_BASE_URL
from urllib.parse import urljoin
class RerankItem:
    def __init__(self, object: str, index: int, document: str, score: float):
        self.object = object
        self.index = index
        self.document = document
        self.score = score

    def to_dict(self):
        return {
            "object": self.object,
            "index": self.index,
            "document": self.document,
            "score": self.score
        }

async def rerank_batch(
    client: httpx.AsyncClient,
    endpoint: str,
    query: str,
    documents: List[Document],
    model: str = "bge-reranker-base",
    top_k: Optional[int] = None
) -> List[RerankItem]:
    payload = {
        "query": query,
        "documents": [doc.page_content for doc in documents],
        "model": model,
        "top_k": top_k
    }
    endpoint = urljoin(endpoint,"rerank")
    resp = await client.post(endpoint, json=payload)
    resp.raise_for_status()
    result = resp.json()

    return [
        RerankItem(
            object=item["object"],
            index=item["index"],
            document=item["document"],
            score=item["score"]
        )
        for item in result["data"]
    ]

async def rerank_with_batching(
    query: str,
    documents: List[Document],
    endpoint: str = EMBEDDING_BASE_URL,
    model: str = "qwen",
    top_k: Optional[int] = 3,
    batch_size: int = 50
) -> List[Document]:
    all_results: List[tuple[Document, float]] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            reranked = await rerank_batch(client, endpoint, query, batch, model)

            for item in reranked:
                # 将得分写入 metadata，保留原文
                doc = batch[item.index]
                scored_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": item.score}
                )
                all_results.append((scored_doc, item.score))

    # 全部打平后统一排序
    all_results.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in all_results]

    if top_k is not None:
        reranked_docs = reranked_docs[:top_k]

    return reranked_docs
