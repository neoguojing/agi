import asyncio
from typing import List, Optional
import httpx
from langchain_core.documents import Document
from agi.config import EMBEDDING_BASE_URL,RAG_EMBEDDING_MODEL,log
from urllib.parse import urljoin
import traceback

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

MAX_CONCURRENCY = 3  # 同时最多有3个请求在跑

sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def rerank_batch(
    client: httpx.AsyncClient,
    endpoint: str,
    query: str,
    documents: List[Document],
    model: str = RAG_EMBEDDING_MODEL,
    top_k: Optional[int] = 3
) -> List[RerankItem]:
    payload = {
        "query": query,
        "documents": [doc.page_content for doc in documents],
        "model": model,
        "top_k": top_k
    }
    endpoint = urljoin(endpoint,"/v1/rerank")
    
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

async def safe_rerank_batch(*args, **kwargs):
    async with sem:
        return await rerank_batch(*args, **kwargs)
    
async def rerank_with_batching(
    query: str,
    documents: List[Document],
    endpoint: str = EMBEDDING_BASE_URL,
    model: str = RAG_EMBEDDING_MODEL,
    top_k: Optional[int] = 3,
    batch_size: int = 10
) -> List[Document]:
    all_results = []
    try:
        async with httpx.AsyncClient() as client:
            tasks = []
            batches = [(documents[i:i + batch_size], i) for i in range(0, len(documents), batch_size)]

            for batch, _ in batches:
                tasks.append(
                    safe_rerank_batch(client, endpoint, query, batch, model)
                )

            results = await asyncio.gather(*tasks)

            for (batch, _), reranked in zip(batches, results):
                for item in reranked:
                    doc = batch[item.index]
                    scored_doc = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "score": item.score}
                    )
                    all_results.append((scored_doc, item.score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in all_results]

        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs
    except Exception as e:
        log.error(f"Error rerank_with_batching: {e}")
        print(traceback.format_exc())
        return documents[:3]
