from __future__ import annotations

from typing import Any

from langchain.tools import tool

from agi.tasks.rag.knowledge import FilterType, SourceType
from agi.tasks.rag.service import get_rag_service


@tool(return_direct=True)
async def rag_list_collections(tenant: str = "") -> list[str]:
    """List available RAG collections for a tenant."""
    service = get_rag_service()
    return service.list_collections(tenant=tenant or None)


@tool(return_direct=True)
async def rag_upload_documents(
    collection_name: str,
    source: str,
    source_type: str = "file",
    tenant: str = "",
    filename: str = "",
    content_type: str = "",
) -> dict[str, Any]:
    """Upload a source into vector store.

    source_type: file | web | youtube
    """
    service = get_rag_service()
    type_map = {
        "file": SourceType.FILE,
        "web": SourceType.WEB,
        "youtube": SourceType.YOUTUBE,
    }
    selected_type = type_map.get((source_type or "file").lower(), SourceType.FILE)
    collection, known_type, docs = await service.upload(
        collection_name=collection_name,
        source=source,
        tenant=tenant or None,
        source_type=selected_type,
        filename=filename or None,
        content_type=content_type or None,
    )
    return {
        "collection_name": collection,
        "known_type": known_type,
        "chunks": len(docs) if isinstance(docs, list) else 0,
    }


@tool(return_direct=True)
async def rag_query(
    query: str,
    collection_name: str = "all",
    tenant: str = "",
    top_k: int = 3,
    bm25: bool = False,
    filter_type: str = "llm_chain_filter",
) -> list[dict[str, Any]]:
    """Run retrieval from vector store and return compact documents."""
    service = get_rag_service()
    filter_map = {
        "llm_chain_filter": FilterType.LLM_FILTER,
        "llm_listwise_rerank": FilterType.LLM_RERANK,
        "embeddings_filter": FilterType.RELEVANT_FILTER,
        "llm_extract": FilterType.LLM_EXTRACT,
    }
    docs = await service.query(
        collection_name=collection_name,
        query=query,
        tenant=tenant or None,
        k=top_k,
        bm25=bm25,
        filter_type=filter_map.get(filter_type, FilterType.LLM_FILTER),
    )
    if not docs:
        return []
    return [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]


rag_builtin_tools = [
    rag_list_collections,
    rag_upload_documents,
    rag_query,
]
