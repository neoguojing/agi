from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document

from agi.tasks.retriever import FilterType, KnowledgeManager, SimAlgoType, SourceType


@dataclass(slots=True)
class RagService:
    """Unified RAG domain service.

    Encapsulates vector-store management, knowledge upload, and retrieval APIs
    so orchestration/subagent code depends on a single reusable interface.
    """

    km: KnowledgeManager

    async def upload(
        self,
        *,
        collection_name: str,
        source: str | list[str],
        tenant: str | None = None,
        source_type: SourceType = SourceType.FILE,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, Any, Any]:
        kwargs: dict[str, Any] = {}
        if filename is not None:
            kwargs["filename"] = filename
        if content_type is not None:
            kwargs["content_type"] = content_type
        return await self.km.store(
            collection_name=collection_name,
            source=source,
            tenant=tenant,
            source_type=source_type,
            **kwargs,
        )

    async def query(
        self,
        *,
        collection_name: str | list[str],
        query: str,
        tenant: str | None = None,
        k: int = 3,
        bm25: bool = False,
        filter_type: FilterType = FilterType.LLM_FILTER,
    ) -> list[Document] | None:
        return await self.km.query_doc(
            collection_name=collection_name,
            query=query,
            tenant=tenant,
            k=k,
            bm25=bm25,
            filter_type=filter_type,
            to_dict=False,
        )

    def list_collections(self, *, tenant: str | None = None) -> list[str]:
        return self.km.list_collections(tenant=tenant)

    def list_documents(self, *, collection_name: str, tenant: str | None = None) -> list[Document]:
        return self.km.list_documets(collection_name, tenant=tenant)

    def get_retriever(
        self,
        *,
        collection_names: str | Sequence[str] = "all",
        tenant: str | None = None,
        k: int = 3,
        bm25: bool = False,
        filter_type: FilterType | None = None,
        sim_algo: SimAlgoType = SimAlgoType.SST,
    ):
        return self.km.get_retriever(
            collection_names=collection_names,
            tenant=tenant,
            k=k,
            bm25=bm25,
            filter_type=filter_type,
            sim_algo=sim_algo,
        )


def get_rag_service() -> RagService:
    # local import to avoid heavy factory import at module import time
    from agi.tasks.task_factory import TaskFactory

    return RagService(km=TaskFactory.get_knowledge_manager())
