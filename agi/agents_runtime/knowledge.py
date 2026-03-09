from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .messages import create_knowledge_system_message, message_to_payload
from .types import MemorySearchResult


@dataclass(slots=True)
class KnowledgeChunk:
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] | None = None


class KnowledgeFusionService:
    """知识库构建/检索与上下文融合入口。"""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    async def build_knowledge(
        self,
        collection: str,
        source: str,
        *,
        tenant_id: str | None = None,
        source_type: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.adapter.store(
            collection_name=collection,
            source=source,
            tenant=tenant_id,
            source_type=source_type,
            **kwargs,
        )

    async def search(
        self,
        collection: str | list[str],
        query: str,
        *,
        tenant_id: str | None = None,
        top_k: int = 4,
    ) -> list[KnowledgeChunk]:
        docs = await self.adapter.query_doc(collection, query, tenant=tenant_id, k=top_k, to_dict=False)
        if not docs:
            return []

        out: list[KnowledgeChunk] = []
        for item in docs:
            content = getattr(item, "page_content", str(item))
            metadata = getattr(item, "metadata", {}) or {}
            out.append(
                KnowledgeChunk(
                    content=content,
                    source=str(metadata.get("source", metadata.get("collection_name", "knowledge"))),
                    score=float(metadata.get("score", 0.0)),
                    metadata=metadata,
                )
            )
        return out

    @staticmethod
    def inject_to_messages(messages: list[dict[str, Any]], chunks: list[KnowledgeChunk]) -> list[dict[str, Any]]:
        if not chunks:
            return messages

        cite = "\n\n".join([f"[{idx+1}] {c.content}" for idx, c in enumerate(chunks)])
        knowledge_msg = create_knowledge_system_message("以下是可用知识库检索结果，请优先参考：\n" + cite)
        return [message_to_payload(knowledge_msg)] + messages

    @staticmethod
    def to_memory_hits(chunks: list[KnowledgeChunk]) -> list[MemorySearchResult]:
        return [
            MemorySearchResult(
                content=item.content,
                source=item.source,
                score=item.score,
                metadata=item.metadata or {},
            )
            for item in chunks
        ]
