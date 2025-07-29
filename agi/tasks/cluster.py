from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import os
import umap
import numpy as np
from sklearn.decomposition import PCA
from langchain_core.documents import Document
from collections import defaultdict
import uuid

from agi.config import log
from agi.tasks.utils import get_last_message_text,split_think_content,graph_print
from langchain.prompts import ChatPromptTemplate
from agi.tasks.task_factory import (
    TaskFactory
)

summary_prompt = """
You are an expert text analyst. Given any long, repetitive or log‑style input in a human language, produce a concise, factual summary in exactly three sentences:

1. Context: one sentence stating the source or type of text (e.g., “model training log,” “system report,” “data export”).  
2. Overview: one sentence describing the main repetitive patterns or key information (e.g., “lists AuxLossBased vs. AuxLossFree comparisons across multiple layers”).  
3. Recommendation: one sentence suggesting a next step (e.g., “convert to a table for easier analysis” or “extract key terms for topic clustering”).

Requirements:  
- No speculative or subjective language (avoid words like “possible,” “likely,” etc.).  
- Strictly three sentences—no more, no less.  
- Output only the three‑sentence summary in the same language as the input.
"""

summary_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            summary_prompt
        ),
        ("human", "{text}")
    ]
)

class TextClusterer:
    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 1,
                 use_umap: bool = False,
                 umap_dim: int = 5):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_dim = umap_dim
        self.cluster_top_k_keywords = 10
        self.llm = TaskFactory.get_llm() 



    def summary(self,text:str):
        value = summary_template.invoke({"text":f'{text} /no_think'})
        ai = self.llm.invoke(value)
        _, result = split_think_content(ai.content)

        return result

    def _reduce_dim(self, vectors: np.ndarray) -> np.ndarray:
        # 先PCA降维至50维，再UMAP降至目标维度，减少大规模时的计算压力
        # 4. 标准化
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)

        pca = PCA(n_components=50, random_state=42)
        pca_result = pca.fit_transform(vectors)
        reducer = umap.UMAP(n_components=self.umap_dim, random_state=42)
        return reducer.fit_transform(pca_result)

    def combined_keywords(self,all_keywords:list):
        # 3.3 加权统计关键词
        keyword_weights = defaultdict(float)
        for kw, weight in all_keywords:
            keyword_weights[kw] += weight

        # 3.4 选择top-k关键词
        sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
        combined_keywords = [kw for kw, _ in sorted_keywords[:self.cluster_top_k_keywords]]
        return combined_keywords

    def cluster(self, docs: List[Document], filtered_texts: List[str], embeddings: np.ndarray) -> list:
        pairs = list(zip(docs, filtered_texts))  # [(原文, 清洗后)]
        if isinstance(embeddings,list):
            embeddings = np.array(embeddings, dtype=float)
        if self.use_umap:
            embeddings = self._reduce_dim(embeddings)

        n_samples = len(embeddings)
        labels = None
        # 1. 样本足够 -> HDBSCAN
        if n_samples >= self.min_cluster_size:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            labels = clusterer.fit_predict(embeddings)
            log.info(f"got {len(set(labels))} clusters by hdbscan")

        # 2. 样本太少 -> fallback
        elif n_samples > 1:
            # 简单处理：用KMeans强制分两类；如果相似度很高则归为一类
            sim = cosine_similarity(embeddings)
            avg_sim = (sim.sum() - np.trace(sim)) / (n_samples*(n_samples-1))
            if avg_sim > 0.85:
                labels = np.zeros(n_samples, dtype=int)  # 所有样本归为同一类
            else:
                kmeans = KMeans(n_clusters=min(n_samples, 2), random_state=0)
                labels = kmeans.fit_predict(embeddings)
            log.info(f"got {len(set(labels))} clusters by KMeans")

        # 3. 只有一个样本
        else:
            labels = np.array([0])

        clusters = []
        # 防止混入np.int64
        labels = labels.tolist()
        for label in set(labels):
            if label == -1:
                continue

            cluster_id = str(uuid.uuid4())
            cluster_doc = Document(page_content="")
            cluster_doc.metadata["doc_id"] = cluster_id
            cluster_doc.metadata["cluster_id"] = cluster_id
            cluster_doc.metadata["label"] = label
            cluster_doc.metadata["related_docs"] = []
            context_texts = ""
            all_keywords = []
            for (doc, filtered), emb, l in zip(pairs, embeddings, labels):
                if l != label:
                    continue
                # 每个 doc 一个唯一 ID
                doc_id = str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
                doc.metadata["cluster_id"] = cluster_id
                doc.metadata["source"] = os.path.basename(doc.metadata["source"])
                if cluster_doc.metadata.get("source") is None:
                    cluster_doc.metadata["source"] = doc.metadata["source"]
                cluster_doc.metadata["related_docs"].append(doc_id)
                context_texts  += "\n\n" + doc.page_content
                keywords = doc.metadata.get("keywords", [])
                all_keywords.extend(keywords)

            cluster_doc.metadata["keywords"] = self.combined_keywords(all_keywords)
            cluster_doc.page_content = self.summary(context_texts)
            clusters.append(cluster_doc)

        return clusters
