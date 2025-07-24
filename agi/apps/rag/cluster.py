from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
import numpy as np
from sklearn.decomposition import PCA
from langchain_core.documents import Document
from agi.config import STOP_WORDS_PATH
from collections import defaultdict
import uuid

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

    def cluster(self, docs: List[Document], filtered_texts: List[str], embeddings: np.ndarray) -> Dict[str, Dict]:
        pairs = list(zip(docs, filtered_texts))  # [(原文, 清洗后)]

        if self.use_umap:
            embeddings = self._reduce_dim(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        labels = clusterer.fit_predict(embeddings)

        clusters = {}

        for label in set(labels):
            if label == -1:
                continue

            cluster_id = str(uuid.uuid4())
            cluster_doc = Document(page_content="")
            cluster_doc.metadata["doc_id"] = cluster_id
            cluster_doc.metadata["cluster_id"] = cluster_id
            cluster_doc.metadata["label"] = label
            cluster_doc.metadata["related_docs"] = []

            all_keywords = []
            for (doc, filtered), emb, l in zip(pairs, embeddings, labels):
                if l != label:
                    continue
                # 每个 doc 一个唯一 ID
                doc_id = str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
                doc.metadata["cluster_id"] = cluster_id
                cluster_doc.metadata["related_docs"].append(doc_id)
                cluster_doc.page_content  += "\n\n" + doc.page_content

                keywords = doc.metadata.get("keywords", [])
                all_keywords.extend(keywords)

            cluster_doc.metadata["keywords"] = self.combined_keywords(all_keywords)

            clusters[cluster_id] = {
                "doc": cluster_doc,
                "embding": None
            }

        return clusters
