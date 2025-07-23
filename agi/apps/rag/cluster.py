from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import hdbscan
import umap
import numpy as np
import re

from agi.tasks.task_factory import TaskFactory  # 嵌入模型工厂

# 假设你有 LLM 摘要接口：
class LLMInterface:
    def summarize(self, texts: List[str]) -> str:
        # 这里写你调用 LLM 的逻辑，例如 OpenAI GPT 或其他模型
        # 下面是示意，返回第一个文本的前100字符
        return texts[0][:100] + "..." if texts else "No summary available."

class TextClusterer:
    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 1,
                 use_umap: bool = False,
                 umap_dim: int = 5,
                 clean: bool = True,
                 parallel: bool = True):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_dim = umap_dim
        self.clean_enabled = clean
        self.parallel = parallel

        self.model = TaskFactory.get_embedding()
        self.llm = LLMInterface()
        self._embedding_cache = {}

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)  # 简单去多余空白
        return text.strip()

    def _clean_batch(self, texts: List[str]) -> List[str]:
        if not self.clean_enabled:
            return texts
        if self.parallel:
            with ThreadPoolExecutor() as executor:
                return list(executor.map(self._clean_text, texts))
        else:
            return [self._clean_text(t) for t in texts]

    def _embed(self, texts: List[str]) -> np.ndarray:
        results = []
        to_compute = []
        indices_to_compute = []

        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                results.append(None)
                continue
            if text in self._embedding_cache:
                results.append(self._embedding_cache[text])
            else:
                results.append(None)
                to_compute.append(text)
                indices_to_compute.append(i)

        if to_compute:
            embeddings = self.model.embed_documents(to_compute)
            for idx, emb in zip(indices_to_compute, embeddings):
                results[idx] = emb
                self._embedding_cache[to_compute[idx - indices_to_compute[0]]] = emb

        emb_dim = len(results[next(i for i,v in enumerate(results) if v is not None)])
        results = [
            np.zeros(emb_dim, dtype=np.float32) if v is None else v
            for v in results
        ]

        return np.array(results)

    def _reduce_dim(self, vectors: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(n_components=self.umap_dim, random_state=42)
        return reducer.fit_transform(vectors)

    def _extract_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """
        简单用TF-IDF提取关键词，返回top_k关键词
        """
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
        top_n = feature_array[tfidf_sorting][:top_k]
        return top_n.tolist()

    def cluster(self, texts: List[str]) -> Dict[int, Dict]:
        """
        聚类并返回带标签和关键词的结构化结果

        返回格式：
        {
            label: {
                "texts": [str],
                "summary": str,
                "keywords": List[str]
            },
            ...
        }
        """
        texts_cleaned = self._clean_batch(texts)
        embeddings = self._embed(texts_cleaned)

        if self.use_umap:
            embeddings = self._reduce_dim(embeddings)

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples)
        labels = clusterer.fit_predict(embeddings_scaled)

        clusters = {}
        for label in set(labels):
            if label == -1:
                continue  # 噪声点可选跳过或单独处理
            cluster_texts = [t for t, l in zip(texts_cleaned, labels) if l == label]
            summary = self.llm.summarize(cluster_texts)
            keywords = self._extract_keywords(cluster_texts)
            clusters[label] = {
                "texts": cluster_texts,
                "summary": summary,
                "keywords": keywords
            }

        return clusters
