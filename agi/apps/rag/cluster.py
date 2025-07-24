from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import hdbscan
import umap
import numpy as np
import re
import jieba
from sklearn.decomposition import PCA

from agi.utils.nlp import TextProcessor
from agi.tasks.task_factory import TaskFactory  # 嵌入模型工厂
from agi.config import STOP_WORDS_PATH


class LLMInterface:
    def summarize(self, texts: List[str]) -> str:
        # 这里写你调用 LLM 的逻辑，例如 OpenAI GPT 或其他模型
        return texts[0][:100] + "..." if texts else "No summary available."


class TextClusterer:
    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 1,
                 use_umap: bool = False,
                 umap_dim: int = 5,
                 clean: bool = True,
                 parallel: bool = True,
                 batch_size: int = 64,
                 max_workers: int = 4):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_dim = umap_dim
        self.clean_enabled = clean
        self.parallel = parallel
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.nlp = TextProcessor()

        self.model = TaskFactory.get_embedding()
        self.llm = LLMInterface()
        self._embedding_cache = {}

    def _embed_parallel(self, texts: List[str]) -> np.ndarray:
        results = [None] * len(texts)
        to_compute = []
        indices_to_compute = []

        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                results[i] = None
                continue
            if text in self._embedding_cache:
                results[i] = self._embedding_cache[text]
            else:
                to_compute.append(text)
                indices_to_compute.append(i)

        def embed_batch(batch_texts):
            return self.model.embed_documents(batch_texts)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for start in range(0, len(to_compute), self.batch_size):
                batch_texts = to_compute[start:start + self.batch_size]
                futures.append(executor.submit(embed_batch, batch_texts))
            for future_i, future in enumerate(futures):
                batch_embeddings = future.result()
                start = future_i * self.batch_size
                for i_b, emb in enumerate(batch_embeddings):
                    idx = indices_to_compute[start + i_b]
                    results[idx] = emb
                    self._embedding_cache[to_compute[start + i_b]] = emb

        emb_dim = len(next(emb for emb in results if emb is not None))
        results = [
            np.zeros(emb_dim, dtype=np.float32) if v is None else v
            for v in results
        ]

        return np.array(results)

    def _reduce_dim(self, vectors: np.ndarray) -> np.ndarray:
        # 先PCA降维至50维，再UMAP降至目标维度，减少大规模时的计算压力
        # 4. 标准化
        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)

        pca = PCA(n_components=50, random_state=42)
        pca_result = pca.fit_transform(vectors)
        reducer = umap.UMAP(n_components=self.umap_dim, random_state=42)
        return reducer.fit_transform(pca_result)

    def cluster(self, texts: List[str]) -> Dict[int, Dict]:
        # 1. 清洗文本并构建 (原文, 清洗后) 对
        texts = self.nlp.clean_batch(texts)
        filtered_texts = self.nlp.remove_stopwords_batch(texts)
        pairs = list(zip(texts, filtered_texts))  # [(原文, 清洗后)]

        # 2. 向量化
        embeddings = self._embed_parallel(filtered_texts)

        # 3. 可选降维
        if self.use_umap:
            embeddings = self._reduce_dim(embeddings)

        # 5. 聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        labels = clusterer.fit_predict(embeddings)

        # 6. 构建聚类结果
        clusters = {}
        for label in set(labels):
            if label == -1:
                continue  # 跳过噪声

            # 收集该聚类下所有文本及其信息
            cluster_items = [
                {
                    "text": orig,
                    "text_filted": filterd,
                    "embedding": emb,
                    "label": label
                }
                for (orig, filterd), emb, l in zip(pairs, embeddings, labels)
                if l == label
            ]

            # 用清洗后的文本做摘要和关键词提取
            cluster_texts_filtered = [item["text_filted"] for item in cluster_items]
            summary = self.llm.summarize(cluster_texts_filtered)
            keywords = self.nlp.batch_process(cluster_texts_filtered)

            clusters[label] = {
                "items": cluster_items,
                "summary": summary,
                "keywords": keywords
            }

        return clusters

