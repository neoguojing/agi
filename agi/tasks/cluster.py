from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
import os
import numpy as np
from langchain_core.documents import Document
from collections import defaultdict
import uuid
from typing import List,Dict,Optional
from agi.config import log,CLUSTER_ALGO
from agi.tasks.utils import get_last_message_text,split_think_content,graph_print
from langchain.prompts import ChatPromptTemplate
from agi.tasks.task_factory import (
    TaskFactory
)
import random
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
summary_prompt = """
You are an expert text analyst. Given any long, repetitive, or log-style input in a human language, generate a concise, factual summary in exactly three sentences, following these rules:

Context: One sentence stating the source or type of text (e.g., “model training log,” “system report,” “data export”).

Overview: One sentence describing the main repetitive patterns or key information (e.g., “records runtime status of different modules across stages”).

Requirements:

Use exactly three sentences, no more, no less.

Avoid speculative or subjective language (no “likely,” “possibly,” etc.).

The output language must match the input language (if the input is in Chinese, return the summary in Chinese).

Output only the three-sentence summary—no explanations or extra text
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
    def __init__(self, cluster_algo=CLUSTER_ALGO,hnsw_m=32, ef_search=128, distance_threshold=0.5,
                 min_cluster_size: int = 2,min_samples: int = 6,
                 use_umap: bool = True,umap_dim: int = 15,umap_n_neighbors: int = 10,umap_min_dist: float = 0.1):
        """
        初始化参数。
        Args:
            hnsw_m (int): HNSW索引的邻居数。
            ef_search (int): HNSW搜索时的邻居数。
            distance_threshold (float): 距离阈值的平方。
                                        对于归一化向量, similarity = 1 - (distance^2 / 2)。
                                        例如, 如果想要相似度 > 0.75, 则 distance^2 应 < 2 * (1 - 0.75) = 0.5。
                                        所以这里设置 distance_threshold=0.5 意味着相似度阈值为 0.75。
        """
        self.cluster_top_k_keywords = 10
        self.llm = TaskFactory.get_llm()

        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.distance_threshold = distance_threshold
        self.candidate_k = 5

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_dim = umap_dim
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        self.cluster_algo = cluster_algo
        

    def summary(self,text:str):
        value = summary_template.invoke({"text":f'{text} /no_think'})
        ai = self.llm.invoke(value)
        _, result = split_think_content(ai.content)

        return result

    def cluster_range_reward(self,n_clusters: int, n_samples: int, 
                         target_min: float = 0.05, target_max: float = 0.15) -> float:
        """
        对于类簇数量是否落在理想比例区间内，给予奖励/惩罚。
        返回值越高表示越接近目标区间。
        """
        ratio = n_clusters / n_samples
        if target_min <= ratio <= target_max:
            return 0.0  # 完美落在区间内，无惩罚
        elif ratio < target_min:
            return (target_min - ratio) ** 2 * 10  # 偏少惩罚
        else:  # ratio > target_max
            return (ratio - target_max) ** 2 * 10  # 偏多惩罚

    def evaluate_clusters(self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 10000,
        random_state: int = 42,
        cluster_penalty: float = None,  # 新增参数，控制惩罚权重
    ) -> Dict[str, float]:
        """
        对聚类结果做内部指标评估，并综合聚类数量考虑最终评分。

        Args:
            embeddings: (n_samples, n_features) 原始或降维后的嵌入矩阵
            labels:     (n_samples,) 每个样本的簇标签
            sample_size: 最大采样数，避免大规模数据计算过慢
            random_state: 随机种子
            cluster_penalty: 聚类数量惩罚权重

        Returns:
            dict:
                silhouette
                davies_bouldin
                calinski_harabasz
                n_clusters
                score -- 综合评分
        """
        n_samples = labels.shape[0]

        if n_samples > sample_size:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            emb_samp = embeddings[idx]
            lab_samp = labels[idx]
        else:
            emb_samp = embeddings
            lab_samp = labels

        unique_labels = set(lab_samp)
        n_clusters = len(unique_labels)

        results = {}
        if n_clusters >= 2:
            results["silhouette"] = silhouette_score(
                emb_samp, lab_samp, metric="euclidean", random_state=random_state
            )
            results["davies_bouldin"] = davies_bouldin_score(emb_samp, lab_samp)
            results["calinski_harabasz"] = calinski_harabasz_score(emb_samp, lab_samp)
        else:
            results["silhouette"] = float("nan")
            results["davies_bouldin"] = float("nan")
            results["calinski_harabasz"] = float("nan")

        results["n_clusters"] = n_clusters

        if cluster_penalty is None:
            cluster_penalty = self.cluster_range_reward(n_clusters, n_samples)
        
        # 综合评分
        if all(np.isfinite([results["silhouette"], results["davies_bouldin"], results["calinski_harabasz"]])):
            results["score"] = (
                results["silhouette"]
                - 0.2 * results["davies_bouldin"]
                + 0.001 * results["calinski_harabasz"]
                - cluster_penalty
            )
        else:
            results["score"] = float("nan")

        return results
    
    def combined_keywords(self,all_keywords:list):
        # 3.3 加权统计关键词
        keyword_weights = defaultdict(float)
        for kw, weight in all_keywords:
            keyword_weights[kw] += weight

        # 3.4 选择top-k关键词
        sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
        combined_keywords = [(kw,weight) for kw, weight in sorted_keywords[:self.cluster_top_k_keywords]]
        return combined_keywords

    def _reduce_dim(self, vectors: np.ndarray) -> np.ndarray:
        # 先PCA降维至50维，再UMAP降至目标维度，减少大规模时的计算压力
        # pca = PCA(n_components=50, random_state=42)
        # vectors = pca.fit_transform(vectors)
        reducer = umap.UMAP(
                n_components=self.umap_dim,
                n_neighbors=self.umap_n_neighbors,
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=42
        )
        return reducer.fit_transform(vectors)
    
    def do_hdbscan(self,embeddings):
        normed_embeddings = embeddings.copy()

        # 3. 标准化
        # scaler = StandardScaler()
        # normed_embeddings = scaler.fit_transform(normed_embeddings)

        # 4. 降维
        if self.use_umap:
            normed_embeddings = self._reduce_dim(normed_embeddings)

        # 5. 聚类
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        )
        labels = clusterer.fit_predict(normed_embeddings)
        
        print(f"Clustering with hdbscan created {len(set(labels))} clusters.")
        return labels
    
    def do_dpmeans(self,embeddings):
        import faiss
        n_samples, dim = embeddings.shape
        if n_samples == 0:
            return []

        # (可选但推荐) 随机打乱输入顺序，减轻顺序敏感性
        # indices = list(range(n_samples))
        # random.shuffle(indices)
        # docs = [docs[i] for i in indices]
        # embeddings = embeddings[indices, :]

        # 1. 归一化嵌入向量
        normed_embeddings = embeddings.copy()
        faiss.normalize_L2(normed_embeddings)

        # 2. 初始化聚类所需的数据结构
        labels = -np.ones(n_samples, dtype=int)
        
        # faiss索引只存储每个簇的“种子”向量，用于快速筛选
        index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
        index.hnsw.efSearch = self.ef_search
        
        # 核心数据结构，用于维护动态质心
        index_to_label: Dict[int, int] = {}       # faiss内部索引 -> 自定义簇标签
        cluster_centroids: Dict[int, np.ndarray] = {} # 簇标签 -> 质心向量
        cluster_members_count: Dict[int, int] = {}    # 簇标签 -> 成员数量

        next_cluster_label = 0

        # 3. 遍历所有文档进行在线聚类
        for i, vec in enumerate(normed_embeddings):
            vec = vec.reshape(1, -1)
            
            # 如果还没有任何簇，创建第一个
            if index.ntotal == 0:
                labels[i] = next_cluster_label
                index.add(vec)
                index_to_label[0] = next_cluster_label
                cluster_centroids[next_cluster_label] = vec.copy()
                cluster_members_count[next_cluster_label] = 1
                next_cluster_label += 1
                continue

            # --- 混合搜索策略 ---
            # 步骤 A: 使用faiss快速筛选出k个候选簇
            k = min(index.ntotal, self.candidate_k)
            _, I_cand = index.search(vec, k)
            
            # 步骤 B: 精确计算与候选簇真实质心的距离
            min_dist_sq = float('inf')
            best_label = -1
            for faiss_idx in I_cand[0]:
                label = index_to_label[faiss_idx]
                centroid = cluster_centroids[label]
                dist_sq = np.sum((vec - centroid) ** 2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_label = label
            
            # 步骤 C: 根据精确距离决策
            if min_dist_sq > self.distance_threshold:
                # 创建新簇
                new_label = next_cluster_label
                labels[i] = new_label
                index.add(vec)
                new_faiss_index = index.ntotal - 1
                index_to_label[new_faiss_index] = new_label
                cluster_centroids[new_label] = vec.copy()
                cluster_members_count[new_label] = 1
                next_cluster_label += 1
            else:
                # 分配到现有簇，并动态更新质心
                labels[i] = best_label
                
                # 增量式更新质心，高效且精确
                old_size = cluster_members_count[best_label]
                old_centroid = cluster_centroids[best_label]
                new_centroid = (old_centroid * old_size + vec) / (old_size + 1)
                
                cluster_centroids[best_label] = new_centroid
                cluster_members_count[best_label] += 1
        
        print(f"Clustering with dynamic centroids created {next_cluster_label} clusters.")
        return labels

    def cluster(self,embeddings: np.ndarray):
        """
        使用带有动态质心更新的在线贪心算法对文档进行聚类。
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

        labels = None
        if self.cluster_algo == "hdbscan":
            labels = self.do_hdbscan(embeddings)
        else:
            labels = self.do_dpmeans(embeddings)
        return labels
        
    def post_processor(self, docs: List[Document], labels: np.ndarray):
        # 4. 根据最终标签聚合文档
        # 按标签分组，提高效率
        clustered_docs_num = 0
        clustered_indices: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label == -1: 
                continue
            clustered_docs_num += 1
            if label not in clustered_indices:
                clustered_indices[label] = []
            clustered_indices[label].append(i)

        final_clusters: List[Document] = []
        for label, member_indices in clustered_indices.items():
            cluster_id = str(uuid.uuid4())
            context_texts = ""
            all_keywords = []
            source_file = None
            related_doc_ids = []

            for idx in member_indices:
                doc = docs[idx] # 使用被打乱顺序后的doc
                doc_id = str(uuid.uuid4())
                doc.metadata["doc_id"] = doc_id
                doc.metadata["cluster_id"] = cluster_id
                # 确保source字段存在且是文件名
                if "source" in doc.metadata and doc.metadata["source"]:
                    doc.metadata["source"] = os.path.basename(doc.metadata["source"])
                    if source_file is None:
                        source_file = doc.metadata["source"]

                related_doc_ids.append(doc_id)
                context_texts += "\n\n" + doc.page_content
                all_keywords.extend(doc.metadata.get("keywords", []))
            
            cluster_doc = Document(
                page_content=self.summary(context_texts.strip()),
                metadata={
                    "doc_id": cluster_id,
                    "cluster_id": cluster_id,
                    "label": int(label),
                    "related_docs": related_doc_ids,
                    "source": source_file,
                    "keywords": self.combined_keywords(all_keywords),
                    "cluster_size": len(member_indices)
                }
            )
            final_clusters.append(cluster_doc)
        print(f"total:{len(docs)},clusted:{clustered_docs_num},cluster num:{len(final_clusters)}")
        return final_clusters


def train(docs: List[Document], embeddings: np.ndarray):

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    # 定义搜索空间
    n_samples, n_features = embeddings.shape

    # 动态设置范围，确保不会越界
    max_umap_dim = min(200, n_features, n_samples - 1)
    max_n_neighbors = min(50, n_samples - 1)

    # 构建合法的搜索空间
    search_space = [
        Integer(2, max(2, min(20, n_samples - 1)), name='min_cluster_size'),
        Integer(1, min(10, n_samples - 1), name='min_samples'),
        Categorical([True, False], name='use_umap'),
        Integer(5, max_umap_dim, name='umap_dim'),
        Integer(5, max_n_neighbors, name='umap_n_neighbors'),
        Real(0.001, 0.5, name='umap_min_dist'),
    ]

    # 假设你有嵌入数据 `embeddings`，和对应的清洗文本列表
    # embeddings: np.ndarray
    # filtered_texts: List[str]

    # 最佳聚类结果缓存
    best_labels = None
    best_score = float("inf")  # 因为是最小化问题
    @use_named_args(search_space)
    def objective(**params):
        print(f"Trying: {params}")
        nonlocal best_labels,best_score
        try:
            clusterer = TextClusterer(**params)
            labels = clusterer.cluster(embeddings)
            results = clusterer.evaluate_clusters(embeddings, labels)

            score = -results["score"]  # 目标函数返回负数，越小越好
            if score < best_score:
                best_score = score
                best_labels = labels
            print(f"score: {results['score']}")
            return score

        except Exception as e:
            print(f"Error: {e}")
            return 1e6  # 失败时返回极大值避免

    # 运行优化
    res = gp_minimize(objective, search_space, n_calls=30, random_state=42)

    # 最佳参数和分数
    print(f"Best parameters: {res.x}")
    print(f"Best score: {-res.fun}")
    # 后处理
    clusterer = TextClusterer()
    final_clusters = clusterer.post_processor(docs,labels=best_labels)

    return final_clusters
    


