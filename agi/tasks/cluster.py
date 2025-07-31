import faiss
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import os
import numpy as np
from langchain_core.documents import Document
from collections import defaultdict
import uuid
from typing import List,Dict
from agi.config import log
from agi.tasks.utils import get_last_message_text,split_think_content,graph_print
from langchain.prompts import ChatPromptTemplate
from agi.tasks.task_factory import (
    TaskFactory
)
import random
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
    def __init__(self, hnsw_m=32, ef_search=128, distance_threshold=0.5):
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
        

    def summary(self,text:str):
        value = summary_template.invoke({"text":f'{text} /no_think'})
        ai = self.llm.invoke(value)
        _, result = split_think_content(ai.content)

        return result

    def combined_keywords(self,all_keywords:list):
        # 3.3 加权统计关键词
        keyword_weights = defaultdict(float)
        for kw, weight in all_keywords:
            keyword_weights[kw] += weight

        # 3.4 选择top-k关键词
        sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
        combined_keywords = [(kw,weight) for kw, weight in sorted_keywords[:self.cluster_top_k_keywords]]
        return combined_keywords

    def cluster(self, docs: List[Document], embeddings: np.ndarray) -> List[Document]:
        """
        使用带有动态质心更新的在线贪心算法对文档进行聚类。
        """
        import pdb;pdb.set_trace()
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

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
                    "label": label,
                    "related_docs": related_doc_ids,
                    "source": source_file,
                    "keywords": self.combined_keywords(all_keywords),
                    "cluster_size": len(member_indices)
                }
            )
            final_clusters.append(cluster_doc)
        print(f"total:{len(docs)},clusted:{clustered_docs_num},cluster num:{len(final_clusters)}")
        result = self.evaluate_clusters(embeddings,labels=labels)
        print(f"evaluate_clusters:{result}")
        return final_clusters
    
    def evaluate_clusters(self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sample_size: int = 10000,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        对聚类结果做内部指标评估。

        Args:
            embeddings: (n_samples, n_features) 原始或降维后的嵌入矩阵，dtype=float32/64
            labels:     (n_samples,) 每个样本的簇标签，-1（若有噪声簇）也会参与计算
            sample_size: 最大采样数，防止大规模数据计算过慢。
            random_state: 采样和轮廓系数的随机种子。

        Returns:
            dict:
                silhouette    -- 轮廓系数（[-1,1]，越大越好）
                davies_bouldin -- DB 指数（[0,∞)，越小越好）
                calinski_harabasz -- CH 指数（[0,∞)，越大越好）
        """
        n_samples = labels.shape[0]
        # 如果样本数过大，随机采样
        if n_samples > sample_size:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            emb_samp = embeddings[idx]
            lab_samp = labels[idx]
        else:
            emb_samp = embeddings
            lab_samp = labels

        results = {}
        # 轮廓系数需要至少 2 个簇，且每个簇至少 1 个样本
        unique_labels = set(lab_samp)
        if len(unique_labels) >= 2:
            results["silhouette"] = silhouette_score(
                emb_samp, lab_samp, metric="euclidean", random_state=random_state
            )
        else:
            results["silhouette"] = float("nan")

        # Davies–Bouldin 指数，同样至少 2 个簇
        if len(unique_labels) >= 2:
            results["davies_bouldin"] = davies_bouldin_score(emb_samp, lab_samp)
        else:
            results["davies_bouldin"] = float("nan")

        # Calinski–Harabasz 指数，也需要至少 2 个簇
        if len(unique_labels) >= 2:
            results["calinski_harabasz"] = calinski_harabasz_score(emb_samp, lab_samp)
        else:
            results["calinski_harabasz"] = float("nan")

        return results


