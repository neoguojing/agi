import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import threading
import time

class Reranker:
    def __init__(self,model_path: str=None, timeout: int = 300, device=None, max_length=8192):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.timeout = timeout
        self.model = None
        self.tokenizer = None

        self.last_used = 0
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

        self.max_length = max_length
        
        self.token_false_id = None
        self.token_true_id = None
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = None
        self.suffix_tokens = None

    def get_model(self):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            return self.model

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        # 如果需要使用加速或者混合精度，可以在这里解注释：
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        ).to(self.device).eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        if not instruction:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        self.get_model()
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=True,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs):
        self.get_model()

        outputs = self.model(**inputs)
        import pdb;pdb.set_trace()
        logits = outputs.logits[:, -1, :]
        # 解码预测的 token（logits 最大值的 token）
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = [self.tokenizer.decode([token_id]) for token_id in predicted_ids]
        print("Predicted tokens:", predicted_tokens)
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, queries, documents, instruction=None):
        """
        输入：
            queries: List[str] 查询列表
            documents: List[str] 文档列表（与queries对应）
            instruction: 任务说明，默认取内置提示
        返回：
            scores: List[float] 每个query-doc对的相关度分数，范围0~1
        """
        assert len(queries) == len(documents), "Queries and documents must have the same length."
        pairs = [self.format_instruction(instruction, q, d) for q, d in zip(queries, documents)]
        print(pairs)
        inputs = self.process_inputs(pairs)
        print(inputs)
        scores = self.compute_logits(inputs)
        return scores
    
    def rerank_topk(
        self,
        queries: List[str],
        candidates: List[List[str]],
        instruction: str = None,
        top_k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """
        对每个 query 对应的候选文档列表做 rerank，返回按分数排序的 Top-k 结果。
        
        输入：
            queries: List[str]，多个查询
            candidates: List[List[str]]，与 queries 对应的候选文档列表
            instruction: rerank任务描述，可选
            top_k: 返回每个query的top_k条结果

        返回：
            List[List[(文档, 分数)]]，每个query对应的top_k个文档及其分数
        """
        assert len(queries) == len(candidates), "Queries and candidates length mismatch."

        all_results = []
        for query, docs in zip(queries, candidates):
            if not docs:
                all_results.append([])
                continue

            # 构建格式化输入对
            pairs = [self.format_instruction(instruction, query, doc) for doc in docs]
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)

            # 添加原始索引
            indexed_docs = list(enumerate(docs))  # [(0, doc0), (1, doc1), ...]
            scored_docs = [(idx, doc, score) for (idx, doc), score in zip(indexed_docs, scores)]

            # 排序并取 top-k
            sorted_docs = sorted(scored_docs, key=lambda x: x[2], reverse=True)
            top_docs = sorted_docs[:top_k]
            all_results.append(top_docs)

        return all_results
    
    def _unload(self):
        print(f"[Model] Unloading model from {self.model_path}")
        del self.model
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    def _monitor(self):
        """后台线程定期检查是否应卸载模型"""
        while True:
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > self.timeout):
                    self._unload()

