import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
from typing import List, Tuple
import threading
import time
import os
from agi.config import RAG_EMBEDDING_MODEL,MODEL_PATH
class Reranker:
    def __init__(self,timeout: int = 300, device=None, max_length=8192):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = RAG_EMBEDDING_MODEL
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

    def get_model(self,model:str = RAG_EMBEDDING_MODEL):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self.model_name = model
                self._load()
            else:
                if model != self.model_name:
                    self._unload()
                    self.model_name = model
                    self._load()
            return self.model

    def _load(self):
        if self.model_name == "qwen":
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH,"Qwen3-Reranker-0.6B"), padding_side='left')
            # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
            # 如果需要使用加速或者混合精度，可以在这里解注释：
            self.model = AutoModelForCausalLM.from_pretrained(
                os.path.join(MODEL_PATH,"Qwen3-Reranker-0.6B"), torch_dtype=torch.float16, attn_implementation="flash_attention_2"
            ).to(self.device).eval()

            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

            self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH,"bge-reranker-v2-m3"))
            self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_PATH,"bge-reranker-v2-m3")).to(self.device).eval()

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        if not instruction:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
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

        outputs = self.model(**inputs)
        import pdb;pdb.set_trace()
        logits = outputs.logits[:, -1, :]

        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rerank(self, queries, documents, model=RAG_EMBEDDING_MODEL,instruction=None):
        """
        输入：
            queries: List[str] 查询列表
            documents: List[str] 文档列表（与queries对应）
            instruction: 任务说明，默认取内置提示
        返回：
            scores: List[float] 每个query-doc对的相关度分数，范围0~1
        """
        assert len(queries) == len(documents), "Queries and documents must have the same length."
        scores = None
        self.get_model(model)
        if model == "qwen":
            pairs = [self.format_instruction(instruction, q, d) for q, d in zip(queries, documents)]
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
        else:
            pairs = [[q,d] for q, d in zip(queries, documents)]
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
                print(inputs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
        print(scores,type(scores))
        return scores
    
    def _unload(self):
        print(f"[Model] Unloading model {self.model_name}")
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

