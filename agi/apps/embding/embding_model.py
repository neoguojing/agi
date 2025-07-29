import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import threading
import time

class QwenEmbedding:
    def __init__(self,model_path: str=None, timeout: int = 300):
        self.max_length = 8192

        self.model_path = model_path
        self.timeout = timeout
        self.model = None
        self.tokenizer = None

        self.last_used = 0
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def get_model(self):
        """访问模型，如果未加载则自动加载"""
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            return self.model

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
        self.model = AutoModel.from_pretrained(self.model_path)

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed_query(self, query: str):
        self.get_model()
        task = "Given a web search query, retrieve relevant passages that answer the query"
        detailed_query = f"Instruct: {task}\nQuery:{query}"
        encoded = self.tokenizer(
            [detailed_query],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**encoded)
            emb = self.last_token_pool(output.last_hidden_state, encoded['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            return emb.squeeze(0).tolist()
        
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
