import triton_python_backend_utils as pb_utils
import torch
import os
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

MAX_LENGTH = 1024
TIMEOUT = 300

class TritonPythonModel:
    def initialize(self, args):
        """Triton Python 后端初始化"""
        # Triton 默认参数
        self.model_repo = args["model_repository"]   # 模型仓库路径
        self.model_name = args["model_name"]        # 模型名称
        self.model_version = args["model_version"]  # 模型版本号

        # Triton 会把每个模型放在 model_repository/model_name/version 目录
        self.model_base_path = os.path.join(self.model_repo, self.model_name, self.model_version)

        # 状态初始化
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_used = 0

        # 后台线程自动卸载空闲模型
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

    def _monitor(self):
        while True:
            import time
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > TIMEOUT):
                    self._unload()

    def _unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()

    def _load_model(self, model_type="qwen"):
        """按需加载模型"""
        with self.lock:
            self.last_used = 0
            if self.model is not None and self.model_type == model_type:
                return

            self._unload()
            self.model_type = model_type

            if model_type == "qwen":
                path = os.path.join(self.model_base_path, "Qwen3-Reranker-0.6B")
                self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    device_map={"": 0},  # 可以改成 pick_free_device()
                    attn_implementation="flash_attention_2"
                ).to(self.device).eval()
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            else:
                path = os.path.join(self.model_base_path, "bge-reranker-v2-m3")
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()