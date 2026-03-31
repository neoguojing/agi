import triton_python_backend_utils as pb_utils
import numpy as np
import threading
import time
import base64
import io
import torch
from PIL import Image as PILImage
from pathlib import Path

# ===== 可选：你原有工具 =====
from agi.config import FILE_STORAGE_PATH, log
from agi.utils.common import path_to_preview_url, Timer
from agi.apps.utils import pick_free_device

style = 'style="width: 100%; max-height: 100vh;"'


class TritonPythonModel:
    def initialize(self, args):
        """
        Triton 模型加载时调用
        """
        self.model_repo = args["model_repository"]
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]

        # ==== 配置 ====
        self.model_path = f"{self.model_repo}/{self.model_name}/{self.model_version}/weights"
        self.timeout = 300
        self.save_image = True
        self.file_path = FILE_STORAGE_PATH

        self.n_steps = 2
        self.guidance_scale = 0.0

        self.model = None
        self.device = None
        self.last_used = 0
        self.lock = threading.Lock()

        # 启动自动卸载线程
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()

        log.info("Triton Image2Image initialized")

    # =========================
    # 模型管理
    # =========================
    def _load(self):
        from diffusers import AutoPipelineForImage2Image

        log.info(f"Loading model from {self.model_path}")
        self.device = pick_free_device()

        self.model = AutoPipelineForImage2Image.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )

        self.model = self.model.to(self.device)
        self.model.enable_model_cpu_offload()

    def _get_model(self):
        with self.lock:
            self.last_used = time.time()
            if self.model is None:
                self._load()
            return self.model

    def _unload(self):
        log.info("Unloading model...")
        del self.model
        self.model = None
        torch.cuda.empty_cache()

    def _monitor(self):
        while True:
            time.sleep(30)
            with self.lock:
                if self.model and (time.time() - self.last_used > self.timeout):
                    self._unload()

    # =========================
    # Triton 推理入口
    # =========================
    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                # ===== 1. 获取输入 =====
                prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()[0].decode()
                image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()[0]
                resp_format = pb_utils.get_input_tensor_by_name(request, "FORMAT").as_numpy()[0].decode()

                # ===== 2. bytes → PIL =====
                input_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
                input_image = input_image.resize((512, 512))

                # ===== 3. 推理 =====
                model = self._get_model()

                with Timer():
                    generated_image = model(
                        prompt,
                        image=input_image,
                        num_inference_steps=self.n_steps,
                        strength=0.5,
                        guidance_scale=self.guidance_scale
                    ).images[0]

                # ===== 4. 输出处理 =====
                if resp_format == "b64_json":
                    output = self._to_base64(generated_image)
                else:
                    output = self._save_image(generated_image)

                # ===== 5. 返回 =====
                out_tensor = pb_utils.Tensor(
                    "OUTPUT",
                    np.array([output.encode("utf-8")], dtype=object)
                )

                responses.append(pb_utils.InferenceResponse([out_tensor]))

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )

        return responses

    # =========================
    # 输出处理
    # =========================
    def _save_image(self, image):
        file_name = f"{int(time.time())}.png"
        output_file = Path(self.file_path) / file_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        image.save(output_file)
        return path_to_preview_url(str(output_file))

    def _to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"

    def finalize(self):
        if self.model:
            self._unload()