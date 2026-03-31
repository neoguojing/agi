import triton_python_backend_utils as pb_utils
import numpy as np
import threading
import time
import os
import io
import base64
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline


MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1440


class TritonPythonModel:
    def initialize(self, args):
        self.model_repo = args["model_repository"]
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]

        self.timeout = 300

        self.model = None
        self.current_model_name = None
        self.device = "cuda"

        self.lock = threading.RLock()
        self.last_used = 0

        self.file_path = "/tmp/generated"

        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor, daemon=True
        )
        self.monitor_thread.start()

    # =========================
    # 模型加载
    # =========================
    def _load(self, model_name):
        print(f"[Triton] Loading model: {model_name}")

        if model_name == "sdxl":
            path = os.path.join(self.model_repo, "sdxl-turbo")
            self.model = AutoPipelineForText2Image.from_pretrained(
                path, torch_dtype=torch.float16
            )

        elif model_name == "sd3.5":
            path = os.path.join(self.model_repo, "sd3.5")
            self.model = StableDiffusion3Pipeline.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

        self.model = self.model.to(self.device)
        self.model.enable_model_cpu_offload()

        self.current_model_name = model_name

    def _unload(self):
        if self.model is None:
            return

        print("[Triton] Unloading model")

        try:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Unload error: {e}")

    def _get_model(self, model_name):
        with self.lock:
            self.last_used = time.time()

            if self.model is None:
                self._load(model_name)

            elif self.current_model_name != model_name:
                self._unload()
                self._load(model_name)

            return self.model

    # =========================
    # 推理
    # =========================
    def execute(self, requests):
        responses = []

        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            neg_tensor = pb_utils.get_input_tensor_by_name(request, "NEG_PROMPT")
            model_tensor = pb_utils.get_input_tensor_by_name(request, "MODEL")

            prompts = prompt_tensor.as_numpy()
            negs = neg_tensor.as_numpy()
            models = model_tensor.as_numpy()

            results = []

            for i, _ in enumerate(prompts):
                prompt = prompts[i].decode("utf-8")
                neg = negs[i].decode("utf-8")
                model_name = models[i].decode("utf-8")

                try:
                    model = self._get_model(model_name)

                    if model_name == "sdxl":
                        steps = 1
                        guidance = 0.0
                    else:
                        steps = 20
                        guidance = 4.5

                    image = model(
                        prompt=prompt,
                        negative_prompt=neg,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        width=512,
                        height=512,
                    ).images[0]

                    result = self._to_base64(image)

                except Exception as e:
                    result = f"ERROR: {str(e)}"

                results.append(result)

            out_tensor = pb_utils.Tensor(
                "IMAGE",
                np.array(results, dtype=np.object_)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_tensor]
                )
            )

        return responses

    # =========================
    # 工具函数
    # =========================
    def _to_base64(self, image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()

    # =========================
    # 自动卸载
    # =========================
    def _monitor(self):
        while not self._stop_event.is_set():
            time.sleep(30)

            with self.lock:
                if (
                    self.model is not None
                    and time.time() - self.last_used > self.timeout
                ):
                    self._unload()

    def finalize(self):
        self._stop_event.set()
        self.monitor_thread.join(timeout=1)

        with self.lock:
            self._unload()