import base64
from io import BytesIO
from typing import List, Optional

import numpy as np
from PIL import Image

class TritonText2ImageClient:
    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",  # "http" or "grpc"
        model_name: str = "text2image",
    ):
        self.url = url
        self.protocol = protocol
        self.model_name = model_name

        if protocol == "http":
            import tritonclient.http as client
        elif protocol == "grpc":
            import tritonclient.grpc as client
        else:
            raise ValueError("protocol must be http or grpc")

        self.client = client.InferenceServerClient(url=url)
        self._client_lib = client

    # -------------------------
    # 内部工具函数
    # -------------------------
    def _to_bytes_array(self, texts: List[str]):
        return np.array([t.encode("utf-8") for t in texts], dtype=object)

    def _build_inputs(self, prompt, neg_prompt, model):
        inputs = [
            self._client_lib.InferInput("PROMPT", [len(prompt)], "BYTES"),
            self._client_lib.InferInput("NEG_PROMPT", [len(prompt)], "BYTES"),
            self._client_lib.InferInput("MODEL", [len(prompt)], "BYTES"),
        ]

        inputs[0].set_data_from_numpy(self._to_bytes_array(prompt))
        inputs[1].set_data_from_numpy(self._to_bytes_array(neg_prompt))
        inputs[2].set_data_from_numpy(self._to_bytes_array(model))

        return inputs

    def _decode_image(self, image_bytes):
        base64_str = image_bytes.decode("utf-8")
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))

    # -------------------------
    # 对外接口
    # -------------------------
    def generate(
        self,
        prompt: str,
        neg_prompt: str = "",
        model: str = "sdxl",
        return_pil: bool = True,
    ):
        """
        单条生成
        """
        results = self.generate_batch(
            [prompt],
            [neg_prompt],
            [model],
            return_pil=return_pil,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: List[str],
        neg_prompts: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        return_pil: bool = True,
    ):
        """
        批量生成
        """
        batch_size = len(prompts)

        if neg_prompts is None:
            neg_prompts = [""] * batch_size
        if models is None:
            models = ["sdxl"] * batch_size

        inputs = self._build_inputs(prompts, neg_prompts, models)

        outputs = [
            self._client_lib.InferRequestedOutput("IMAGE")
        ]

        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )

        results = response.as_numpy("IMAGE")

        if return_pil:
            return [self._decode_image(r) for r in results]
        else:
            return results

    def save(
        self,
        prompt: str,
        path: str,
        neg_prompt: str = "",
        model: str = "sdxl",
    ):
        """
        直接保存图片
        """
        image = self.generate(prompt, neg_prompt, model)
        image.save(path)
        return path