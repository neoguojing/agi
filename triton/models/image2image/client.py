import base64
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


class TritonImage2ImageClient:
    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",  # "http" or "grpc"
        model_name: str = "image2image",
        timeout: Optional[int] = None,
    ):
        self.url = url
        self.protocol = protocol
        self.model_name = model_name
        self.timeout = timeout

        if protocol == "http":
            import tritonclient.http as client
        elif protocol == "grpc":
            import tritonclient.grpc as client
        else:
            raise ValueError("protocol must be http or grpc")

        self.client = client.InferenceServerClient(url=url)
        self._client_lib = client

    # -------------------------
    # 工具函数
    # -------------------------
    def _image_to_numpy(self, image: Image.Image) -> np.ndarray:
        """
        PIL → uint8 bytes（Triton 输入）
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return np.frombuffer(image_bytes, dtype=np.uint8)

    def _file_to_numpy(self, image_path: str) -> np.ndarray:
        """
        文件 → uint8 bytes
        """
        with open(image_path, "rb") as f:
            return np.frombuffer(f.read(), dtype=np.uint8)

    def _decode_base64_image(self, b64_str: str) -> Image.Image:
        """
        base64 → PIL
        """
        if b64_str.startswith("data:image"):
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))

    def _to_bytes_array(self, text: str):
        return np.array([text.encode("utf-8")], dtype=object)

    # -------------------------
    # 核心接口
    # -------------------------
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        return_pil: bool = True,
    ):
        """
        单张 image2image
        """
        if image is None and image_path is None:
            raise ValueError("image or image_path must be provided")

        if image is not None:
            image_np = self._image_to_numpy(image)
        else:
            image_np = self._file_to_numpy(image_path)

        # Triton 输入必须带 batch 维度
        image_np = np.expand_dims(image_np, axis=0)

        inputs = [
            self._client_lib.InferInput("PROMPT", [1], "BYTES"),
            self._client_lib.InferInput("IMAGE", image_np.shape, "UINT8"),
            self._client_lib.InferInput("FORMAT", [1], "BYTES"),
        ]

        inputs[0].set_data_from_numpy(self._to_bytes_array(prompt))
        inputs[1].set_data_from_numpy(image_np)
        inputs[2].set_data_from_numpy(self._to_bytes_array(
            "b64_json" if return_pil else "url"
        ))

        outputs = [
            self._client_lib.InferRequestedOutput("OUTPUT")
        ]

        response = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self.timeout,
        )

        result = response.as_numpy("OUTPUT")[0].decode("utf-8")

        # 返回 PIL
        if return_pil:
            return self._decode_base64_image(result)

        return result

    # -------------------------
    # 便捷接口
    # -------------------------
    def generate_from_path(
        self,
        prompt: str,
        image_path: str,
        save_path: Optional[str] = None,
    ):
        img = self.generate(prompt, image_path=image_path, return_pil=True)

        if save_path:
            img.save(save_path)

        return img

    def generate_and_save(
        self,
        prompt: str,
        image: Image.Image,
        save_path: str,
    ):
        img = self.generate(prompt, image=image, return_pil=True)
        img.save(save_path)
        return save_path