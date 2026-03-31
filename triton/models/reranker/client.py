import numpy as np

class TritonRerankerClient:
    def __init__(self, url="localhost:8000", protocol="http", model_name="reranker"):
        """
        url:
            http -> localhost:8000
            grpc -> localhost:8001
        protocol: "http" or "grpc"
        """
        self.protocol = protocol
        self.model_name = model_name

        if protocol == "http":
            import tritonclient.http as httpclient
            self.client = httpclient.InferenceServerClient(url=url)
            self.http = True
        else:
            import tritonclient.grpc as grpcclient
            self.client = grpcclient.InferenceServerClient(url=url)
            self.http = False

    def _create_input(self, name, data):
        if self.http:
            import tritonclient.http as httpclient
            inp = httpclient.InferInput(name, data.shape, "BYTES" if data.dtype == object else "FP32")
        else:
            import tritonclient.grpc as grpcclient
            inp = grpcclient.InferInput(name, data.shape, "BYTES" if data.dtype == object else "FP32")

        inp.set_data_from_numpy(data)
        return inp

    def rerank(self, queries, documents, model="qwen"):
        """
        queries: List[str]
        documents: List[str]
        model: "qwen" | "bge"
        """
        assert len(queries) == len(documents)

        # 转 numpy（Triton string 必须是 object + bytes）
        queries_np = np.array([q.encode("utf-8") for q in queries], dtype=object)
        docs_np = np.array([d.encode("utf-8") for d in documents], dtype=object)
        model_np = np.array([model.encode("utf-8")] * len(queries), dtype=object)

        inputs = [
            self._create_input("QUERIES", queries_np),
            self._create_input("DOCUMENTS", docs_np),
            self._create_input("MODEL", model_np),
        ]

        if self.http:
            response = self.client.infer(self.model_name, inputs)
            scores = response.as_numpy("SCORES")
        else:
            response = self.client.infer(self.model_name, inputs)
            scores = response.as_numpy("SCORES")

        return scores.flatten().tolist()