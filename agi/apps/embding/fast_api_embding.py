from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union,Required,Iterable
from langchain_ollama import OllamaEmbeddings
from agi.config import RAG_EMBEDDING_MODEL,OLLAMA_API_BASE_URL,RAG_EMBEDDING_MODEL_PATH,RAG_RERANK_MODEL_PATH
from agi.apps.embding.embding_model import QwenEmbedding
from agi.apps.embding.rerank import Reranker

app = FastAPI(
    title="AGI embding API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
    # lifespan=lifespan
)

ollama_embding = OllamaEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            base_url=OLLAMA_API_BASE_URL,
        )

qwen_embding = QwenEmbedding(model_path=RAG_EMBEDDING_MODEL_PATH)
rerank = Reranker(model_path=RAG_RERANK_MODEL_PATH)

# 定义请求体模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    model: Optional[str] = "qwen"
    dimensions: Optional[int] = 1024
    encoding_format: Optional[Literal["float", "base64"]]
    user: Optional[str] = "default"

@app.post("/v1/embeddings",summary="文本向量")
async def get_embedding(request: EmbeddingRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    # 生成嵌入向量
    embedding = None
    if request.model == "qwen":
        embedding = qwen_embding.embed_query(request.input,request.dimensions)
    else:
        embedding = ollama_embding.embed_query(request.input)
    return {
        "object": "list",
        "data": [
            {
            "object": "embedding",
            "embedding": embedding,
            "index": 0
            }
        ],
        "model": "bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }


class RerankItem(BaseModel):
    object: Literal["rerank"]
    index: int
    document: str
    score: float

class RerankResponse(BaseModel):
    object: Literal["list"]
    data: List[RerankItem]
    model: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: Optional[str] = "bge-reranker-base"
    top_k: Optional[int] = None  # 默认为返回所有

@app.post("/v1/rerank", summary="排序文档", response_model=RerankResponse)
async def rerank_api(request: RerankRequest):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query and documents cannot be empty.")

    # 加载 reranker（比如你已有的 qwen_reranker, ollama_reranker）
    if request.model == "qwen":
        scores = rerank.rerank(request.query, request.documents)
    else:
        scores = rerank.rerank(request.query, request.documents)

    # 排序 & 截取 top_n
    results = sorted(
        zip(request.documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:request.top_n]

    # 构造响应
    return {
        "object": "list",
        "data": [
            {
                "object": "rerank",
                "index": i,
                "document": doc,
                "score": round(score, 4)
            }
            for i, (doc, score) in enumerate(results)
        ],
        "model": request.model
    }

