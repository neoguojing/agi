from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union,Required,Iterable
from langchain_ollama import OllamaEmbeddings
from agi.config import OLLAMA_API_BASE_URL,RAG_EMBEDDING_MODEL_PATH,RAG_RERANK_MODEL_PATH,log
from agi.apps.embding.embding_model import QwenEmbedding
from agi.apps.embding.rerank import Reranker
import time
app = FastAPI(
    title="AGI embding API",
    description="兼容 OpenAI API 的 AGI 接口",
    version="1.0.0",
    # lifespan=lifespan
)

ollama_embding = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url=OLLAMA_API_BASE_URL,
        )

qwen_embding = QwenEmbedding(model_path=RAG_EMBEDDING_MODEL_PATH)
rerank = Reranker(model_path=RAG_RERANK_MODEL_PATH)

# 请求体模型
class OllamaEmbedRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    truncate: Optional[bool] = True
    options: Optional[dict] = None
    keep_alive: Optional[str] = "5m"

# 响应体模型
class OllamaEmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = 0
    load_duration: Optional[int] = 0
    prompt_eval_count: Optional[int] = 0

# 接口定义
@app.post("/api/embed", summary="Ollama-compatible embedding API", response_model=OllamaEmbedResponse)
async def embed(request: OllamaEmbedRequest):
    log.info(f"Received embed request: {request}")

    start_time = time.time()

    # 准备输入
    inputs = request.input if isinstance(request.input, list) else [request.input]

    # 加载嵌入模型（可记录 load_duration）
    try:
        if request.model.lower() == "qwen":
            embeddings = [qwen_embding.embed_query(text, 1024) for text in inputs]
        else:
            embeddings = [ollama_embding.embed_query(text) for text in inputs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    total_duration = int((time.time() - start_time) * 1_000_000)  # 微秒

    return OllamaEmbedResponse(
        model=request.model,
        embeddings=embeddings,
        total_duration=total_duration,
        load_duration=total_duration,
        prompt_eval_count=len(inputs)
    )


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
    top_k: Optional[int] = 3  # 默认为返回所有

@app.post("/v1/rerank", summary="排序文档", response_model=RerankResponse)
async def rerank_api(request: RerankRequest):
    if not request.query or not request.documents:
        raise HTTPException(status_code=400, detail="Query and documents cannot be empty.")
    log.info(request)
    # 加载 reranker（比如你已有的 qwen_reranker, ollama_reranker）
    if request.model == "qwen":
        queries = [request.query] * len(request.documents)
        scores = rerank.rerank(queries, request.documents)
    else:
        scores = rerank.rerank(request.query, request.documents)

    # 排序 & 截取 top_n
    results = sorted(
        zip(request.documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:request.top_k]

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

