from deepagents import create_deep_agent
from .models import ModelProvider
# 1. 初始化模型
# 主 Agent 使用强大的 GPT-4o
llm_main = ModelProvider.get_chat_model("openai", "gpt-4o")

# 子 Agent 或 Embedding 使用本地 Ollama (如 llama3 或 nomic-embed-text)
embed_model = ModelProvider.get_embeddings("ollama", "nomic-embed-text")
llm_sub = ModelProvider.get_chat_model("ollama", "llama3:8b")

# 2. 组装 Deep Agent
agent = create_deep_agent(
    name="Hybrid-Deep-Agent",
    model=llm_main, # 传入已实例化的 BaseChatModel
    subagents=[
        {
            "name": "local_worker",
            "model": llm_sub, # 子 Agent 使用本地模型
            "description": "处理不涉及隐私的本地基础任务"
        }
    ],
    # 如果你的 Middleware 需要向量检索，这里传入 embed_model
    # middleware=[VectorSearchMiddleware(embedding=embed_model)] 
)