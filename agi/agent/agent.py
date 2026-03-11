from deepagents import create_deep_agent
from .models import ModelProvider
from langchain_core.messages import SystemMessage

# 1. 初始化模型
# 主 Agent 使用强大的 GPT-4o
llm_main = ModelProvider.get_chat_model("openai", "gpt-4o")

# 子 Agent 或 Embedding 使用本地 Ollama (如 llama3 或 nomic-embed-text)
embed_model = ModelProvider.get_embeddings("ollama", "nomic-embed-text")
llm_sub = ModelProvider.get_chat_model("ollama", "llama3:8b")

# 1. 定义特化编码子 Agent 规范
coding_expert = {
    "name": "developer",
    "description": "用于执行复杂的本地代码编写、重构和测试任务",
    "system_prompt": "你是一个遵循 Claude Code 哲学的工程师。先观察，后编码，必须运行测试验证。",
    "skills": ["/skills/coding/python_best_practices.md"],
    "middleware": [
        # 为编码 Agent 绑定隔离的 Docker 后端
        FilesystemMiddleware(backend=DockerSafeBackend()) 
    ]
}

# 2. 定义多模态与搜索子 Agent
utility_worker = {
    "name": "utility",
    "description": "处理图片生成、语音识别、Web 检索和知识库查询",
    "tools": [web_search, image_gen, speech_to_text, rag_retrieval]
}

# 3. 创建主 Agent (Orchestrator)
orchestrator = create_deep_agent(
    model="anthropic:claude-3-5-sonnet-20240620",
    name="Deep-Orchestrator",
    system_prompt=SystemMessage(content="你是项目的中枢。利用黑板协调子 Agent。"),
    
    # 核心中间件
    memory=["/memory/PROJECT_CONTEXT.md"], # 长记忆
    subagents=[coding_expert, utility_worker], # 注册子 Agent
    
    # 热插拔旁路系统
    middleware=[
        HotSwapMiddleware(registry),  # 实时更新工具/技能
        DynamicToolMiddleware(retriever) # 渐进式披露工具
    ],
    
    # 人工干预
    interrupt_on={"edit_file": True, "execute": True},
    checkpointer=my_checkpointer, # 状态持久化
    store=my_blackboard_store      # 共享黑板存储
)