from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware import wrap_model_call
from langchain.tools import tool, ToolRuntime
import asyncio


@dataclass
class Context:
    user_id: str
    role: str



@tool
def get_user_profile(runtime: ToolRuntime[Context]) -> str:
    """Get user profile info"""

    user_id = runtime.context.user_id
    store = runtime.store

    profile = store.get(("users",), user_id)

    if not profile:
        return "No profile found"

    return f"User name: {profile.value['name']}"


@dynamic_prompt
def role_prompt(request):
    role = request.runtime.context.role

    if role == "admin":
        return "You are an admin assistant."
    else:
        return "You are a normal assistant."
    

@wrap_model_call
def inject_user_memory(request, handler):

    user_id = request.runtime.context.user_id
    store = request.runtime.store

    memory = store.get(("memory",), user_id)

    if memory:
        messages = [
            *request.messages,
            {
                "role": "system",
                "content": f"User preference: {memory.value}"
            }
        ]
        request = request.override(messages=messages)

    return handler(request)


class ContextProvider:
    def load(self, runtime, state) -> dict:
        ...

class LongMemoryProvider(ContextProvider):
    def __init__(self, store):
        self.store = store

    def load(self, runtime, state):
        user_id = runtime.context.user_id
        memory = self.store.get(("memory",), user_id)

        return {
            "long_memory": memory.value if memory else None
        }
    
class UserProfileProvider(ContextProvider):
    def load(self, runtime, state):
        return {
            "role": runtime.context.role,
            "user_id": runtime.context.user_id
        }
    
class KnowledgeProvider(ContextProvider):
    def __init__(self, retriever):
        self.retriever = retriever

    def load(self, runtime, state):
        query = state["messages"][-1].content

        docs = self.retriever.retrieve(query)

        return {
            "knowledge": docs
        }
    
class ToolContextProvider(ContextProvider):
    def load(self, runtime, state):
        return {
            "tools": runtime.store.get(("tools",), "all")
        }
    


class AsyncContextBuilder:
    def __init__(self, providers):
        self.providers = providers

    async def build(self, runtime, state):
        # 并行执行所有 Provider 的 load 方法
        tasks = [p.load(runtime, state) for p in self.providers]
        results = await asyncio.gather(*tasks)
        
        context = {}
        for r in results:
            context.update(r)
        return context
    

@wrap_model_call
async def context_engineering(request, handler):

    runtime = request.runtime
    state = {"messages": request.messages}

    ctx = builder.build(runtime, state)

    # 👇 构建最终 messages
    messages = request.messages

    if ctx.get("long_memory"):
        messages.append({
            "role": "system",
            "content": f"Memory: {ctx['long_memory']}"
        })

    if ctx.get("knowledge"):
        messages.append({
            "role": "system",
            "content": f"Knowledge: {ctx['knowledge']}"
        })

    request = request.override(messages=messages)

    response = await handler(request)
    
    # --- 后置：更新逻辑 (Fire and Forget) ---
    # 提取最新的对话内容，触发异步更新
    updater = ProfileUpdater(request.runtime.extra["extractor"])
    await updater.update(
        request.runtime, 
        request.messages, 
        response.content
    )


class ProfileUpdater:
    def __init__(self, extractor_model):
        self.extractor_model = extractor_model

    async def update(self, runtime, messages, ai_response):
        """
        从对话中异步提取用户偏好并存入 Store
        """
        user_id = runtime.context.user_id
        store = runtime.store
        
        # 1. 构造提取 Prompt (仅当 AI 回复包含有价值信息时)
        extraction_prompt = f"""
        Based on the latest user message and AI response, extract user preferences or traits.
        User: {messages[-1]['content']}
        AI: {ai_response}
        Return JSON format.
        """
        
        # 假设这里调用了一个轻量级模型进行提取
        new_traits = await self.extractor_model.apredict(extraction_prompt)
        
        if new_traits:
            # 2. 获取旧记忆并合并
            existing = store.get(("memory",), user_id)
            updated_memory = self._merge(existing.value if existing else {}, new_traits)
            
            # 3. 写回 Store
            store.put(("memory",), user_id, updated_memory)

    def _merge(self, old, new):
        # 简单的字典合并逻辑
        return {**old, **new}