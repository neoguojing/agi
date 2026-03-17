import json
import asyncio
from typing import Callable, List
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, BaseMessage
from agi.agent.context import AsyncContextBuilder,ContextRenderer

class ContextEngineeringMiddleware(AgentMiddleware):
    def __init__(self, builder: AsyncContextBuilder, renderer: ContextRenderer):
        self.builder = builder
        self.renderer = renderer

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        
        runtime = request.runtime
        # state 包含当前对话历史，供 Provider 使用（如 RAG 检索）
        state = {"messages": request.messages}

        # ===== 1. 异步构建上下文 (并发抓取 Memory/RAG 等) =====
        ctx_data = await self.builder.build(runtime, state)

        # ===== 2. 渲染为消息对象 =====
        # 将 dict 转换为 List[SystemMessage]
        injected_messages = self.renderer.render(ctx_data)

        # ===== 3. 构造新的 Request (避免修改原始 request) =====
        # 策略：[注入的系统消息] + [原始请求中的所有消息]
        # 注意：如果 request.system_message 存在，它通常在 request.messages 的开头
        new_messages = injected_messages + request.messages
        
        # 使用 override 创建新请求，保持 runtime 和 tools 不变
        request = request.override(messages=new_messages)

        # ===== 4. 调试日志 (参考你的 Debug 模式) =====
        self._log_debug_info(request, ctx_data)

        # ===== 5. 执行模型调用 =====
        response = await handler(request)

        # ===== 6. 后置处理：异步更新 Profile (不阻塞响应) =====
        # 假设 updater 已在外部初始化
        if hasattr(runtime, "extra") and "updater" in runtime.extra:
            asyncio.create_task(
                asyncio.shield(
                    runtime.extra["updater"].update(runtime, request.messages, response.content)
                )
            )

        return response

    def _log_debug_info(self, request: ModelRequest, ctx_data: dict):
        """打印最终注入后的上下文状态"""
        debug_info = {
            "injected_keys": list(ctx_data.keys()),
            "total_messages": len(request.messages),
            "final_prompt_preview": [
                {"role": m.type, "content": m.content[:100] + "..." if len(m.content) > 100 else m.content}
                for m in request.messages
            ]
        }
        print("\n" + "🚀" * 15)
        print("CONTEXT ENGINEERING COMPLETE")
        print(json.dumps(debug_info, indent=2, ensure_ascii=False))
        print("🚀" * 15 + "\n")