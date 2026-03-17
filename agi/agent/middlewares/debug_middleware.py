from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable
import json


class DebugLLMContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:

        # ===== 1. system prompt =====
        system_msg = None
        if request.system_message:
            system_msg = request.system_message.content

        # ===== 2. messages =====
        messages = []
        for msg in request.messages:
            m_info = {
                "type": msg.__class__.__name__,
                "content": getattr(msg, "content", None),
            }
            # 如果是 AI 发出的工具调用，记录下来
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                m_info["tool_calls"] = msg.tool_calls
            # 如果是工具回复的消息，记录 ID
            if hasattr(msg, "tool_call_id"):
                m_info["tool_call_id"] = msg.tool_call_id
            messages.append(m_info)

        # ===== 3. tools =====
        tools = []
        for t in request.tools:
            if isinstance(t, dict):
                tools.append(t.get("name", "builtin_tool"))
            else:
                tools.append(getattr(t, "name", str(t)))

        # ===== 4. model info =====
        model_name = getattr(request.model, "model_name", str(request.model))

        debug_context = {
            "model": model_name,
            "system_message": system_msg,
            "messages": messages,
            "tools": tools,
            "tool_count": len(tools),
        }

        print("\n" + "=" * 30)
        print("🔥 LLM FINAL CONTEXT")
        print(json.dumps(debug_context, indent=2, ensure_ascii=False))
        print("=" * 30 + "\n")

        # ===== 执行模型 =====
        return handler(request)