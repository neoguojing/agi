from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable, Awaitable
    

class DebugLLMContextMiddleware(AgentMiddleware):
    def __init__(
        self, 
        show_messages: bool = True, 
        show_tools: bool = True, 
        show_state: bool = False,
        show_settings: bool = False,
        content_limit: int = 300
    ):
        """
        :param show_messages: 是否显示对话消息流
        :param show_tools: 是否显示工具定义及 tool_choice
        :param show_state: 是否显示 AgentState (可能非常大)
        :param show_settings: 是否显示模型配置 (temperature, stop sequences 等)
        :param content_limit: 消息内容截断长度
        """
        self.show_messages = show_messages
        self.show_tools = show_tools
        self.show_state = show_state
        self.show_settings = show_settings
        self.limit = content_limit

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:
        
        print(f"\n{'='*25} 🛠️ LLM DEBUG 仪表盘 {'='*25}")

        # 1. 基础模型信息
        model_name = getattr(request.model, "model_name", str(request.model))
        print(f"【模型】: {model_name}")

        # 2. 模型设置与工具策略 (开关控制)
        if self.show_settings and request.model_settings:
            print(f"【配置】: {request.model_settings}")
        
        if self.show_tools:
            t_names = [t.name if hasattr(t, 'name') else str(t) for t in request.tools]
            print(f"【工具】: {', '.join(t_names) if t_names else '无'}")
            if request.tool_choice:
                print(f"【策略】: tool_choice = {request.tool_choice}")

        # 3. Agent 内部状态 (开关控制)
        if self.show_state and request.state:
            print(f"【状态】: {request.state}")

        print("-" * 66)

        # 4. 消息流解析
        if self.show_messages:
            # 合并 SystemMessage 和普通消息列表进行展示
            all_msgs = []
            if request.system_message:
                all_msgs.append(request.system_message)
            all_msgs.extend(request.messages)

            for i, msg in enumerate(all_msgs):
                role = msg.type.upper()
                icon = {"SYSTEM": "⚙️", "HUMAN": "👤", "AI": "🤖", "TOOL": "🛠️"}.get(role, "📝")
                
                # --- ✅ 修复开始：安全处理 content ---
                raw_content = msg.content
                content_str = ""

                if isinstance(raw_content, str):
                    content_str = raw_content.strip()
                elif isinstance(raw_content, list):
                    # 如果 content 是列表 (如多模态或工具参数)，尝试提取文本
                    text_parts = []
                    for item in raw_content:
                        if isinstance(item, dict):
                            # 处理 [{"type": "text", "text": "..."}] 格式
                            if item.get("type") == "text":
                                text_parts.append(str(item.get("text", "")))
                            else:
                                # 非文本部分简要标记
                                text_parts.append(f"[{item.get('type', 'unknown')}]")
                        else:
                            text_parts.append(str(item))
                    content_str = "\n".join(text_parts).strip()
                else:
                    # 其他类型直接转字符串
                    content_str = str(raw_content).strip()
                # --- ✅ 修复结束 ---

                # 截断长内容
                if len(content_str) > self.limit:
                    half = self.limit // 2
                    content_str = f"{content_str[:half]}\n... [已省略 {len(content_str)-self.limit} 字] ...\n{content_str[-half:]}"
                
                # 如果内容为空，显示占位符
                if not content_str:
                    content_str = "[无内容]"

                print(f"{icon} [{role:^6}] | {content_str}")

        print(f"{'='*66}\n")

        # 继续执行
        response = await handler(request)
        return response