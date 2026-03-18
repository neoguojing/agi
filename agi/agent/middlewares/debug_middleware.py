from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable
    

class DebugLLMContextMiddleware(AgentMiddleware):
    def __init__(
        self, 
        show_messages: bool = True, 
        show_tools: bool = True, 
        show_state: bool = True,
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

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
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
                
                content = msg.content.strip()
                if len(content) > self.limit:
                    half = self.limit // 2
                    content = f"{content[:half]}\n... [已省略 {len(content)-self.limit} 字] ...\n{content[-half:]}"
                
                print(f"{icon} [{role:^6}] | {content}")

        print(f"{'='*66}\n")

        # 继续执行
        return await handler(request)