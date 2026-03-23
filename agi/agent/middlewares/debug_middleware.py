from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable, Awaitable
from typing import List
from langchain_core.messages import BaseMessage
    

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
        request.messages = deduplicate_messages_by_id(request.messages)  # 去重，防止重复消息干扰调试
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
    
    from langchain_core.messages import BaseMessage

def deduplicate_messages_by_id(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    根据 LangChain 消息对象的 .id 属性进行去重。
    保留列表中第一次出现的消息，移除后续 ID 重复的消息。
    """
    seen_ids = set()
    unique_messages = []
    
    for msg in messages:
        # 直接访问顶层 id 属性
        # 在你提供的示例中：msg.id 如 'e72247b6-26b7-4bed-ae84-3a75433b686e'
        msg_id = msg.id
        
        if msg_id is None:
            # 如果某些消息没有 ID (极少见，除非手动构造未初始化)，
            # 策略：视为唯一消息保留，或者根据需求跳过。这里选择保留。
            unique_messages.append(msg)
            continue
        
        if msg_id not in seen_ids:
            seen_ids.add(msg_id)
            unique_messages.append(msg)
        else:
            # 调试信息：如果发现重复，可以打印出来
            # print(f"Skipping duplicate message with ID: {msg_id}")
            pass
            
    return unique_messages