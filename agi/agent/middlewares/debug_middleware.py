from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable, Awaitable
from typing import List,Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import AgentMiddleware
from langgraph.prebuilt.tool_node import ToolCallRequest
import json
import time

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
        
        print(f"{len(request.messages)}")
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
        request.messages = deduplicate_messages_by_content_pairs(request.messages)  # 去重，防止重复消息干扰调试
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
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        # 1. 提取关键信息
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "unknown")
        tool_id = tool_call.get("id", "no-id")
        args = tool_call.get("args", {})
        
        # 格式化参数以便打印 (限制长度防止日志爆炸)
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "... [truncated]"

        print("\n" + "="*60)
        print(f"🛠️  [TOOL START] {tool_name}")
        print(f"   ID: {tool_id}")
        print(f"   Args: {args_str}")
        print("-" * 60)

        # 2. 记录开始时间
        start_time = time.perf_counter()
        
        try:
            # 3. 执行真正的工具调用 (调用 handler)
            result = await handler(request)
            
            # 4. 计算耗时
            duration = time.perf_counter() - start_time
            
            # 5. 处理结果并打印
            content_preview = ""
            if isinstance(result, ToolMessage):
                content_preview = str(result.content)
                if len(content_preview) > 150:
                    content_preview = content_preview[:150] + "... [truncated]"
                status_icon = "✅"
            elif isinstance(result, Command):
                content_preview = f"Command(goto={result.goto})"
                status_icon = "🔀"
            else:
                content_preview = str(result)
                status_icon = "❓"

            print(f"{status_icon} [TOOL END]   {tool_name}")
            print(f"   Duration: {duration:.4f}s")
            print(f"   Result Preview: {content_preview}")
            print("="*60 + "\n")
            
            return result

        except Exception as e:
            # 6. 异常处理
            duration = time.perf_counter() - start_time
            print(f"❌ [TOOL ERROR]  {tool_name}")
            print(f"   Duration: {duration:.4f}s")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Error Msg:  {str(e)}")
            # 可选：打印堆栈跟踪 (生产环境建议关闭或仅记录到文件)
            # traceback.print_exc() 
            print("="*60 + "\n")
            
            # 重要：必须重新抛出异常，除非你想在这里吞掉错误
            raise

def deduplicate_messages_by_content_pairs(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    去除连续重复的 (Human, AI) 消息对。
    如果当前的 [Human, AI] 对的内容与上一对 [Human, AI] 完全一致，则移除当前这对。
    
    示例场景:
    Input:  [H: hello1, A: Hi, H: hello1, A: Hi, H: hello2]
    Output: [H: hello1, A: Hi, H: hello2]
    """
    if not messages:
        return []
    
    unique_messages = []
    
    # 我们按步长 2 遍历，尝试提取 (Human, AI) 对
    i = 0
    while i < len(messages):
        # 获取当前潜在的一对消息
        current_human = messages[i]
        current_ai = messages[i+1] if (i + 1) < len(messages) else None
        
        # 检查是否构成完整的一对 (Human -> AI)
        # 注意：这里假设列表顺序严格是 Human, AI, Human, AI...
        # 如果结构混乱，可能需要更复杂的逻辑，但通常对话历史是有序的
        is_pair = (
            isinstance(current_human, HumanMessage) and 
            (current_ai is None or isinstance(current_ai, AIMessage))
        )
        
        if is_pair and current_ai is not None:
            # 如果有上一对消息在结果列表中，进行比较
            if len(unique_messages) >= 2:
                prev_ai = unique_messages[-1]
                prev_human = unique_messages[-2]
                
                # 检查上一对是否也是完整的 Human-AI 对
                if isinstance(prev_human, HumanMessage) and isinstance(prev_ai, AIMessage):
                    # 比较内容 (content)
                    # 使用 str() 或直接 .content 属性，防止 content 为 None 的情况
                    curr_h_content = getattr(current_human, "content", "")
                    curr_a_content = getattr(current_ai, "content", "")
                    prev_h_content = getattr(prev_human, "content", "")
                    prev_a_content = getattr(prev_ai, "content", "")
                    
                    if curr_h_content == prev_h_content and curr_a_content == prev_a_content:
                        # 内容完全重复，跳过当前这对 (不添加到 unique_messages)
                        # print(f"Skipped duplicate pair: Human='{curr_h_content}', AI='{curr_a_content}'")
                        i += 2
                        continue
            
            # 如果不重复，或者这是第一对，添加这两个消息
            unique_messages.append(current_human)
            unique_messages.append(current_ai)
            i += 2
            
        else:
            # 如果不是成对出现（例如最后剩一个 Human 消息，或者顺序错乱）
            # 直接保留当前消息，步进 1
            unique_messages.append(current_human)
            i += 1
            
    return unique_messages