import json
import time
import traceback
from typing import Callable, Awaitable, List, Any, Optional
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.types import Command
from langchain.tools.tool_node import ToolCallRequest

class DebugLLMContextMiddleware(AgentMiddleware):
    def __init__(
        self, 
        name: str = "DEFAULT",  # 新增：Namespace/模块名称
        show_messages: bool = True, 
        show_tools: bool = True, 
        show_state: bool = False,
        show_settings: bool = False,
        content_limit: int = 300,
        color_header: str = "\033[95m", # 紫色
        color_reset: str = "\033[0m"
    ):
        self.namespace = name.upper()
        self.show_messages = show_messages
        self.show_tools = show_tools
        self.show_state = show_state
        self.show_settings = show_settings
        self.limit = content_limit
        self.c1 = color_header
        self.reset = color_reset

    def _format_content(self, content: Any,id: Any) -> str:
        """
        统一处理消息内容提取与截断。
        支持：纯文本、多模态列表（Image, File, Audio, Video）。
        """
        if isinstance(content, str):
            res = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                
                # 提取类型
                c_type = item.get("type", "unknown").upper()
                
                if c_type == "TEXT":
                    parts.append(item.get("text", "").strip())
                
                # 处理多媒体/文件类型: image, file, audio, video
                elif c_type in ["IMAGE", "FILE", "AUDIO", "VIDEO"]:
                    # 识别来源标识
                    source = "unknown"
                    if "url" in item:
                        source = f"URL: {item['url']}"
                    elif "file_id" in item:
                        source = f"FileID: {item['file_id']}"
                    elif "base64" in item:
                        # Base64 太长，只显示前 10 位和长度，以及 MIME 类型
                        mime = item.get("mime_type", "unknown-mime")
                        b64_val = str(item['base64'])
                        source = f"Base64({mime}, len={len(b64_val)}) {b64_val[:10]}..."
                    
                    parts.append(f"[{c_type} | {source}]")
                
                else:
                    parts.append(f"[Unsupported Type: {c_type}]")
            
            res = "\n".join(parts).strip()
        else:
            res = str(content).strip()

        if not res:
            return "[Empty Content]"
        
        # 截断长内容逻辑
        if len(res) > self.limit:
            half = self.limit // 2
            return f"{res[:half]}\n... [已省略 {len(res)-self.limit} 字] ...\n{res[-half:]}"
        return f"{id}-{res}"

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]], 
    ) -> ModelResponse:
        # 构建头部日志
        header = f"\n{self.c1}>>> [{self.namespace}] LLM CALL START <<<{self.reset}"
        lines = [header]

        # 1. 模型与基础信息
        model_id = getattr(request.model, "model_name", "Unknown Model")
        lines.append(f"【Model】: {model_id}")

        if self.show_settings and request.model_settings:
            lines.append(f"【Config】: {request.model_settings}")
        
        if self.show_tools and request.tools:
            t_names = [getattr(t, 'name', str(t)) for t in request.tools]
            lines.append(f"【Tools】: {', '.join(t_names)}")
            if request.tool_choice:
                lines.append(f"【Policy】: {request.tool_choice}")

        if self.show_state and request.state:
            lines.append(f"【State】: {str(request.state)[:200]}...")

        lines.append("-" * 50)

        # 2. 消息流解析
        if self.show_messages:
            if request.system_message:
                msg = request.system_message
                role = "SYSTEM"
                icon = "⚙️"
                # content = self._format_content(msg.content)
                content = msg.content
                lines.append(f"{icon} [{role:^6}] | {content}")
            for msg in list(request.messages):
                role = msg.type.upper()
                icon = {"SYSTEM": "⚙️", "HUMAN": "👤", "AI": "🤖", "TOOL": "🛠️"}.get(role, "📝")
                content = self._format_content(msg.content,msg.id)
                lines.append(f"{icon} [{role:^6}] | {content}")

        lines.append(f"{self.c1}>>> [{self.namespace}] END CALL <<<{self.reset}\n")
        
        # 一次性打印，减少异步干扰
        print("\n".join(lines))

        return await handler(request)
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        
        tool_call = request.tool_call
        t_name = tool_call.get("name", "unknown")
        t_id = tool_call.get("id", "no-id")
        
        print(f"\n{self.c1}🛠️  [{self.namespace}] TOOL EXE: {t_name}{self.reset}")
        print(f"   Args: {json.dumps(tool_call.get('args', {}), ensure_ascii=False)[:200]}...")

        start_t = time.perf_counter()
        try:
            result = await handler(request)
            duration = time.perf_counter() - start_t
            
            # 结果预览处理
            if isinstance(result, ToolMessage):
                preview = self._format_content(result.content,result.id)
                status = "✅"
            elif isinstance(result, Command):
                preview = f"Command(goto={result.goto})"
                status = "🔀"
            else:
                preview = str(result)
                status = "❓"

            print(f"{status} [{self.namespace}] DONE in {duration:.3f}s")
            print(f"   Result: {preview[:150]}...")
            return result

        except Exception as e:
            duration = time.perf_counter() - start_t
            print(f"❌ [{self.namespace}] ERROR in {t_name} ({duration:.3f}s)")
            print(f"   {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise e