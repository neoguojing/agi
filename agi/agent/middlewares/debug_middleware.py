import json
import time
import traceback
from typing import Callable, Awaitable, List, Any, Optional, Union, Generator
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.types import Command
from langchain.tools.tool_node import ToolCallRequest


class DebugLLMContextMiddleware(AgentMiddleware):
    def __init__(
        self,
        name: str = "DEFAULT",
        show_messages: bool = True,
        show_tools: bool = True,
        show_state: bool = False,
        show_settings: bool = False,
        content_limit: int = 10000,
        color_header: str = "\033[95m",
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

    def _yield_formatted_parts(self, content: Any, msg_id: Any = None) -> Generator[str, None, None]:
        """
        将复杂内容拆解为独立可打印字符串，支持列表、生成器、字典等。
        """
        if content is None:
            return

        # 支持生成器、列表、单条内容
        if isinstance(content, Generator):
            items = list(content)
        elif isinstance(content, list):
            items = content
        else:
            items = [content]

        for item in items:
            prefix = f"[{msg_id}] " if msg_id else ""
            res = ""

            if isinstance(item, str):
                res = item.strip()
            elif isinstance(item, dict):
                c_type = str(item.get("type", "unknown")).upper()
                if c_type == "TEXT":
                    res = str(item.get("text", "")).strip()
                elif c_type in ["IMAGE", "FILE", "AUDIO", "VIDEO"]:
                    source = "unknown"
                    if "url" in item:
                        source = f"URL: {item['url']}"
                    elif "file_id" in item:
                        source = f"FileID: {item['file_id']}"
                    elif "base64" in item:
                        mime = item.get("mime_type", "unknown-mime")
                        source = f"Base64({mime}, len={len(str(item['base64']))})"
                    res = f"[{c_type} | {source}]"
                else:
                    res = f"[Unsupported Type: {c_type}]"
            else:
                res = str(item).strip()

            if not res:
                continue

            # 截断处理，保证省略字数非负
            if len(res) > self.limit:
                half = self.limit // 2
                omitted = max(len(res) - self.limit, 0)
                res = f"{res[:half]}\n    ... [已省略 {omitted} 字] ...\n    {res[-half:]}"

            yield f"{prefix}{res}"

    def _append_log_line(self, lines: List[str], icon: str, role: str, content: Any, msg_id: Any = None):
        """将消息内容拆分为多行并添加到 lines 列表"""
        for part in self._yield_formatted_parts(content, ""):
            for i, line in enumerate(part.splitlines()):
                if i == 0:
                    lines.append(f"{icon} [{role:^7}] | {line}")
                else:
                    lines.append(f"{'':10} | {line}")  # 统一对齐

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        header = f"\n{self.c1}>>> [{self.namespace}] LLM CALL START <<<{self.reset}"
        lines: List[str] = [header]

        # 1️⃣ 元信息
        model_id = getattr(request.model, "model_name", "Unknown Model")
        lines.append(f" 🤖 【Model】: {model_id}")

        if self.show_tools and request.tools:
            t_names = [getattr(t, 'name', str(t)) for t in request.tools]
            lines.append(f" 🛠️ 【Tools】: {', '.join(t_names)}")

        lines.append("=" * 60)

        # 2️⃣ 消息流解析
        if self.show_messages:
            if request.system_message:
                self._append_log_line(lines, "⚙️", "SYSTEM", request.system_message.content, "SYS")

            for msg in request.messages:
                role_map = {"human": ("👤", "USER"), "ai": ("🤖", "ASSIST"), "tool": ("🛠️", "TOOL")}
                icon, role_name = role_map.get(str(msg.type), ("📝", str(msg.type).upper()))
                self._append_log_line(lines, icon, role_name, msg.content, getattr(msg, 'id', None))

        lines.append(f"{self.c1}>>> [{self.namespace}] END CALL <<<{self.reset}\n")
        print("\n".join(lines))

        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command[Any]]]],
    ) -> Union[ToolMessage, Command[Any]]:

        tool_call = request.tool_call
        t_name = tool_call.get("name", "unknown")

        print(f"\n{self.c1}🚀 [{self.namespace}] TOOL START: {t_name}{self.reset}")
        print(f"   📥 Args: {json.dumps(tool_call.get('args', {}), ensure_ascii=False)}")

        start_t = time.perf_counter()
        try:
            result = await handler(request)
            duration = time.perf_counter() - start_t

            # 提取 result preview
            res_id = getattr(result, 'id', 'res')
            content = getattr(result, 'content', str(result))
            parts = list(self._yield_formatted_parts(content, ""))
            preview = parts[0] if parts else "[No Preview]"

            status = "✅" if not isinstance(result, Exception) else "❌"
            print(f"{status} [{self.namespace}] COMPLETED ({duration:.3f}s)")
            print(f"   📤 Result: {preview}")

            return result

        except Exception as e:
            print(f"❌ [{self.namespace}] FAILED: {type(e).__name__}")
            traceback.print_exc()
            raise e