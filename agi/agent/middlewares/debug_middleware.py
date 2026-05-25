import json
import time
import traceback
from typing import Callable, Awaitable, List, Any, Optional, Union, Generator
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
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

    def _divider(self, title: str = "", width: int = 80) -> str:
        if not title:
            return "─" * width
        label = f" {title} "
        side = max((width - len(label)) // 2, 1)
        return f"{'─' * side}{label}{'─' * side}"

    def _json_preview(self, data: Any, limit: int = 2000) -> str:
        try:
            rendered = json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            rendered = str(data)

        if len(rendered) <= limit:
            return rendered
        omitted = len(rendered) - limit
        return f"{rendered[:limit]} ... [omitted {omitted} chars]"

    def _analyze_tool_pairing(self, messages: List[BaseMessage]) -> dict[str, Any]:
        declared_calls: List[dict[str, str]] = []
        tool_results: List[dict[str, str]] = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", []) or []:
                    declared_calls.append(
                        {
                            "id": str(call.get("id") or ""),
                            "name": str(call.get("name") or "unknown"),
                        }
                    )
            elif isinstance(msg, ToolMessage):
                tool_results.append(
                    {
                        "id": str(getattr(msg, "tool_call_id", "") or ""),
                        "name": str(getattr(msg, "name", "") or "unknown"),
                    }
                )

        declared_ids = {c["id"] for c in declared_calls if c["id"]}
        result_ids = {r["id"] for r in tool_results if r["id"]}

        missing_results = [c for c in declared_calls if c["id"] and c["id"] not in result_ids]
        orphan_results = [r for r in tool_results if r["id"] and r["id"] not in declared_ids]

        return {
            "declared_calls": declared_calls,
            "tool_results": tool_results,
            "missing_results": missing_results,
            "orphan_results": orphan_results,
            "paired": not missing_results and not orphan_results,
        }

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
        msg_prefix = f"#{msg_id} " if msg_id else ""
        for part in self._yield_formatted_parts(content, ""):
            for i, line in enumerate(part.splitlines()):
                if i == 0:
                    lines.append(f"{icon} [{role:^7}] | {msg_prefix}{line}")
                else:
                    lines.append(f"{'':10} | {line}")  # 统一对齐

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        header = f"\n{self.c1}╔═ [{self.namespace}] LLM CALL START{self.reset}"
        lines: List[str] = [header]

        # 1️⃣ 元信息
        model_id = getattr(request.model, "model_name", "Unknown Model")
        lines.append(f" 🤖 【Model】: {model_id}")

        if self.show_tools and request.tools:
            t_names = [getattr(t, 'name', str(t)) for t in request.tools]
            lines.append(f" 🛠️ 【Tools】: {', '.join(t_names)}")

        lines.append(self._divider("Request Meta"))

        pairing = self._analyze_tool_pairing(request.messages)
        pair_status = "✅ Paired" if pairing["paired"] else "⚠️ Unpaired"
        lines.append(
            f" 🔗 【Tool Pairing】: {pair_status} | "
            f"declared={len(pairing['declared_calls'])}, results={len(pairing['tool_results'])}"
        )

        if pairing["missing_results"]:
            lines.append("   ↳ Missing results for calls: " + ", ".join(
                f"{x['name']}#{x['id']}" for x in pairing["missing_results"]
            ))
        if pairing["orphan_results"]:
            lines.append("   ↳ Orphan tool results: " + ", ".join(
                f"{x['name']}#{x['id']}" for x in pairing["orphan_results"]
            ))

        lines.append(self._divider("Message Trace"))
        # 2️⃣ 消息流解析
        if self.show_messages:
            if request.system_message:
                self._append_log_line(lines, "⚙️", "SYSTEM", request.system_message.content, "SYS")

            for msg in request.messages:
                role_map = {"human": ("👤", "USER"), "ai": ("🤖", "ASSIST"), "tool": ("🛠️", "TOOL")}
                icon, role_name = role_map.get(str(msg.type), ("📝", str(msg.type).upper()))
                self._append_log_line(lines, icon, role_name, msg.content, getattr(msg, 'id', None))

        lines.append(f"{self.c1}╚═ [{self.namespace}] END CALL{self.reset}\n")
        print("\n".join(lines))

        try:
            return await handler(request)
        except Exception as e:
            print(f"❌ [{self.namespace}] MODEL FAILED: {type(e).__name__}")
            traceback.print_exc()
            return ModelResponse(result=[AIMessage(content=(
                f"Model call failed in middleware '{self.namespace}': {type(e).__name__}: {e}"
            ))])

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command[Any]]]],
    ) -> Union[ToolMessage, Command[Any]]:

        tool_call = request.tool_call
        t_name = tool_call.get("name", "unknown")

        t_id = tool_call.get("id", "unknown")
        print(f"\n{self.c1}╔═ [{self.namespace}] TOOL START{self.reset}")
        print(f" 🔧 Name: {t_name}")
        print(f" 🆔 Call ID: {t_id}")
        print(f" 📥 Args: {self._json_preview(tool_call.get('args', {}), limit=3000)}")

        start_t = time.perf_counter()
        try:
            result = await handler(request)
            duration = time.perf_counter() - start_t

            # 提取 result preview
            content = getattr(result, 'content', str(result))
            parts = list(self._yield_formatted_parts(content, ""))
            preview = parts[0] if parts else "[No Preview]"
            preview = preview[:1200] + (" ..." if len(preview) > 1200 else "")

            paired_note = ""
            if isinstance(result, ToolMessage):
                res_tool_call_id = getattr(result, "tool_call_id", "")
                paired_note = "✅ matched" if str(res_tool_call_id) == str(t_id) else f"⚠️ mismatch (result={res_tool_call_id}, request={t_id})"
            else:
                paired_note = "ℹ️ non-ToolMessage result"

            status = "✅" if not isinstance(result, Exception) else "❌"
            print(f"{status} [{self.namespace}] COMPLETED ({duration:.3f}s)")
            print(f" 🔗 Pairing: {paired_note}")
            print(f" 📤 Result: {preview}")
            print(f"{self.c1}╚═ [{self.namespace}] TOOL END{self.reset}")

            return result

        except Exception as e:
            print(f"❌ [{self.namespace}] FAILED: {type(e).__name__}")
            traceback.print_exc()
            tool_call_id = tool_call.get("id", "unknown")
            print(f"{self.c1}╚═ [{self.namespace}] TOOL END{self.reset}")
            return ToolMessage(
                content=(
                    f"Tool call failed in middleware '{self.namespace}' for '{t_name}': "
                    f"{type(e).__name__}: {e}"
                ),
                tool_call_id=tool_call_id,
                name=t_name,
            )
