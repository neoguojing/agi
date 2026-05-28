import json
import time
import traceback
import unicodedata
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
        max_line_width: int = 150,  # 终端显示最大宽度，您可以根据自己的终端拉伸程度调整（如 120）
        color_header: str = "\033[95m",
        color_reset: str = "\033[0m"
    ):
        self.namespace = name.upper()
        self.show_messages = show_messages
        self.show_tools = show_tools
        self.show_state = show_state
        self.show_settings = show_settings
        self.limit = content_limit
        self.max_line_width = max_line_width
        self.c1 = color_header
        self.reset = color_reset
        
        # 统一左侧宽度 (Icon 1字符 + 空格 + [ 7字符 ] + 空格 = 12字符视觉，为安全取 14 以防 emoji 宽度异常)
        self.gutter_width = 14
        self.empty_gutter = f"{'':{self.gutter_width}}| "

    def _split_and_wrap(self, text: str, width_offset: int = 0) -> List[str]:
        """自定义硬换行，完美支持中英文混排，按终端视觉宽度强行截断"""
        effective_width = max(self.max_line_width - self.gutter_width - width_offset, 40)
        result = []
        
        for line in str(text).splitlines():
            if not line:
                result.append("")
                continue
                
            current_line = ""
            current_width = 0
            
            for char in line:
                # 判断字符的视觉宽度（全角/宽字符为2，其他为1）
                char_w = 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
                
                if current_width + char_w > effective_width:
                    # 达到行宽，截断当前行
                    result.append(current_line)
                    current_line = char
                    current_width = char_w
                else:
                    current_line += char
                    current_width += char_w
                    
            if current_line:
                result.append(current_line)
                
        return result

    def _divider(self, title: str = "", width: int = 80) -> str:
        if not title:
            return "─" * width
        label = f" {title} "
        side = max((width - len(label)) // 2, 1)
        return f"{'─' * side}{label}{'─' * side}"

    def _json_preview(self, data: Any, limit: int = 2000, pretty: bool = True) -> str:
        try:
            if pretty:
                rendered = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            else:
                rendered = json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            rendered = str(data)

        if len(rendered) <= limit:
            return rendered
        omitted = len(rendered) - limit
        return f"{rendered[:limit]}\n... [已省略 {omitted} 字符]"

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
        if content is None:
            return

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
                    source = item.get("url") or item.get("file_id") or "unknown source"
                    if "base64" in item:
                        mime = item.get("mime_type", "unknown-mime")
                        source = f"Base64({mime}, len={len(str(item['base64']))})"
                    res = f"[{c_type} | {source}]"
                else:
                    res = self._json_preview(item, limit=self.limit, pretty=True)
            else:
                res = str(item).strip()

            if not res:
                continue

            if len(res) > self.limit:
                half = self.limit // 2
                omitted = max(len(res) - self.limit, 0)
                res = f"{res[:half]}\n... [已省略 {omitted} 字] ...\n{res[-half:]}"

            yield f"{prefix}{res}"

    def _append_log_line(self, lines: List[str], icon: str, role: str, content: Any, msg_id: Any = None):
        msg_prefix = f"[{msg_id}] " if msg_id else ""
        first_gutter = f"{icon} [{role[:7]:^7}]"
        
        # 使用 ljust 固定英文字符宽度对齐左侧边栏
        first_gutter = f"{first_gutter:<{self.gutter_width}}| "
        current_gutter = first_gutter

        for part in self._yield_formatted_parts(content, ""):
            # 考虑 msg_prefix 的长度偏移，确保消息 ID 出现时也不会溢出
            prefix_w = len(msg_prefix)
            wrapped_lines = self._split_and_wrap(part, width_offset=prefix_w)
            
            for line in wrapped_lines:
                lines.append(f"{current_gutter}{msg_prefix}{line}")
                current_gutter = self.empty_gutter
                msg_prefix = "" 

    def _append_tool_call_details(self, lines: List[str], msg: AIMessage):
        tool_calls = getattr(msg, "tool_calls", []) or []
        if not tool_calls:
            return

        lines.append(f"{self.empty_gutter}├─ 🔧 Tool Calls ({len(tool_calls)})")

        for idx, call in enumerate(tool_calls, start=1):
            call_id = str(call.get("id") or "unknown")
            call_name = str(call.get("name") or "unknown")
            
            lines.append(f"{self.empty_gutter}│  [{idx}] name={call_name}")
            lines.append(f"{self.empty_gutter}│      id={call_id}")
            
            args_str = self._json_preview(call.get("args", {}), limit=3000, pretty=True)
            args_lines = self._split_and_wrap(args_str, width_offset=10) # 10是缩进偏移
            
            if len(args_lines) == 1:
                lines.append(f"{self.empty_gutter}│      args: {args_lines[0]}")
            else:
                lines.append(f"{self.empty_gutter}│      args:")
                for arg_line in args_lines:
                    lines.append(f"{self.empty_gutter}│        {arg_line}")

    def _append_tool_result_details(self, lines: List[str], msg: ToolMessage):
        tool_call_id = str(getattr(msg, "tool_call_id", "") or "unknown")
        tool_name = str(getattr(msg, "name", "") or "unknown")
        artifact = getattr(msg, "artifact", None)

        lines.append(f"{self.empty_gutter}├─ 🧰 Tool Result")
        lines.append(f"{self.empty_gutter}│    name={tool_name}")
        lines.append(f"{self.empty_gutter}│    tool_call_id={tool_call_id}")
        
        if artifact is not None:
            art_str = self._json_preview(artifact, limit=3000, pretty=True)
            art_lines = self._split_and_wrap(art_str, width_offset=8)
            
            if len(art_lines) == 1:
                lines.append(f"{self.empty_gutter}│    artifact={art_lines[0]}")
            else:
                lines.append(f"{self.empty_gutter}│    artifact:")
                for line in art_lines:
                    lines.append(f"{self.empty_gutter}│      {line}")

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        header = f"\n{self.c1}╔═ [{self.namespace}] LLM CALL START{self.reset}"
        lines: List[str] = [header]

        model_id = getattr(request.model, "model_name", "Unknown Model")
        lines.append(f" 🤖 【Model】: {model_id}")

        if self.show_tools and request.tools:
            t_names = [getattr(t, 'name', str(t)) for t in request.tools]
            t_lines = self._split_and_wrap(', '.join(t_names), width_offset=15)
            lines.append(f" 🛠️ 【Tools】: {t_lines[0]}")
            for t_line in t_lines[1:]:
                lines.append(f"               {t_line}")

        lines.append(self._divider("Request Meta"))

        pairing = self._analyze_tool_pairing(request.messages)
        pair_status = "✅ Paired" if pairing["paired"] else "⚠️ Unpaired"
        lines.append(
            f" 🔗 【Tool Pairing】: {pair_status} | "
            f"declared={len(pairing['declared_calls'])}, results={len(pairing['tool_results'])}"
        )

        if pairing["missing_results"]:
            lines.append("   ↳ Missing results for calls: " + ", ".join(
                f"{x['name']}[{x['id']}]" for x in pairing["missing_results"]
            ))
        if pairing["orphan_results"]:
            lines.append("   ↳ Orphan tool results: " + ", ".join(
                f"{x['name']}[{x['id']}]" for x in pairing["orphan_results"]
            ))

        lines.append(self._divider("Message Trace"))
        
        if self.show_messages:
            if request.system_message:
                self._append_log_line(lines, "⚙️", "SYSTEM", request.system_message.content, "SYS")

            for msg in request.messages:
                role_map = {"human": ("👤", "USER"), "ai": ("🤖", "ASSIST"), "tool": ("🛠️", "TOOL")}
                icon, role_name = role_map.get(str(msg.type), ("📝", str(msg.type).upper()))
                
                self._append_log_line(lines, icon, role_name, msg.content, getattr(msg, 'id', None))
                
                if isinstance(msg, AIMessage):
                    self._append_tool_call_details(lines, msg)
                elif isinstance(msg, ToolMessage):
                    self._append_tool_result_details(lines, msg)

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
        
        args_str = self._json_preview(tool_call.get('args', {}), limit=5000, pretty=True)
        args_lines = self._split_and_wrap(args_str, width_offset=8)
        
        if len(args_lines) == 1:
            print(f" 📥 Args: {args_lines[0]}")
        else:
            print(" 📥 Args:")
            for line in args_lines:
                print(f"    {line}")

        start_t = time.perf_counter()
        try:
            result = await handler(request)
            duration = time.perf_counter() - start_t

            content = getattr(result, 'content', str(result))
            parts = list(self._yield_formatted_parts(content, ""))
            preview = parts[0] if parts else "[No Preview]"
            
            if "\n" in preview:
                preview_lines = preview.splitlines()
                preview = f"{preview_lines[0]} ... [Multline Content]"
            else:
                preview = preview[:100] + (" ..." if len(preview) > 100 else "")

            paired_note = ""
            if isinstance(result, ToolMessage):
                res_tool_call_id = getattr(result, "tool_call_id", "")
                paired_note = "✅ matched" if str(res_tool_call_id) == str(t_id) else f"⚠️ mismatch (res={res_tool_call_id}, req={t_id})"
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