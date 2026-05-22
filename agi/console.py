import asyncio
import io
import json
import mimetypes
import os
import shlex
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from langgraph.graph.message import add_messages
from langgraph.types import Overwrite
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from agi.agent.agent import stream_agent_async
from agi.agent.context import Context
from agi.api.media import process_multimodal_content
from agi.apps.common import FileObject, ImageURL, MessageContent

STATE_CACHE = ".cli_session.json"
HISTORY_CACHE = ".cli_prompt_history"
MAX_INLINE_DOC_CHARS = 20000
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".xml", ".html", ".rst"}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
console = Console()


@dataclass
class CLICommand:
    name: str
    handler: Callable[[str], bool]
    help_text: str


class HybridCompleter(Completer):
    def __init__(self, command_words: List[str]):
        self.command_words = set(command_words)
        self.command_completer = WordCompleter(command_words, ignore_case=True)
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        first = stripped.split(maxsplit=1)[0] if stripped else ""
        is_command_context = first.startswith("/") and first in self.command_words
        if is_command_context:
            yield from self.command_completer.get_completions(document, complete_event)
        yield from self.path_completer.get_completions(document, complete_event)


class DeepAgentCLI:
    def __init__(self):
        self.cwd = Path.cwd()
        self.load_session()
        self.command_map: Dict[str, CLICommand] = {}
        self._register_commands()

        completer = HybridCompleter(list(self.command_map.keys()))
        self.session = PromptSession(
            completer=completer,
            history=FileHistory(HISTORY_CACHE),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
        )

    def load_session(self):
        if os.path.exists(STATE_CACHE):
            try:
                with open(STATE_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.user_id = data.get("user_id", "admin")
                    self.conversation_id = data.get("conversation_id", str(uuid.uuid4()))
                    self.thread_id = self.conversation_id
                    self.state = {"messages": [], "user_id": self.user_id}
                    return
            except Exception:
                pass
        self.user_id = "admin"
        self.conversation_id = str(uuid.uuid4())
        self.thread_id = self.conversation_id
        self.state = {"messages": []}

    def _save_session(self):
        tmp_path = STATE_CACHE + ".tmp"
        try:
            data = {
                "user_id": self.user_id,
                "thread_id": self.thread_id,
                "conversation_id": self.conversation_id,
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, STATE_CACHE)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _register_commands(self):
        self.command_map = {
            "/help": CLICommand("/help", self._cmd_help, "显示帮助"),
            "/quit": CLICommand("/quit", self._cmd_quit, "退出"),
            "/reset": CLICommand("/reset", self._cmd_reset, "清空上下文"),
            "/pwd": CLICommand("/pwd", self._cmd_pwd, "显示当前目录"),
            "/cd": CLICommand("/cd", self._cmd_cd, "切换目录: /cd <path>"),
            "/ls": CLICommand("/ls", self._cmd_ls, "列目录: /ls [path]"),
            "/cat": CLICommand("/cat", self._cmd_cat, "显示文件: /cat <file>"),
            "/history": CLICommand("/history", self._cmd_history, "查看历史消息(支持翻页)"),
        }

    def _cmd_help(self, _: str) -> bool:
        lines = [f"{k:<10} {v.help_text}" for k, v in self.command_map.items()]
        lines.append("\n提示: ↑/↓ 浏览输入历史, Tab 自动补全路径与命令。")
        lines.append("多模态输入: img:<path|url> file:<path> doc:<path> audio:<path> video:<path>")
        lines.append("也支持直接输入存在的文件路径（如 /aaa/bbb/a.txt）自动作为输入传给 Agent。")
        console.print(Panel("\n".join(lines), title="Commands", border_style="cyan"))
        return True

    def _cmd_quit(self, _: str) -> bool:
        return False

    def _cmd_reset(self, _: str) -> bool:
        self.state["messages"] = []
        console.print("[dim]上下文已清空[/dim]")
        return True

    def _cmd_pwd(self, _: str) -> bool:
        console.print(str(self.cwd))
        return True

    def _cmd_cd(self, arg: str) -> bool:
        target = (self.cwd / arg).expanduser().resolve() if arg else Path.home()
        if not target.exists() or not target.is_dir():
            console.print(f"[red]目录不存在: {target}[/red]")
            return True
        self.cwd = target
        os.chdir(target)
        console.print(f"[green]已切换目录: {target}[/green]")
        return True

    def _cmd_ls(self, arg: str) -> bool:
        target = (self.cwd / arg).expanduser().resolve() if arg else self.cwd
        if not target.exists() or not target.is_dir():
            console.print(f"[red]目录不存在: {target}[/red]")
            return True
        rows = []
        for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            mark = "📁" if p.is_dir() else "📄"
            rows.append(f"{mark} {p.name}")
        console.print(Panel("\n".join(rows) if rows else "(空目录)", title=f"ls {target}"))
        return True

    def _cmd_cat(self, arg: str) -> bool:
        if not arg:
            console.print("[yellow]用法: /cat <file>[/yellow]")
            return True
        target = (self.cwd / arg).expanduser().resolve()
        if not target.exists() or not target.is_file():
            console.print(f"[red]文件不存在: {target}[/red]")
            return True
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            console.print(f"[red]不是 UTF-8 文本文件: {target}[/red]")
            return True
        with console.pager(styles=True):
            console.print(Panel(text, title=str(target)))
        return True

    def _cmd_history(self, _: str) -> bool:
        msgs = self.state.get("messages", [])
        if not msgs:
            console.print("[dim]暂无会话消息[/dim]")
            return True
        rendered = []
        for i, msg in enumerate(msgs, 1):
            content = str(getattr(msg, "content", "")).strip()
            if len(content) > 400:
                content = content[:400] + "..."
            rendered.append(f"[{i}] {type(msg).__name__}:\n{content}\n")
        with console.pager(styles=True):
            console.print(Markdown("\n---\n".join(rendered)))
        return True

    def _parse_command(self, user_input: str) -> Optional[bool]:
        stripped = user_input.lstrip()
        if not stripped.startswith("/"):
            return None
        parts = shlex.split(stripped)
        if not parts:
            return None
        cmd = parts[0].lower()
        handler = self.command_map.get(cmd)
        if not handler:
            return None
        arg = " ".join(parts[1:]) if len(parts) > 1 else ""
        return handler.handler(arg)

    def _resolve_path(self, raw_path: str) -> Path:
        return (self.cwd / raw_path).expanduser().resolve()

    def _read_document_for_prompt(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return f"[文档不存在: {path}]"
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            return f"[暂不支持直接读取该文档类型，请使用 file: 附件方式传入: {path}]"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[文档非 UTF-8 编码，无法直接读取: {path}]"
        if len(text) > MAX_INLINE_DOC_CHARS:
            text = text[:MAX_INLINE_DOC_CHARS] + "\n...\n[文档过长，已截断]"
        return f"\n[DOC_BEGIN: {path}]\n{text}\n[DOC_END]\n"

    def _looks_like_existing_path(self, token: str) -> Optional[Path]:
        if not token or token.startswith(("http://", "https://", "data:")):
            return None
        candidate = self._resolve_path(token)
        if candidate.exists():
            return candidate
        return None

    def _directory_snapshot_for_prompt(self, path: Path) -> str:
        try:
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except Exception as exc:
            return f"[目录读取失败: {path}, error={exc}]"

        preview = []
        for item in entries[:200]:
            prefix = "DIR" if item.is_dir() else "FILE"
            preview.append(f"- {prefix}: {item.name}")
        if len(entries) > 200:
            preview.append("- ... [目录内容过多，已截断]")
        return f"\n[DIR_BEGIN: {path}]\n" + "\n".join(preview) + "\n[DIR_END]\n"

    def _append_file_content(self, path: Path, contents: List[MessageContent]):
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        if path.suffix.lower() in TEXT_EXTENSIONS:
            doc_content = self._read_document_for_prompt(path)
            contents.append(MessageContent(type="text", text=doc_content))
            return
        contents.append(
            MessageContent(
                type="file",
                file=FileObject(file_id=str(path), mime_type=mime),
            )
        )

    def _smart_parse(self, text: str):
        tokens = text.split()
        contents = []
        text_buffer = []

        def flush_text():
            if text_buffer:
                contents.append(MessageContent(type="text", text=" ".join(text_buffer)))
                text_buffer.clear()

        for t in tokens:
            if t.startswith("img:"):
                flush_text()
                source = t[4:]
                if source and not source.startswith(("http://", "https://", "data:")):
                    source = str(self._resolve_path(source))
                contents.append(MessageContent(type="image_url", image_url=ImageURL(url=source)))
            elif t.startswith(("file:", "audio:", "video:")):
                flush_text()
                _, raw = t.split(":", 1)
                self._append_file_content(self._resolve_path(raw), contents)
            elif t.startswith("doc:"):
                flush_text()
                doc_path = self._resolve_path(t[4:])
                doc_content = self._read_document_for_prompt(doc_path)
                contents.append(MessageContent(type="text", text=doc_content))
            else:
                maybe_path = self._looks_like_existing_path(t)
                if maybe_path is not None:
                    flush_text()
                    if maybe_path.is_dir():
                        contents.append(
                            MessageContent(
                                type="text",
                                text=self._directory_snapshot_for_prompt(maybe_path),
                            )
                        )
                    else:
                        self._append_file_content(maybe_path, contents)
                else:
                    text_buffer.append(t)
        flush_text()
        return contents



    def _extract_text_from_content_blocks(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        txt = item.get("text", "")
                        if txt:
                            parts.append(str(txt))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(x for x in parts if x).strip()
        if isinstance(content, dict):
            if content.get("type") == "text":
                return str(content.get("text", "")).strip()
            return str(content)
        return str(content or "").strip()

    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        result: Dict[str, Any] = {
            "type": type(msg).__name__,
            "content": getattr(msg, "content", ""),
            "response_metadata": getattr(msg, "response_metadata", {}) or {},
            "usage_metadata": getattr(msg, "usage_metadata", {}) or {},
            "additional_kwargs": getattr(msg, "additional_kwargs", {}) or {},
            "name": getattr(msg, "name", None),
            "id": getattr(msg, "id", None),
            "tool_calls": getattr(msg, "tool_calls", None),
            "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
        }
        return result

    def _message_type_name(self, msg_data: Dict[str, Any]) -> str:
        raw_type = str(msg_data.get("type") or "Message")
        if "|" in raw_type:
            raw_type = raw_type.split("|")[0].strip()
        return raw_type

    def _parse_messages_event(self, part: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = part.get("data")
        if isinstance(data, tuple) and data:
            msg = data[0]
            meta = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
            return self._message_to_dict(msg), meta
        if isinstance(data, list) and data:
            msg = data[0]
            meta = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
            return self._message_to_dict(msg), meta
        if isinstance(data, dict):
            msg = data.get("message") or data.get("chunk") or data.get("data")
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            if msg is not None:
                return self._message_to_dict(msg), meta
        return {}, {}

    def _update_stats_from_message(self, msg_data: Dict[str, Any], stats_info: Dict[str, Any], start_time: float):
        metadata = msg_data.get("response_metadata") or {}
        usage = msg_data.get("usage_metadata") or {}
        additional = msg_data.get("additional_kwargs") or {}

        model_name = metadata.get("model_name") or metadata.get("model") or additional.get("model_name")
        if model_name:
            stats_info["model"] = str(model_name)

        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        total_t = usage.get("total_tokens")

        if isinstance(in_t, int):
            stats_info["input_tokens"] = max(stats_info.get("input_tokens", 0), in_t)
        if isinstance(out_t, int):
            stats_info["output_tokens"] = stats_info.get("output_tokens", 0) + max(out_t, 0)
        if isinstance(total_t, int):
            stats_info["total_tokens"] = max(
                stats_info.get("total_tokens", 0),
                total_t,
                stats_info.get("input_tokens", 0) + stats_info.get("output_tokens", 0),
            )
        else:
            stats_info["total_tokens"] = max(
                stats_info.get("total_tokens", 0),
                stats_info.get("input_tokens", 0) + stats_info.get("output_tokens", 0),
            )

        stats_info["tokens"] = (
            f"In: {stats_info.get('input_tokens', 0)} | "
            f"Out: {stats_info.get('output_tokens', 0)} | "
            f"Total: {stats_info.get('total_tokens', 0)}"
        )

        elapsed = max(time.time() - start_time, 1e-6)
        out_total = stats_info.get("output_tokens", 0)
        stats_info["tps"] = out_total / elapsed if out_total else 0.0

    def _format_message_preview(self, msg_data: Dict[str, Any], max_len: int = 1000) -> str:
        msg_type = self._message_type_name(msg_data)
        content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
        preview = content if content else "(empty)"
        if len(preview) > max_len:
            preview = preview[:max_len] + "..."
        line = f"- **{msg_type}**: {preview}"

        usage = msg_data.get("usage_metadata") or {}
        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        total_t = usage.get("total_tokens")
        if any(isinstance(x, int) for x in (in_t, out_t, total_t)):
            line += f"\n  - tokens: in={in_t or 0}, out={out_t or 0}, total={total_t or 0}"

        meta = msg_data.get("response_metadata") or {}
        model = meta.get("model_name")
        finish_reason = meta.get("finish_reason")
        provider = meta.get("model_provider")
        if model or finish_reason or provider:
            line += (
                f"\n  - meta: model={model or 'N/A'}, provider={provider or 'N/A'}, "
                f"finish_reason={finish_reason or 'N/A'}"
            )

        return line

    def _normalize_message_content(self, msg_content: Any) -> str:
        if isinstance(msg_content, list):
            normalized = []
            for item in msg_content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        normalized.append(str(item.get("text", "")))
                    else:
                        normalized.append(str(item))
                else:
                    normalized.append(str(item))
            return "\n".join(x for x in normalized if x).strip()

        return str(msg_content or "").strip()

    async def handle_stream(self, live):
        full_response = ""
        seen_messages: Set[Tuple[str, str]] = set()
        latest_ai_message = ""
        updates_trace: List[str] = []
        max_updates_trace = 30

        config = {"configurable": {"thread_id": self.thread_id}}
        context = Context(user_id=self.user_id, conversation_id=self.conversation_id)
        start_time = time.time()
        last_update_time = 0.0
        stats_info: Dict[str, Any] = {
            "model": "N/A",
            "tokens": "In: 0 | Out: 0 | Total: 0",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "node": "N/A",
            "tps": 0.0,
        }

        async for part in stream_agent_async(self.state, config=config, context=context, stream_mode=["messages"]):
            if isinstance(part, dict) and part.get("type") == "messages":
                msg_data, event_meta = self._parse_messages_event(part)
                if msg_data:
                    msg_type = self._message_type_name(msg_data)
                    content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
                    stream_node = event_meta.get("langgraph_node") or event_meta.get("lc_agent_name")
                    if stream_node:
                        stats_info["node"] = str(stream_node)
                    if content:
                        full_response += str(content).replace("→", "->")
                    if "AIMessage" in msg_type and content:
                        latest_ai_message = (latest_ai_message + content) if "Chunk" in msg_type else content
                    self._update_stats_from_message(msg_data, stats_info, start_time)

            elif isinstance(part, dict) and part.get("type") == "updates":
                updates = part.get("data", {})
                if isinstance(updates, dict):
                    for node_name, node_payload in updates.items():
                        if not isinstance(node_payload, dict) or "messages" not in node_payload:
                            continue
                        stats_info["node"] = str(node_name)
                        msgs = node_payload.get("messages", [])
                        if isinstance(msgs, Overwrite):
                            continue
                        if not isinstance(msgs, list):
                            continue

                        for raw_msg in msgs:
                            msg_data = self._message_to_dict(raw_msg)
                            msg_type = self._message_type_name(msg_data)
                            content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
                            msg_key = (str(node_name), f"{msg_type}|{content}|{msg_data.get('id')}")
                            if msg_key in seen_messages:
                                continue
                            seen_messages.add(msg_key)

                            self._update_stats_from_message(msg_data, stats_info, start_time)
                            if msg_type == "AIMessage" and content:
                                latest_ai_message = content
                            preview = self._format_message_preview(msg_data, max_len=400)
                            updates_trace.append(f"### Node `{node_name}`\n{preview}")
                            if len(updates_trace) > max_updates_trace:
                                updates_trace = updates_trace[-max_updates_trace:]

            now = time.time()
            if now - last_update_time > 0.05:
                elapsed = now - start_time
                sections: List[str] = []
                if updates_trace:
                    sections.append("## Updates Trace")
                    sections.extend(updates_trace[-10:])
                if latest_ai_message:
                    sections.append("\n---\n\n## Latest AIMessage\n" + latest_ai_message)
                elif full_response:
                    sections.append("\n---\n\n## Streamed Response\n" + full_response)
                body = "\n\n".join(sections).strip()
                if not body:
                    body = "(waiting for updates...)"

                subtitle = (
                    f"[bold cyan]{elapsed:.1f}s[/bold cyan] | {stats_info['model']} | "
                    f"Node: [yellow]{stats_info['node']}[/yellow] | {stats_info['tokens']} | "
                    f"[magenta]{stats_info['tps']:.1f} t/s[/magenta]"
                )
                live.update(
                    Panel(
                        Markdown(body),
                        title="[bold blue]Agent Response[/bold blue]",
                        subtitle=subtitle,
                        subtitle_align="right",
                        border_style="blue",
                    )
                )
                last_update_time = now

        return latest_ai_message or full_response

    async def run(self):
        self._save_session()
        console.print(Panel(f"🔥 [bold green]Agent 已就绪[/bold green]\nThread: {self.thread_id[:8]}...\nCWD: {self.cwd}", border_style="green"))

        while True:
            try:
                user_input = await self.session.prompt_async(HTML("\n👤 <b><ansiyellow>You > </ansiyellow></b>"))
                if not user_input:
                    continue
                command_result = self._parse_command(user_input)
                if command_result is not None:
                    if command_result is False:
                        break
                    continue

                human_message = process_multimodal_content(self._smart_parse(user_input))
                self.state["messages"] = add_messages(self.state["messages"], [human_message])

                with Live(
                    Panel(Spinner("dots", text="思考中..."), title="Agent Response", border_style="blue"),
                    console=console,
                    refresh_per_second=10,
                    transient=False,
                ) as live:
                    await self.handle_stream(live)
                self._save_session()
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                traceback.print_exc()
                console.print(f"[red]发生错误: {e}[/red]")


if __name__ == "__main__":
    try:
        cli = DeepAgentCLI()
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Agent 崩溃: {e}")
        traceback.print_exc()
        sys.exit(1)
