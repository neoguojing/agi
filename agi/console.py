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
from typing import Any, Callable, Dict, List, Optional

# ---- Textual 核心组件 ----
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Markdown, Static
from textual.containers import VerticalScroll
from textual import work, on
from textual.binding import Binding

# ---- Rich 美化组件 ----
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---- 外部依赖接口 (保持不变) ----
from langgraph_sdk import get_client
from langgraph.graph.message import add_messages

from agi.agent.agent import stream_agent_async
from agi.agent.context import Context
from agi.agent.stream_processor import StreamProcessor
from agi.api.media import process_multimodal_content
from agi.apps.common import FileObject, ImageURL, MessageContent
from agi.config import LANGGRAPH_MAIN_URL

STATE_CACHE = ".cli_session.json"
HISTORY_CACHE = ".cli_prompt_history"
MAX_INLINE_DOC_CHARS = 20000
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".xml", ".html", ".rst"}
STREAM_DONE = object()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


@dataclass
class CLICommand:
    name: str
    handler: Callable[[str], bool]
    help_text: str


# =====================================================================
# 1. 核心网络/事件流处理 (保持完全不变)
# =====================================================================
class ThreadEventProducer:
    def __init__(self, client, thread_id: str, assistant_id: str, event_queue: asyncio.Queue):
        self.client = client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.event_queue = event_queue
        self.thread_stream = None
        self.ready = asyncio.Event()

    async def start(self):
        try:
            async with self.client.threads.stream(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
            ) as thread:
                self.thread_stream = thread
                self.ready.set()
                async for event in thread.events:
                    await self.event_queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()
            raise

    async def wait_ready(self):
        await self.ready.wait()

    async def submit_and_wait(self, input_data: dict[str, Any]):
        await self.wait_ready()
        await self.thread_stream.run.start(input=input_data)
        try:
            return await self.thread_stream.output
        finally:
            await self.event_queue.put(STREAM_DONE)


class EventConsumer:
    def __init__(self, event_queue: asyncio.Queue):
        self.event_queue = event_queue

    async def run(self) -> StreamProcessor:
        processor = StreamProcessor()
        while True:
            event = await self.event_queue.get()
            if event is STREAM_DONE:
                break
            try:
                processor.process_part(event)
            except Exception:
                traceback.print_exc()
        return processor


# =====================================================================
# 2. Textual 现代 TUI 展现层
# =====================================================================
class DeepAgentTUI(App):
    TITLE = "DeepAgent Workspace"
    
    # 用 CSS 优雅定义全屏布局与组件样式
    CSS = """
    #chat-container {
        height: 1fr;
        border: solid cyan;  /* 👈 删掉 cubic，改为 solid 或者 round */
        padding: 0 1;
        overflow-y: scroll;
        background: $surface;
    }
    .msg-user {
        margin: 1 0;
        background: $boost;
        padding: 0 1;
    }
    .msg-agent {
        margin: 1 0;
        padding: 0 1;
    }
    .tool-badge {
        color: $accent;
        text-style: italic;
        margin-left: 2;
    }
    Input {
        dock: bottom;
        margin: 1 0 0 0;
        border: tall double gray;
    }
    Input:focus {
        border: tall double cyan;
    }
    """

    # 快捷键绑定
    BINDINGS = [
        Binding("ctrl+q", "quit", "退出系统", show=True),
        Binding("ctrl+l", "clear_screen", "清屏", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.cwd = Path.cwd()
        self.load_session()
        self.client = get_client(url=LANGGRAPH_MAIN_URL)
        self.assistant_id = None
        self.command_map: Dict[str, CLICommand] = {}
        self._register_commands()

    def compose(self) -> ComposeResult:
        """组装静态 UI 架构"""
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield Input(placeholder="输入提示词或命令 (如 /help, /cd, img:路径)...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        """当 UI 加载完成后的初始化行为"""
        self.container = self.query_one("#chat-container", VerticalScroll)
        self.input_box = self.query_one("#chat-input", Input)
        self._update_status_bar()
        
        # 打印欢迎面板
        welcome_panel = Panel(
            f"🔥 [bold green]Agent 工作台已就绪[/bold green]\n"
            f"会话单号: [yellow]{self.thread_id[:12]}[/yellow]\n"
            f"当前工作目录: [cyan]{self.cwd}[/cyan]", 
            border_style="green"
        )
        self.container.mount(Static(welcome_panel))
        self.input_box.focus()

    def _update_status_bar(self):
        """动态更新顶栏副标题"""
        self.sub_title = f"📁 目录: {self.cwd.name} | 🧵 线程: {self.thread_id[:8]}"

    # =====================================================================
    # 3. 核心异步流式渲染监听 (The Magic Box)
    # =====================================================================
    @on(Input.Submitted, "#chat-input")
    async def handle_input_event(self, event: Input.Submitted):
        user_input = event.value.strip()
        if not user_input:
            return
        
        # 清空输入区，抢先一步上屏用户消息
        event.input.value = ""
        user_md = Markdown(f"👤 **You >** {user_input}", classes="msg-user")
        await self.container.mount(user_md)
        self.container.scroll_end(animate=False)

        # 检查并拦截斜杠命令
        command_result = self._parse_command(user_input)
        if command_result is not None:
            if command_result is False:
                self.exit()
            return

        # 智能多模态解析并压入状态栈
        try:
            human_message = process_multimodal_content(self._smart_parse(user_input))
            self.state["messages"] = add_messages(self.state["messages"], [human_message])
            
            # 唤醒后台异步流渲染线程
            self.stream_agent_response_task()
        except Exception as e:
            await self.container.mount(Static(f"[red]输入解析失败: {e}[/red]"))

    @work(exclusive=True)
    async def stream_agent_response_task(self) -> None:
        """后台专职工作协程：负责抽干队列并实时刷新 UI，绝不卡死主界面"""
        # 1. 预先挂载一个专门接收 Agent 响应的 Markdown 组件和状态徽章
        status_badge = Static("⚙️ [dim]Agent 正在整理思绪...[/dim]", classes="tool-badge")
        agent_markdown = Markdown("🤖 **Agent >** \n", classes="msg-agent")
        
        await self.container.mount(status_badge)
        await self.container.mount(agent_markdown)
        self.container.scroll_end(animate=False)

        event_queue = asyncio.Queue()
        producer = ThreadEventProducer(
            client=self.client,
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            event_queue=event_queue,
        )

        producer_task = asyncio.create_task(producer.start())
        processor = StreamProcessor()

        try:
            if self.assistant_id and self.client:
                input_data = {"messages": self._prepare_input_data()}
                asyncio.create_task(producer.submit_and_wait(input_data))
            else:
                # 本地调试流回退通道
                config = {"configurable": {"thread_id": self.thread_id}}
                context = Context(user_id=self.user_id, conversation_id=self.conversation_id)
                async def _local_pusher():
                    async for part in stream_agent_async(self.state, config=config, context=context, stream_mode=["messages", "updates"]):
                        await event_queue.put(part)
                    await event_queue.put(STREAM_DONE)
                asyncio.create_task(_local_pusher())

            # 2. 进入高频消费循环，只要队列有东西，立刻重绘对应的 Markdown
            while True:
                event = await event_queue.get()
                if event is STREAM_DONE:
                    break

                processor.process_part(event)
                
                # 动态捕捉当前的节点状态（如正在调用某个特定 Tool）
                current_node = processor.stats.node or "执行中"
                status_badge.update(f"⚙️ [bold yellow]当前节点: {current_node}[/bold yellow] ...")

                # 提取截止当前时间节点拼接完毕的 Markdown 文本
                content_text = processor.get_presentation_body()
                if content_text.strip():
                    # 极其丝滑的全局打字机效果刷新
                    agent_markdown.update(f"🤖 **Agent >**\n{content_text}")
                    self.container.scroll_end(animate=False)

            # 3. 完结撒花：定格最终状态，并追加轻量化的统计尾巴
            status_badge.update("✅ [dim]响应完成[/dim]")
            elapsed = time.time() - processor.start_time
            stats = processor.stats
            tps = stats.output_tokens / max(elapsed, 1e-6) if stats.output_tokens else 0.0
            
            stats_footer = Static(
                f"[dim]⏱️ 耗时: {elapsed:.1f}s  |  🚀 速度: {tps:.1f} t/s  "
                f"|  ⬇️ Input: {stats.input_tokens}  |  ⬆️ Output: {stats.output_tokens}[/dim]"
            )
            await self.container.mount(stats_footer)
            self.container.scroll_end(animate=True)
            self._save_session()

        except Exception as e:
            status_badge.update("❌ [bold red]流式连接崩溃[/bold red]")
            await self.container.mount(Static(f"[red]{traceback.format_exc()}[/red]"))
        finally:
            if not producer_task.done():
                producer_task.cancel()

    # =====================================================================
    # 4. 内置快捷键动作与斜杠命令处理 (无缝重定向至 TUI 挂载)
    # =====================================================================
    def action_clear_screen(self) -> None:
        """绑定的 Ctrl+L 清屏动作"""
        self.container.clear()
        self.container.mount(Static("[dim]会话大底已重置清空[/dim]"))

    def _cmd_help(self, _: str) -> bool:
        lines = [f"{k:<10} {v.help_text}" for k, v in self.command_map.items()]
        lines.append("\n💡 提示: 支持 Tab 键在输入框中触发框架原生高亮。")
        lines.append("📂 多模态快捷键: img:图片路径 file:附件路径 doc:文本文档")
        self.container.mount(Static(Panel("\n".join(lines), title="帮助菜单", border_style="cyan")))
        self.container.scroll_end()
        return True

    def _cmd_quit(self, _: str) -> bool:
        return False

    def _cmd_reset(self, _: str) -> bool:
        self.state["messages"] = []
        self.container.mount(Static("[dim]上下文记忆已被抹除[/dim]"))
        self.container.scroll_end()
        return True

    def _cmd_pwd(self, _: str) -> bool:
        self.container.mount(Static(f"[cyan]{self.cwd}[/cyan]"))
        self.container.scroll_end()
        return True

    def _cmd_cd(self, arg: str) -> bool:
        target = (self.cwd / arg).expanduser().resolve() if arg else Path.home()
        if not target.exists() or not target.is_dir():
            self.container.mount(Static(f"[red]路径不存在: {target}[/red]"))
            return True
        self.cwd = target
        os.chdir(target)
        self._update_status_bar()
        self.container.mount(Static(f"[green]已下潜至新目录: {target}[/green]"))
        self.container.scroll_end()
        return True

    def _cmd_ls(self, arg: str) -> bool:
        target = (self.cwd / arg).expanduser().resolve() if arg else self.cwd
        if not target.exists() or not target.is_dir():
            self.container.mount(Static(f"[red]路径异常: {target}[/red]"))
            return True
        rows = []
        for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            mark = "📁" if p.is_dir() else "📄"
            rows.append(f"{mark} {p.name}")
        self.container.mount(Static(Panel("\n".join(rows) if rows else "(空目录)", title=f"ls {target.name}")))
        self.container.scroll_end()
        return True

    def _cmd_cat(self, arg: str) -> bool:
        if not arg:
            self.container.mount(Static("[yellow]用法提示: /cat <文件名称>[/yellow]"))
            return True
        target = (self.cwd / arg).expanduser().resolve()
        if not target.exists() or not target.is_file():
            self.container.mount(Static(f"[red]目标文件不存在: {target}[/red]"))
            return True
        try:
            text = target.read_text(encoding="utf-8")
            # TUI 优化：直接作为一个代码块 Markdown 挂载进滚动区域，免去传统 pager 阻塞
            self.container.mount(Markdown(f"```text\n{text}\n```"))
        except UnicodeDecodeError:
            self.container.mount(Static(f"[red]非标准 UTF-8 文本无法预览: {target}[/red]"))
        self.container.scroll_end()
        return True

    def _cmd_history(self, _: str) -> bool:
        msgs = self.state.get("messages", [])
        if not msgs:
            self.container.mount(Static("[dim]当前历史消息栈为空[/dim]"))
            return True
        rendered = []
        for i, msg in enumerate(msgs, 1):
            content = str(getattr(msg, "content", "")).strip()
            if len(content) > 300:
                content = content[:300] + "\n...[内容过长已折叠]"
            rendered.append(f"### [{i}] {type(msg).__name__}\n{content}\n")
        self.container.mount(Markdown("\n---\n".join(rendered)))
        self.container.scroll_end()
        return True

    # =====================================================================
    # 5. 辅助工具与上下文管理方法 (保持完全不变)
    # =====================================================================
    def load_session(self):
        import getpass
        current_user = getpass.getuser()
        if os.path.exists(STATE_CACHE):
            try:
                with open(STATE_CACHE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.user_id = data.get("user_id", current_user)
                    self.conversation_id = data.get("conversation_id", str(uuid.uuid4()))
                    self.thread_id = self.conversation_id
                    self.state = {"messages": [], "user_id": self.user_id}
                    return
            except Exception:
                pass
        self.user_id = current_user
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
            os.replace(tmp_path, STATE_CACHE)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _register_commands(self):
        self.command_map = {
            "/help": CLICommand("/help", self._cmd_help, "显示 TUI 帮助指南"),
            "/quit": CLICommand("/quit", self._cmd_quit, "安全退出工作台"),
            "/reset": CLICommand("/reset", self._cmd_reset, "清空当前 Session 记忆"),
            "/pwd": CLICommand("/pwd", self._cmd_pwd, "显示当前工作目录"),
            "/cd": CLICommand("/cd", self._cmd_cd, "切换 CWD 目录: /cd <路径>"),
            "/ls": CLICommand("/ls", self._cmd_ls, "浏览目录结构: /ls [路径]"),
            "/cat": CLICommand("/cat", self._cmd_cat, "预览文本文件: /cat <文件>"),
            "/history": CLICommand("/history", self._cmd_history, "查看当前消息快照栈"),
        }

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
        mime = mime or "application_octet_stream"
        if path.suffix.lower() in TEXT_EXTENSIONS:
            doc_content = self._read_document_for_prompt(path)
            contents.append(MessageContent(type="text", text=doc_content))
            return
        contents.append(MessageContent(type="file", file=FileObject(file_id=str(path), mime_type=mime)))

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
                        contents.append(MessageContent(type="text", text=self._directory_snapshot_for_prompt(maybe_path)))
                    else:
                        self._append_file_content(maybe_path, contents)
                else:
                    text_buffer.append(t)
        flush_text()
        return contents

    def _prepare_input_data(self) -> List[Dict[str, Any]]:
        return [{"role": "user", "content": m.content} for m in self.state["messages"] if hasattr(m, "content")]


if __name__ == "__main__":
    app = DeepAgentTUI()
    app.run()