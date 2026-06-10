import asyncio
import io
import json
import mimetypes
import logging
import os
import shlex
import sys
import time
import traceback
import uuid
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---- Textual 核心组件 ----
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Markdown, Static,OptionList,RichLog
from textual.containers import VerticalScroll
from textual import work, on
from textual.suggester import Suggester
from textual.binding import Binding
from textual.widgets.option_list import Option
from textual.containers import Container
from textual.geometry import Offset
from textual.containers import Vertical

# ---- Rich 美化组件 ----
from rich.panel import Panel

# ---- 外部依赖接口 ----
from langgraph_sdk import get_client
from langgraph.graph.message import add_messages

from agi.agent.agent import stream_agent_async
from agi.agent.context import Context
from agi.agent.stream_processor import StreamProcessor
from agi.api.media import process_multimodal_content
from agi.apps.common import FileObject, ImageURL, MessageContent
from agi.config import LANGGRAPH_MAIN_URL,CLOUD_MODE

STATE_CACHE = ".cli_session.json"
PROMPT_HISTORY_CACHE = ".cli_prompt_history"
MAX_INLINE_DOC_CHARS = 20000
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv", ".log", ".xml", ".html", ".rst"}

STREAM_START = object()
STREAM_DONE = object()

class StreamError:
    def __init__(self, exc: Exception):
        self.exc = exc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


@dataclass
class CLICommand:
    name: str
    handler: Callable[[str], bool]
    help_text: str


# =====================================================================
# 🪝 1. 拦截 Python Logger 体系的自定义 Handler
# =====================================================================
class TUIAppLoggingHandler(logging.Handler):
    """将整个 Python logging 模块的日志转发到 Textual TUI 的 RichLog 组件中"""
    def __init__(self, write_log_func):
        super().__init__()
        self.write_log_func = write_log_func

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                formatted_msg = f"[bold red][ERROR][/bold red] {msg}"
            elif record.levelno >= logging.WARNING:
                formatted_msg = f"[bold yellow][WARN][/bold yellow] {msg}"
            elif record.levelno >= logging.INFO:
                formatted_msg = f"[bold cyan][INFO][/bold cyan] {msg}"
            else:
                formatted_msg = f"[dim][DEBUG][/dim] {msg}"
            self.write_log_func(formatted_msg)
        except Exception:
            self.handleError(record)

# =====================================================================
# 🪝 2. 拦截 print() 和 sys.stderr 的虚拟流对象
# =====================================================================
class TUIStreamRedirector:
    """伪装成 file-like 对象，把系统的 print() 和报错堆栈接管过来"""
    def __init__(self, write_log_func, prefix="[PRINT]"):
        self.write_log_func = write_log_func
        self.prefix = prefix

    def write(self, buffer: str):
        for line in buffer.rstrip().splitlines():
            if line.strip():
                if "stderr" in self.prefix.lower():
                    self.write_log_func(f"[bold red]{self.prefix}[/bold red] {line}")
                else:
                    self.write_log_func(f"[dim]{self.prefix}[/dim] {line}")

    def flush(self):
        pass

# =====================================================================
# 1. 独立封装的自定义折叠 Markdown 卡片
# =====================================================================
class CollapsibleCard(Vertical):
    """时序安全、干净排版、支持最终输出不含折叠外壳的组件容器"""
    
    def __init__(self, raw_markdown: str, category: str, title: str, **kwargs):
        super().__init__(**kwargs)
        self.raw_markdown = raw_markdown
        self.category = category
        self.title = title
        
        # 建立内容组件
        css_class = "msg-agent" if category == "Final_Output" else "msg-pipeline-trace"
        self.content_widget = Markdown(raw_markdown, classes=css_class)
        
        # 🎯 分流处理：如果是最终输出，彻底不要折叠标题栏
        if category == "Final_Output":
            self.header_widget = None
            self.is_collapsed = False
            self.content_widget.styles.display = "block"
        else:
            # 工具/日志层：清洗乱码并截断，默认折叠
            clean_title = re.sub(r'\[\/?[a-zA-Z0-9 #_=-]+\]', '', title)
            self.display_title = clean_title if len(clean_title) <= 40 else clean_title[:40] + "..."
            
            self.is_collapsed = True
            self.header_widget = Static(f"▶️ [bold cyan]{self.display_title}[/bold cyan]", classes="fold-header")
            self.content_widget.styles.display = "none"

    def compose(self):
        # 🎯 如果 header_widget 存在才挂载（Final_Output 此时不会挂载标题栏）
        if self.header_widget:
            yield self.header_widget
        yield self.content_widget

    def toggle(self) -> None:
        """切换折叠/展开状态（最终输出不参与折叠）"""
        if self.category == "Final_Output" or not self.header_widget:
            return
            
        if self.is_collapsed:
            self.content_widget.styles.display = "block"
            self.header_widget.update(f"▼ [bold yellow]{self.display_title}[/bold yellow]")
            self.is_collapsed = False
        else:
            self.content_widget.styles.display = "none"
            self.header_widget.update(f"▶️ [bold cyan]{self.display_title}[/bold cyan]")
            self.is_collapsed = True
            
        self.refresh(layout=True)

# =====================================================================
# ⚙️ 极简高效率单项直出 CommandInput (已拔除下拉菜单)
# =====================================================================
class CommandInput(Input):
    """支持快捷键历史、Zsh 式单项首选直接补全的轻量输入框"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: List[str] = []
        self.history_index: int = -1
        self._load_history()

    def _load_history(self):
        try:
            if os.path.exists(PROMPT_HISTORY_CACHE):
                with open(PROMPT_HISTORY_CACHE, "r", encoding="utf-8") as f:
                    self.history = [line.strip() for line in f if line.strip()]
        except Exception: pass

    def append_history(self, text: str):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            try:
                with open(PROMPT_HISTORY_CACHE, "w", encoding="utf-8") as f:
                    for line in self.history: f.write(line + "\n")
            except Exception: pass
        self.history_index = -1

    def on_key(self, event) -> None:
        # 🌟 劫持 Tab 键，触发单项首选补全
        if event.key == "tab":
            event.stop()
            event.prevent_default()
            self._handle_tab_completion()
            return

        elif event.key == "up":
            event.stop()
            if self.history:
                if self.history_index == -1:
                    self._search_prefix = self.value
                    filtered = [h for h in self.history if h.startswith(self._search_prefix)]
                    self._filtered_history = filtered if filtered else self.history
                    self.history_index = len(self._filtered_history) - 1
                elif self.history_index > 0:
                    self.history_index -= 1
                
                if self._filtered_history:
                    self.value = self._filtered_history[self.history_index]
                    self.cursor_position = len(self.value)
            
        elif event.key == "down":
            event.stop()
            if self.history and self.history_index != -1:
                if self.history_index < len(self._filtered_history) - 1:
                    self.history_index += 1
                    self.value = self._filtered_history[self.history_index]
                else:
                    self.history_index = -1
                    self.value = getattr(self, "_search_prefix", "")
                self.cursor_position = len(self.value)

    def _handle_tab_completion(self) -> None:
        raw_val = self.value
        if not raw_val: return

        is_cmd = raw_val.startswith(":") and " " not in raw_val
        candidates = []
        prefix = ""

        if is_cmd:
            cmds = list(getattr(self.app, "command_map", {}).keys())
            candidates = [c for c in cmds if c.startswith(raw_val)]
            prefix = raw_val
        else:
            # 路径解析基块
            if raw_val.endswith(" "): last_token = ""
            else:
                try: last_token = shlex.split(raw_val)[-1]
                except Exception: last_token = raw_val.split()[-1] if raw_val.split() else ""

            current_cwd = getattr(self.app, "cwd", Path.cwd())
            try:
                if last_token:
                    p = Path(os.path.expanduser(last_token))
                    if last_token.endswith(("/", "\\")):
                        search_dir = p if p.is_absolute() else current_cwd / p
                        prefix = ""
                    else:
                        search_dir = p.parent if p.is_absolute() else current_cwd / p.parent
                        prefix = p.name
                else:
                    search_dir = current_cwd
                    prefix = ""

                if search_dir.exists() and search_dir.is_dir():
                    # 对结果做字母序排序，确保匹配表现稳定可预测
                    for item in sorted(search_dir.iterdir(), key=lambda x: x.name):
                        if item.name.startswith(prefix):
                            suffix = "/" if item.is_dir() else " "
                            candidates.append(item.name + suffix)
            except Exception as e:
                self.app.log.error(f"Scan Directory Error: {e}")

        # 🎯 核心优化：只要有候选，直接无脑把第一个（最匹配的）丢给应用层，绝不弹窗
        if candidates:
            self._apply_completion(candidates[0], prefix, is_cmd)

    def _apply_completion(self, match_result: str, prefix: str, is_cmd: bool):
        """精准计算并替换输入框最后的 Token 文本"""
        raw_val = self.value
        
        if is_cmd:
            self.value = match_result + " "
        else:
            # 智能路径拼接算法：
            if prefix:
                # 斩断末尾的不完整前缀，拼上完整匹配项
                self.value = raw_val[:-len(prefix)] + match_result
            else:
                # 如果没有前缀 (如输入 agi/) 直接追加首选匹配项
                self.value = raw_val + match_result
                
        # 强制通知 DOM 刷新光标到文本最末端
        self.cursor_position = len(self.value)

# =====================================================================
# 2. 🔮 完美的云端事件常驻监听服务 (保留你的设计灵魂)
# =====================================================================
class CloudLifecycleManager:
    """全局常驻的云端事件监听服务：负责维护长连接，具备断线自动重连机制"""
    def __init__(self, client, thread_id: str, assistant_id: str, event_queue: asyncio.Queue,logger: Any):
        self.client = client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.event_queue = event_queue
        self.log = logger
        
        self.ready = asyncio.Event()

    def is_run_done(self,event_stream) -> bool:
        """
        解析状态标记，判断运行是否完成。
        
        :param event_stream: 包含事件元数据的对象或字典
        :return: 如果状态为 'run_done' 返回 True，否则返回 False
        """
        # 兼容性处理：如果传入的是对象，先尝试获取它的 data 属性；如果本身就是字典，直接读取
        data = getattr(event_stream, 'data', None)
        
        if data is None and isinstance(event_stream, dict):
            data = event_stream.get('data', {})
            
        # 安全地获取 status 并进行比对
        if isinstance(data, dict):
            return data.get('status') == 'run_done'
            
        return False

    async def run_forever(self):
        await self.client.threads.create(thread_id=self.thread_id,graph_id="main",if_exists="do_nothing")

        while True:
            try:
                stream = await self.client.threads.join_stream(
                    thread_id=self.thread_id
                )
                self.ready.set()
                async for event in stream:
                    if self.is_run_done(event):
                        await self.event_queue.put(STREAM_DONE)
                    await self.event_queue.put(event)
            except Exception as e:
                self.log.error(f"run_forever:{e}")
                await self.event_queue.put(StreamError(RuntimeError(f"云端连接异常跌落，正在尝试重连... (Error: {e})")))
                await asyncio.sleep(3)

    async def submit(self, input_data: dict[str, Any]):
        """修复：移除内部多余的 STREAM_DONE 投递，只关注数据层交互"""
        await self.ready.wait()
        try:
            await self.client.runs.create(
                    thread_id=self.thread_id,
                    assistant_id=self.assistant_id,
                    input=input_data,
                    stream_mode=["messages"]
                )
        except Exception as e:
                self.log.error(f"submit:{e}")
            


# =====================================================================
# 3. Textual 现代 TUI 展现层
# =====================================================================
class DeepAgentTUI(App):
    TITLE = "DeepAgent Workspace"
    
    CSS = """
    #chat-container {
        height: 1fr;
        border: solid cyan;
        padding: 0 1;
        overflow-y: scroll;
        background: $surface;
    }
    #system-log {
        dock: right;
        width: 50%;              /* 🌟 核心：右侧 50% 宽度分屏 */
        display: none;          /* 默认隐藏，通过命令/快捷键唤醒 */
        border-left: tall magenta;
        background: $surface;
        color: $text;
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
    CommandInput {
        dock: bottom;
        margin: 1 0 0 0;
        border: tall double gray;
    }
    CommandInput:focus {
        border: tall double cyan;
    }
    CollapsibleCard {
        margin: 1 0;
        border: none;
        height: auto;
    }
    .fold-header {
        background: $boost;
        padding: 0 1;
        color: $text;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "退出系统", show=True),
        Binding("ctrl+l", "clear_screen", "清屏", show=True),
        Binding("ctrl+o", "toggle_all_collapse", "展开/折叠最新内容", show=True),
        Binding("ctrl+b", "toggle_log_panel", "显示/隐藏控制台", show=True),
    ]

    def __init__(self):
        super().__init__()
        self.cwd = Path.cwd()
        
        # 🛡️ 修复：在 init 里显式补全所有缺失属性，彻底规避 AttributeError
        self.current_status_badge: Optional[Static] = None
        self.current_agent_markdown: Optional[Markdown] = None
        self.user_id: str = ""
        self.conversation_id: str = ""
        self.thread_id: str = ""
        self.state: Dict[str, Any] = {"messages": []}
        
        self.load_session()
        self.client = get_client(url=LANGGRAPH_MAIN_URL)
        self.assistant_id = "main"
        
        self.is_cloud_mode = CLOUD_MODE
        self.cloud_manager: Optional[CloudLifecycleManager] = None
        
        self.event_queue = asyncio.Queue()
        self.command_map: Dict[str, CLICommand] = {}
        self._register_commands()

        self._mounted_widgets: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield RichLog(id="system-log", highlight=True, markup=True, wrap=True, min_width=0)
        yield CommandInput(placeholder="输入提示词或命令 (如 :help, :cd)...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.container = self.query_one("#chat-container", VerticalScroll)
        self.input_box = self.query_one("#chat-input", CommandInput)
        self.log_panel = self.query_one("#system-log", RichLog)
        self._update_status_bar()
        
        # =====================================================================
        # 🔥 全局日志与 Print 管道全量劫持
        # =====================================================================
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        tui_handler = TUIAppLoggingHandler(self.write_log)
        tui_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
        root_logger.addHandler(tui_handler)
        root_logger.setLevel(logging.INFO) 

        sys.stdout = TUIStreamRedirector(self.write_log, prefix="[STDOUT]")
        sys.stderr = TUIStreamRedirector(self.write_log, prefix="[STDERR]")
        # =====================================================================

        self.persistent_consumer()
        
        if self.is_cloud_mode:
            self.run_worker(self.start_global_cloud_listener(), thread=False)
        else:
            self.container.mount(Static("[bold yellow]ℹ️ 当前运行于：本地直连 Mock 模式[/bold yellow]"))
        
        self.write_log("[bold green][SYSTEM][/bold green] 日志管道全量接管成功。所有的 print 和 logger 均已重定向至此。")

        self.input_box.focus()

    def _update_status_bar(self):
        self.sub_title = f"📁 目录: {self.cwd.name} | 🧵 线程: {self.thread_id[:8]}"

    def write_log(self, message: str) -> None:
        """供系统各个后台任务随时调用，向右侧控制台追加原生格式化日志"""
        try:
            if hasattr(self, "log_panel") and self.log_panel:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_panel.write(f"[dim]\[{timestamp}][/dim] {message}")
        except Exception:
            pass
    # =====================================================================
    # 🌟 快捷键触发动作 (Action)
    # =====================================================================
    def action_toggle_all_collapse(self) -> None:
        """当用户按下 Ctrl+O 时执行的动作"""
        # 策略 A：折叠或展开【当前最后一轮对话】的所有卡片（最符合日常使用直觉）
        if self._mounted_widgets:
            # 拿到最近更新的卡片列表，对其状态进行切换
            for card in self._mounted_widgets.values():
                if isinstance(card, CollapsibleCard):
                    card.toggle()
            
            # 状态切换完毕后，顺滑贴底滚动
            self.container.scroll_end(animate=True)

    def action_toggle_log_panel(self) -> None:
        """按 Ctrl+B 在三种视窗大小间无缝循环"""
        current_width = self.log_panel.styles.width.value if self.log_panel.display else 0
        
        if current_width == 0:
            self.log_panel.display = True
            self.log_panel.styles.width = "30%"  # 🤏 瘦子模式
        elif current_width == 30:
            self.log_panel.styles.width = "70%"  # 🐋 胖子模式（左边聊天窗自动被挤压到30%）
        else:
            self.log_panel.display = False       # ❌ 隐藏模式
            
        self.container.scroll_end(animate=False)
    # =====================================================================
    # 后台常驻统一消费者
    # =====================================================================
    @work(group="consumers", exclusive=False)
    async def persistent_consumer(self) -> None:
        processor = StreamProcessor()
        self._mounted_widgets = {}
        
        while True:
            event = await self.event_queue.get()

            if event is STREAM_START:
                processor = StreamProcessor()
                self._mounted_widgets = {}
                
                self.current_status_badge = Static("⚙️ [dim]Agent 正在整理思绪...[/dim]", classes="tool-badge")
                self.container.mount(self.current_status_badge)
                self.container.scroll_end(animate=False)
                continue
                
            if event is STREAM_DONE:
                if self.current_status_badge:
                    self.current_status_badge.update("✅ [bold green]响应完成[/bold green]")
                
                elapsed = time.time() - processor.start_time
                stats = processor.stats
                tps = stats.output_tokens / max(elapsed, 1e-6) if stats.output_tokens else 0.0
                
                stats_footer = Static(
                    f"\n[dim]⏱️ 耗时: [cyan]{elapsed:.1f}s[/cyan]  |  🚀 速度: [magenta]{tps:.1f} t/s[/magenta]  "
                    f"|  ⬇️ Input: [yellow]{stats.input_tokens}[/yellow]  |  ⬆️ Output: [green]{stats.output_tokens}[/green][/dim]\n"
                )
                self.container.mount(stats_footer)
                
                divider = Static("[dim]─" * 20 + " EOF (End of Turn) " + "─" * 20 + "[/dim]")
                divider.styles.text_align = "center"
                divider.styles.margin = (1, 0, 2, 0)
                self.container.mount(divider)
                
                self.container.scroll_end(animate=True)
                self._save_session()
                continue
                
            if isinstance(event, StreamError):
                if self.current_status_badge:
                    self.current_status_badge.update("❌ [bold red]流式连接异常[/bold red]")
                self.container.mount(Static(f"[bold red]错误提示: {event.exc}[/bold red]"))
                self.container.scroll_end(animate=True)
                continue

            try:
                self.log.info(f"**********8{event}")
                stream_ev = processor.process_part(event)
                if not stream_ev:
                    continue

                current_node = stream_ev.stats.node
                if self.current_status_badge and current_node and current_node != "N/A":
                    self.current_status_badge.update(f"⚙️ [bold yellow]当前步骤: {current_node}[/bold yellow] ...")

                for item in stream_ev.structured_logs:
                    widget_key = f"{item.category}_{item.title}"
                    
                    if item.category == "Tool":
                        if item.detail.strip():
                            display_text = f"{item.detail.strip()}"
                        else:
                            display_text = "[dim]  * 正在等待工具响应...[/dim]"
                    else:
                        display_text = f"{item.detail.strip()}"

                    if widget_key not in self._mounted_widgets:
                        new_card = CollapsibleCard(
                            raw_markdown=display_text, 
                            category=item.category, 
                            title=item.title
                        )
                        self._mounted_widgets[widget_key] = new_card
                        self.container.mount(new_card)
                        self.container.scroll_end(animate=False)
                    else:
                        existing_card = self._mounted_widgets[widget_key]
                        existing_widget = existing_card.content_widget
                        
                        if getattr(existing_widget, "_last_raw_text", "") != display_text:
                            existing_widget.update(display_text)
                            existing_widget._last_raw_text = display_text
                            
                            # Final_Output 直接吐字更新并丝滑贴底
                            if stream_ev.is_delta and item.category == "Final_Output":
                                self.container.scroll_end(animate=False)

            except Exception as e:
                import traceback
                self.log.error(f"Render Layer Crash: {traceback.format_exc()}")
                self.container.mount(Static(f"[bold red]终端渲染层故障: {e}[/bold red]"))
                self.container.scroll_end(animate=True)

    async def start_global_cloud_listener(self) -> None:
        try:
            self.cloud_manager = CloudLifecycleManager(
                client=self.client,
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                event_queue=self.event_queue,
                logger=self.log
            )
            await self.cloud_manager.run_forever()
        except Exception as e:
            self.log.error(e)

    @on(Input.Submitted, "#chat-input")
    async def handle_input_event(self, event: Input.Submitted):
        user_input = event.value.strip()
        if not user_input: return
        
        if hasattr(event.input, "append_history"):
            event.input.append_history(user_input)
            
        event.input.value = ""
        self.container.mount(Markdown(f"👤 **You >** {user_input}", classes="msg-user"))
        self.container.scroll_end(animate=False)

        command_result = self._parse_command(user_input)
        if command_result is not None:
            if command_result is False: self.exit()
            return

        try:
            human_message = process_multimodal_content(self._smart_parse(user_input))
            self.state["messages"] = add_messages(self.state["messages"], [human_message])
            
            if self.is_cloud_mode:
                self.trigger_cloud_run_task()
            else:
                self.trigger_local_run_task()
                
        except Exception as e:
            await self.container.mount(Static(f"[red]输入解析失败: {e}[/red]"))

    @work(group="agent-tasks", exclusive=True)
    async def trigger_cloud_run_task(self) -> None:
        if not self.cloud_manager:
            await self.event_queue.put(StreamError(RuntimeError("云端监听服务未就绪")))
            return
            
        try:
            await self.event_queue.put(STREAM_START)
            
            input_data = {"messages": self._prepare_input_data()}
            # 踢出临门一脚，并在此处同步挂起，直到本次运行彻底返回
            await self.cloud_manager.submit(input_data)
            
            # 修复：有且仅有这里负责投递单次运行结束状态
            # await self.event_queue.put(STREAM_DONE)
        except Exception as e:
            self.log.error(f"{e}\n{traceback.format_exc()}")
            await self.event_queue.put(StreamError(e))

    @work(group="agent-tasks", exclusive=True)
    async def trigger_local_run_task(self) -> None:
        await self.event_queue.put(STREAM_START)
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            context = Context(user_id=self.user_id, conversation_id=self.conversation_id)
            input_messages = self._prepare_input_data()
            if not input_messages:
                # 及时抛出错误，看是不是这里把输入吞了
                raise ValueError("本地准备投递的 messages 上下文为空，请检查消息流状态！")
            input_data = {"messages": input_messages}
            async for part in stream_agent_async(input_data,config=config, context=context, stream_mode=["messages", "updates"]):
                await self.event_queue.put(part)
            await self.event_queue.put(STREAM_DONE)
        except Exception as e:
            self.log.error(f"{e}\n{traceback.format_exc()}")
            
            await self.event_queue.put(StreamError(e))

    # =====================================================================
    # 内置基础路由与辅助逻辑修复
    # =====================================================================
    def action_clear_screen(self) -> None:
        self.container.query("*").remove()
        self.container.mount(Static("[dim]会话大底已重置清空[/dim]"))
        self.container.scroll_home(animate=False)

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
        try:
            target = self._resolve_path(arg) if arg else Path.home()
            if not target.exists() or not target.is_dir():
                self.container.mount(Static(f"[red]路径不存在: {target}[/red]"))
                return True
            self.cwd = target
            os.chdir(target)
            self._update_status_bar()
            self.container.mount(Static(f"[green]已下潜至新目录: {target}[/green]"))
        except Exception as e:
            self.container.mount(Static(f"[red]切换失败: {e}[/red]"))
        self.container.scroll_end()
        return True

    def _cmd_ls(self, arg: str) -> bool:
        try:
            target = self._resolve_path(arg) if arg else self.cwd
            if not target.exists() or not target.is_dir():
                self.container.mount(Static(f"[red]路径异常: {target}[/red]"))
                return True
            rows = []
            for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                mark = "📁" if p.is_dir() else "📄"
                rows.append(f"{mark} {p.name}")
            self.container.mount(Static(Panel("\n".join(rows) if rows else "(空目录)", title=f"ls {target.name}")))
        except Exception as e:
            self.container.mount(Static(f"[red]读取失败: {e}[/red]"))
        self.container.scroll_end()
        return True

    def _cmd_cat(self, arg: str) -> bool:
        if not arg:
            self.container.mount(Static("[yellow]用法提示: :cat <文件名称>[/yellow]"))
            return True
        try:
            target = self._resolve_path(arg)
            if not target.exists() or not target.is_file():
                self.container.mount(Static(f"[red]目标文件不存在: {target}[/red]"))
                return True
            text = target.read_text(encoding="utf-8")
            self.container.mount(Markdown(f"```text\n{text}\n```"))
        except UnicodeDecodeError:
            self.container.mount(Static(f"[red]非标准 UTF-8 文本无法预览: {target}[/red]"))
        except Exception as e:
            self.container.mount(Static(f"[red]预览异常: {e}[/red]"))
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
                    return
            except Exception:
                pass
        self.user_id = current_user
        self.conversation_id = str(uuid.uuid4())
        self.thread_id = self.conversation_id

    def _save_session(self):
        tmp_path = STATE_CACHE + ".tmp"
        try:
            data = {"user_id": self.user_id, "thread_id": self.thread_id, "conversation_id": self.conversation_id}
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, STATE_CACHE)
        except Exception:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def _register_commands(self):
        # 🌟 指令注册全部改用冒号前缀
        self.command_map = {
            ":help": CLICommand(":help", self._cmd_help, "显示 TUI 帮助指南"),
            ":quit": CLICommand(":quit", self._cmd_quit, "安全退出工作台"),
            ":reset": CLICommand(":reset", self._cmd_reset, "清空当前 Session 记忆"),
            ":pwd": CLICommand(":pwd", self._cmd_pwd, "显示当前工作目录"),
            ":cd": CLICommand(":cd", self._cmd_cd, "切换 CWD 目录: :cd <路径>"),
            ":ls": CLICommand(":ls", self._cmd_ls, "浏览目录结构: :ls [路径]"),
            ":cat": CLICommand(":cat", self._cmd_cat, "预览文本文件: :cat <文件>"),
            ":history": CLICommand(":history", self._cmd_history, "查看当前消息快照栈"),
        }

    def _parse_command(self, user_input: str) -> Optional[bool]:
        stripped = user_input.lstrip()
        # 🌟 核心拦截机制：由斜杠改为冒号
        if not stripped.startswith(":"): return None
        parts = shlex.split(stripped)
        if not parts: return None
        cmd = parts[0].lower()
        handler = self.command_map.get(cmd)
        if not handler: return None
        arg = " ".join(parts[1:]) if len(parts) > 1 else ""
        return handler.handler(arg)

    def _cmd_help(self, _: str) -> bool:
        lines = [f"{k:<10} {v.help_text}" for k, v in self.command_map.items()]
        # 🌟 帮助提示同步精简，拿掉多余的前缀提示
        lines.append("\n📂 原生多模态: 直接输入本地文件/文件夹路径，系统会自动检索上下文并上传")
        self.container.mount(Static(Panel("\n".join(lines), title="帮助菜单", border_style="cyan")))
        self.container.scroll_end()
        return True

    def _resolve_path(self, raw_path: str) -> Path:
        """修复：独立对原生字符进行展开，完美解决 ~ 无法拼接问题"""
        expanded = os.path.expanduser(str(raw_path))
        p = Path(expanded)
        if p.is_absolute():
            return p.resolve()
        return (self.cwd / p).resolve()

    def _read_document_for_prompt(self, path: Path) -> str:
        if not path.exists() or not path.is_file(): return f"[文档不存在: {path}]"
        if path.suffix.lower() not in TEXT_EXTENSIONS: return f"[暂不支持直接读取该文档类型: {path}]"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError: return f"[文档非 UTF-8 编码: {path}]"
        if len(text) > MAX_INLINE_DOC_CHARS: text = text[:MAX_INLINE_DOC_CHARS] + "\n...\n[已截断]"
        return f"\n[DOC_BEGIN: {path}]\n{text}\n[DOC_END]\n"

    def _looks_like_existing_path(self, token: str) -> Optional[Path]:
        """修复：增加严格门槛拦截，非路径语法特征词绝不触发本地扫描"""
        if not token or token.startswith(("http://", "https://", "data:")): return None
        
        # 🛡️ 核心防御：如果一个词不包含斜杠、反斜杠或点，说明他大概率只是个普通日常词汇（如 build / log）
        if not any(char in token for char in ("/", "\\", ".")) and not os.path.isabs(token):
            return None
            
        try:
            candidate = self._resolve_path(token)
            if candidate.exists(): return candidate
        except Exception:
            pass
        return None

    def _directory_snapshot_for_prompt(self, path: Path) -> str:
        try: 
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except Exception as exc: 
            return f"[目录读取失败: {path}, error={exc}]"
        preview = [f"- {'DIR' if item.is_dir() else 'FILE'}: {item.name}" for item in entries[:200]]
        if len(entries) > 200: preview.append("- ... [已截断]")
        return f"\n[DIR_BEGIN: {path}]\n" + "\n".join(preview) + "\n[DIR_END]\n"

    def _append_file_content(self, path: Path, contents: List[MessageContent]):
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application_octet_stream"
        
        # 1. 文本类型依旧保持 Inline 注入
        if path.suffix.lower() in TEXT_EXTENSIONS:
            contents.append(MessageContent(type="text", text=self._read_document_for_prompt(path)))
            return
            
        # 🌟 修复冗余并补全多模态：如果是图片后缀，直接组装为标准 image_url
        if mime.startswith("image/"):
            contents.append(MessageContent(type="image_url", image_url=ImageURL(url=str(path))))
            return
            
        # 2. 其余未知或通用二进制类型（PDF、音频、视频等）降级为通用 file 挂载
        contents.append(MessageContent(type="file", file=FileObject(file_id=str(path), mime_type=mime)))

    def _smart_parse(self, text: str):
        # 使用 shlex.split 代替纯 split，防止路径里带空格时被无情切断
        try:
            tokens = shlex.split(text)
        except Exception:
            tokens = text.split()

        contents = []
        text_buffer = []

        def flush_text():
            if text_buffer:
                contents.append(MessageContent(type="text", text=" ".join(text_buffer)))
                text_buffer.clear()

        for t in tokens:
            # 1. 尝试直接进行原生路径/文件探测
            maybe_path = self._looks_like_existing_path(t)
            if maybe_path is not None:
                flush_text()
                if maybe_path.is_dir():
                    contents.append(MessageContent(type="text", text=self._directory_snapshot_for_prompt(maybe_path)))
                else:
                    self._append_file_content(maybe_path, contents)
            else:
                # 2. 普通对话文本归档
                text_buffer.append(t)
                
        flush_text()
        return contents

    def _prepare_input_data(self) -> List[Dict[str, Any]]:
        user_msgs = [m for m in self.state["messages"] if hasattr(m, "content")]
        if user_msgs:
            last_msg = user_msgs[-1]
            return [{"role": "user", "content": last_msg.content}]
        return []

if __name__ == "__main__":
    app = DeepAgentTUI()
    app.run()