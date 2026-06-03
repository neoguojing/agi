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
from textual.widgets import Header, Footer, Input, Markdown, Static,OptionList
from textual.containers import VerticalScroll
from textual import work, on
from textual.suggester import Suggester
from textual.binding import Binding
from textual.widgets.option_list import Option
from textual.containers import Container
from textual.geometry import Offset

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
from agi.config import LANGGRAPH_MAIN_URL

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
# 🔮 冒号前缀智能提示器 (保留 Zsh 行内暗影)
# =====================================================================
class AgentSuggester(Suggester):
    def __init__(self, command_input: "CommandInput"):
        super().__init__(use_cache=False)
        self.cmd_input = command_input

    async def get_suggestion(self, value: str) -> str | None:
        if not value: return None
        if value.startswith(":") and " " not in value:
            cmds = getattr(self.cmd_input.app, "command_map", {})
            for cmd in cmds.keys():
                if cmd.startswith(value) and cmd != value: return cmd
        if not value.startswith(":"):
            for hist in reversed(self.cmd_input.history):
                if hist.startswith(value) and hist != value: return hist
        return None


# =====================================================================
# ⚙️ 支持 Tab 呼出提示选项菜单的 CommandInput
# =====================================================================
class CommandInput(Input):
    """支持 Tab 弹出多选项菜单、原生路径补全的现代输入框"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history: List[str] = []
        self.history_index: int = -1
        self._load_history()
        self.suggester = AgentSuggester(self)
        
        # 🌟 修复方案：初始化时作为一个不属于 DOM 树的变量引用，或者先声明
        self.menu: Optional[OptionList] = None

    def on_mount(self) -> None:
        """在组件挂载到 App 时，一次性提前初始化悬浮提示菜单"""
        self.menu = OptionList(id="completion-menu")
        
        # 预设现代悬浮样式
        self.menu.styles.layer = "above"
        self.menu.styles.dock = "bottom"
        self.menu.styles.margin = (0, 2, 4, 2)
        self.menu.styles.max_height = 6
        self.menu.styles.border = ("panel", "cyan")
        self.menu.styles.background = "#1e1e1e"
        
        # 🌟 核心：默认隐藏并直接挂载到 App 树
        self.menu.visible = False
        self.app.mount(self.menu)

    def _show_menu(self, items: List[str], prefix: str, is_cmd: bool):
        """动态复用现有的提示选项菜单，杜绝重复插入 DOM 导致的 ID 冲突"""
        if not self.menu:
            return

        # 保存补全上下文信息
        self._menu_prefix = prefix
        self._menu_is_cmd = is_cmd

        # 🌟 修复：直接清空老数据，并注入新匹配到的候选数据
        self.menu.clear_options()
        for item in items:
            self.menu.add_option(Option(item, id=item))
            
        # 展现菜单并强行抢占焦点
        self.menu.visible = True
        self.menu.focus()

    def _close_menu(self):
        """关闭菜单只需将其隐藏，并将焦点送回输入框"""
        if self.menu and self.menu.visible:
            self.menu.visible = False
            self.focus()

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
        # 🌟 核心修复：当菜单可见时，精确劫持回车事件
        if self.menu and self.menu.visible:
            if event.key == "escape":
                self._close_menu()
                event.stop()
                event.prevent_default()
                return
                
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                
                # 🌟 主动获取当前菜单中被高亮选中的索引和选项
                idx = self.menu.highlighted
                if idx is not None:
                    # 从菜单的 _options 中安全提取选中的 Option 对象
                    option = self.menu._options[idx]
                    selected_text = str(option.id)
                    
                    # 立即执行补全并关闭菜单
                    self._apply_completion(selected_text, self._menu_prefix, self._menu_is_cmd)
                    self._close_menu()
                return
                
            if event.key in ("up", "down"):
                # 放行让 OptionList 自身处理高亮上下移动
                return

        # ---- 下面是你原有的 Tab 和 History 逻辑 ----
        # if event.key == "tab":
        #     event.stop()
        #     event.prevent_default()
        #     self._handle_tab_completion()
        #     return

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

        # 区分命令还是路径
        is_cmd = raw_val.startswith(":") and " " not in raw_val
        candidates = []
        prefix = ""

        if is_cmd:
            cmds = list(getattr(self.app, "command_map", {}).keys())
            candidates = [c for c in cmds if c.startswith(raw_val)]
            prefix = raw_val
        else:
            # 路径解析
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
                    for item in search_dir.iterdir():
                        if item.name.startswith(prefix):
                            suffix = "/" if item.is_dir() else " "
                            candidates.append(item.name + suffix)
            except Exception as e:
                self.app.log.error(f"Menu Scan Error: {e}")

        if not candidates:
            self._close_menu()
            return

        # ---- 核心分支：单项直接补全，多项弹出提示选项 ----
        if len(candidates) == 1:
            self._close_menu()
            self._apply_completion(candidates[0], prefix, is_cmd)
        else:
            # 存在多个选项，展示悬浮提示菜单
            self._show_menu(candidates, prefix, is_cmd)

    def _apply_completion(self, match_result: str, prefix: str, is_cmd: bool):
        """精准计算并替换输入框最后的 Token 文本"""
        raw_val = self.value
        
        if is_cmd:
            self.value = match_result + " "
        else:
            # 🌟 智能路径拼接算法：
            # 如果末尾有输入前缀 (如输入 agi/con，按 Tab 弹窗选了 console.py )
            if prefix:
                # 斩断末尾的不完整前缀，拼上完整匹配项
                self.value = raw_val[:-len(prefix)] + match_result
            else:
                # 如果没有前缀 (如输入 agi/，按 Tab 弹窗选了 utils/)
                # 直接追加匹配项
                self.value = raw_val + match_result
                
        # 强行刷新光标位置到最后，并通知组件内容已重绘
        self.cursor_position = len(self.value)

    # 🌟 监听提示菜单的选择锁定事件
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """当用户在菜单中按下回车或点击某一项时触发"""
        event.stop()  # 阻止事件继续向上传递
        
        # 获取用户选中的文本（即我们在 _show_menu 里塞进去的 Option(item, id=item)）
        selected_text = str(event.option_id)
        
        # 调用补全应用函数，把文本追加/替换到输入框中
        self._apply_completion(selected_text, self._menu_prefix, self._menu_is_cmd)
        
        # 自动关闭菜单，焦点回到输入框
        self._close_menu()


# =====================================================================
# 2. 🔮 完美的云端事件常驻监听服务 (保留你的设计灵魂)
# =====================================================================
class CloudLifecycleManager:
    """全局常驻的云端事件监听服务：负责维护长连接，具备断线自动重连机制"""
    def __init__(self, client, thread_id: str, assistant_id: str, event_queue: asyncio.Queue):
        self.client = client
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.event_queue = event_queue
        
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
                await self.event_queue.put(StreamError(RuntimeError(f"云端连接异常跌落，正在尝试重连... (Error: {e})")))
                await asyncio.sleep(3)

    async def submit(self, input_data: dict[str, Any]):
        """修复：移除内部多余的 STREAM_DONE 投递，只关注数据层交互"""
        await self.ready.wait()
        await self.client.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                input=input_data,
                stream_mode=["messages"]
            )


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
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "退出系统", show=True),
        Binding("ctrl+l", "clear_screen", "清屏", show=True),
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
        
        self.is_cloud_mode = bool(LANGGRAPH_MAIN_URL and self.assistant_id)
        # self.is_cloud_mode = False
        self.cloud_manager: Optional[CloudLifecycleManager] = None
        
        self.event_queue = asyncio.Queue()
        self.command_map: Dict[str, CLICommand] = {}
        self._register_commands()

        self._mounted_widgets: Dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield VerticalScroll(id="chat-container")
        yield CommandInput(placeholder="输入提示词或命令 (如 :help, :cd)...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        self.container = self.query_one("#chat-container", VerticalScroll)
        self.input_box = self.query_one("#chat-input", CommandInput)
        self._update_status_bar()
        
        self.persistent_consumer()
        
        if self.is_cloud_mode:
            self.run_worker(self.start_global_cloud_listener(), thread=False)
        else:
            self.container.mount(Static("[bold yellow]ℹ️ 当前运行于：本地直连 Mock 模式[/bold yellow]"))
        
        self.input_box.focus()

    def _update_status_bar(self):
        self.sub_title = f"📁 目录: {self.cwd.name} | 🧵 线程: {self.thread_id[:8]}"

    # =====================================================================
    # 后台常驻统一消费者
    # =====================================================================
    @work(group="consumers", exclusive=False)
    async def persistent_consumer(self) -> None:
        processor = StreamProcessor()
        # 清空 TUI 挂载组件缓存
        self._mounted_widgets = {}
        
        while True:
            event = await self.event_queue.get()

            if event is STREAM_START:
                processor = StreamProcessor()
                self._mounted_widgets = {}
                
                # 初始化顶层全局状态标签
                self.current_status_badge = Static("⚙️ [dim]Agent 正在整理思绪...[/dim]", classes="tool-badge")
                # 🌟 修复：mount 是同步函数，去掉 await 确保 DOM 树原子级挂载成功
                self.container.mount(self.current_status_badge)
                self.container.scroll_end(animate=False)
                continue
                
            if event is STREAM_DONE:
                if self.current_status_badge:
                    self.current_status_badge.update("✅ [bold green]响应完成[/bold green]")
                
                elapsed = time.time() - processor.start_time
                stats = processor.stats
                tps = stats.output_tokens / max(elapsed, 1e-6) if stats.output_tokens else 0.0
                
                # 1. 精美终端页脚统计
                stats_footer = Static(
                    f"\n[dim]⏱️ 耗时: [cyan]{elapsed:.1f}s[/cyan]  |  🚀 速度: [magenta]{tps:.1f} t/s[/magenta]  "
                    f"|  ⬇️ Input: [yellow]{stats.input_tokens}[/yellow]  |  ⬆️ Output: [green]{stats.output_tokens}[/green][/dim]\n"
                )
                self.container.mount(stats_footer)
                
                # 🌟 2. 【核心修复】：改用纯 Rich 富文本的 Static 模拟大师级分割线
                divider = Static("[dim]─" * 20 + " EOF (End of Turn) " + "─" * 20 + "[/dim]")
                divider.styles.text_align = "center"  # 让虚线和文字在终端里绝对居中
                divider.styles.margin = (1, 0, 2, 0) # 上边距 1 行，下边距 2 行，拉开呼吸感
                
                self.container.mount(divider)
                
                # 3. 顺滑滚动并落盘
                self.container.scroll_end(animate=True)
                self._save_session()
                continue
                
            if isinstance(event, StreamError):
                if self.current_status_badge:
                    self.current_status_badge.update("❌ [bold red]流式连接异常[/bold red]")
                # 🌟 修复：去掉这里的 await
                self.container.mount(Static(f"[bold red]错误提示: {event.exc}[/bold red]"))
                self.container.scroll_end(animate=True)
                continue

            try:
                # 1. 喂入处理器，并获取归一化事件快照
                stream_ev = processor.process_part(event)
                if not stream_ev:
                    continue

                # 2. 精准更新顶部状态徽章
                current_node = stream_ev.stats.node
                if self.current_status_badge and current_node and current_node != "N/A":
                    self.current_status_badge.update(f"⚙️ [bold yellow]当前步骤: {current_node}[/bold yellow] ...")

                # 3. 消费结构化关联日志链
                for item in stream_ev.structured_logs:
                    widget_key = f"{item.category}_{item.title}"
                    
                    # 💡 就在此渲染层按需格式化。既让名称与内容严格换行，又将样式彻底从数据层剥离
                    if item.category == "Tool":
                        # 格式要求：第一行 func(arg)，换行紧跟使用 [dim]（微弱注释字）包裹的返回值
                        display_text = f"[bold cyan]{item.title}[/bold cyan]\n"
                        if item.detail.strip():
                            display_text += f"[dim]{item.detail.strip()}[/dim]"
                        else:
                            display_text += "[dim]  * 正在等待工具响应...[/dim]"
                    elif item.category == "Final_Output":
                        # 格式要求：名称和内容之间换行，内容使用柔和对比度展示
                        display_text = f"[bold green]{item.title}[/bold green]\n{item.detail.strip()}"
                    else:
                        display_text = f"[bold yellow]{item.title}[/bold yellow]\n[dim]{item.detail.strip()}[/dim]"

                    # 挂载控制
                    if widget_key not in self._mounted_widgets:
                        # 第一次见，挂载通用的 Textual Static 组件
                        new_widget = Static(display_text, classes="pipeline-node-card")
                        self._mounted_widgets[widget_key] = new_widget
                        await self.container.mount(new_widget)
                        self.container.scroll_end(animate=False)
                    else:
                        existing_widget = self._mounted_widgets[widget_key]
                        # 原地就地刷新
                        if getattr(existing_widget, "_last_raw_text", "") != display_text:
                            existing_widget.update(display_text)
                            existing_widget._last_raw_text = display_text
                            
                            if stream_ev.is_delta and item.category == "Final_Output":
                                self.container.scroll_end(animate=False)

            except Exception as e:
                import traceback
                self.log.error(f"Render Layer Crash: {traceback.format_exc()}")
                # 🌟 修复：去掉这里的 await
                self.container.mount(Static(f"[bold red]终端渲染层故障: {e}[/bold red]"))
                self.container.scroll_end(animate=True)

    async def start_global_cloud_listener(self) -> None:
        try:
            self.cloud_manager = CloudLifecycleManager(
                client=self.client,
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                event_queue=self.event_queue
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
            self.log.error(e)
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