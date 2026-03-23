import asyncio
import uuid
import os
import sys
import time
import subprocess
import signal
import json       # [修复 1] 补全导入
import mimetypes  # [修复 1] 补全导入
import tempfile   # [修复 2] 用于原子写入
import pathlib
from typing import List, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import traceback

# --- 配置 ---
IGNORED_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".idea", ".pytest_cache", "dist", "build"}
# [修复 4] 扩展监听文件类型，支持配置变更触发重启
WATCH_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".env"} 
STATE_CACHE = ".cli_session.json"
RESTART_DEBOUNCE_SECONDS = 1.5


# --- 1. CLI 逻辑部分 ---
def get_agent_cls():
    # 延迟导入重型依赖
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import PathCompleter
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.panel import Panel
    from rich.spinner import Spinner
    from prompt_toolkit.formatted_text import HTML 
    
    try:
        from agi.agent.agent import stream_agent_async
        from agi.agent.context import Context
        from agi.apps.common import MessageContent, ImageURL, FileObject
        from agi.api.media import process_multimodal_content
    except ImportError as e:
        print(f"\n❌ 错误: 无法导入 AGI 模块。\n详情: {e}")
        sys.exit(1)

    console = Console()

    class DeepAgentCLI:
        def __init__(self):
            self.load_session()
            self.session = PromptSession(completer=PathCompleter(expanduser=True))

        def load_session(self):
            if os.path.exists(STATE_CACHE):
                try:
                    with open(STATE_CACHE, 'r') as f:
                        data = json.load(f)
                        self.user_id = data.get("user_id", "admin")
                        self.conversation_id = data.get("conversation_id", str(uuid.uuid4()))
                        self.thread_id = self.conversation_id
                        self.state = {"messages": [], "user_id": self.user_id}
                        return
                except (json.JSONDecodeError, IOError):
                    # 文件损坏或读取失败，静默重置
                    pass
            self.user_id = "admin"
            self.conversation_id = str(uuid.uuid4())
            self.thread_id = self.conversation_id
            self.state = {"messages": []}

        def _save_session(self):
            """
            [修复 2] 原子写入会话状态
            防止写入过程中被中断导致 JSON 损坏
            """
            tmp_path = STATE_CACHE + ".tmp"
            try:
                data = {
                    "user_id": self.user_id,
                    "thread_id": self.thread_id,
                    "conversation_id": self.conversation_id
                }
                # 先写入临时文件
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno()) # 确保落盘
                
                # 原子替换原文件 (Windows 和 Linux 均支持)
                os.replace(tmp_path, STATE_CACHE)
            except Exception as e:
                # 清理可能残留的临时文件
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
                # 不抛出异常以免打断主流程，仅记录（实际项目中可用 logger）
                # print(f"⚠️ 保存会话失败: {e}")

        async def handle_stream(self, live):
            full_response = ""
            config = {"configurable": {"thread_id": self.thread_id}}
            context = Context(user_id=self.user_id, conversation_id=self.conversation_id)
            
            # [修复 5] UI 渲染限流变量
            last_update_time = 0
            update_interval = 0.05 # 50ms 刷新一次，平衡流畅度与 CPU

            async for part in stream_agent_async(self.state, config=config, context=context, stream_mode=["messages"]):
                if isinstance(part, dict) and part.get("type") == "messages":
                    data = part.get("data")
                    if data and len(data) > 0:
                        content = getattr(data[0], "content", "")
                        full_response += str(content)
                        
                        # [修复 5] 限流逻辑：避免高频更新导致 UI 抖动和 CPU 飙升
                        now = time.time()
                        if now - last_update_time > update_interval:
                            live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
                            last_update_time = now
            
            # 确保最后一次更新被渲染
            if full_response:
                live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
                
            return full_response

        def _smart_parse(self, text: str):
            tokens = text.split()
            contents = []
            for t in tokens:
                if t.startswith("img:"):
                    contents.append(MessageContent(type="image_url", image_url=ImageURL(url=t[4:])))
                elif t.startswith("file:"):
                    mime, _ = mimetypes.guess_type(t[5:])
                    contents.append(MessageContent(type="file", file=FileObject(file_id=t[5:], mime_type=mime or "application/octet-stream")))
                else:
                    contents.append(MessageContent(type="text", text=t))
            return contents

        async def run(self):
            self._save_session()
            console.print(Panel(f"🔥 [bold green]Agent 已就绪[/bold green]\nThread: {self.thread_id[:8]}...", border_style="green"))
            
            while True:
                try:
                    user_input = await self.session.prompt_async(HTML("\n👤 <b><ansiyellow>You > </ansiyellow></b>"))
                    if not user_input: continue
                    if user_input.lower() in ["exit", "q", "/quit"]: 
                        break
                    
                    if user_input.startswith("/reset"):
                        self.state["messages"] = []
                        console.print("[dim]上下文已清空[/dim]")
                        continue

                    processed, _ = process_multimodal_content(self._smart_parse(user_input))
                    self.state["messages"].append({"role": "user", "content": processed})
                    
                    with Live(Spinner("dots", text="思考中..."), console=console, refresh_per_second=10) as live:
                        ans = await self.handle_stream(live)
                    
                    self.state["messages"].append({"role": "assistant", "content": ans})
                    self._save_session()
                    
                except (EOFError, KeyboardInterrupt): 
                    break
                except Exception as e:
                    traceback.print_stack()
                    console.print(f"[red]发生错误: {e}[/red]")

    return DeepAgentCLI

# --- 2. 热重载监控逻辑 ---
class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.last_restart = 0
        self.active = True  # True 表示可以重启
        self.start_agent()  # 首次启动

    def start_agent(self):
        if not self.active:
            return  # 禁止重启
        now = time.time()
        if now - self.last_restart < RESTART_DEBOUNCE_SECONDS:
            return
        self.last_restart = now

        if self.process and self.process.poll() is None:
            print("\n⏹️  正在停止旧进程...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("⚠️  强制杀死旧进程...")
                self.process.kill()
                self.process.wait()

        print(f"\n🔄 [{time.strftime('%H:%M:%S')}] 启动服务...")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "agi.cli", "--child"],
            env=os.environ,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

    def on_modified(self, event):
        if event.is_directory or not self.active:
            return
        
        src_path = pathlib.Path(event.src_path)

        # 忽略目录
        if any(ignore in src_path.parts for ignore in IGNORED_DIRS):
            return
        
        # 忽略 session 文件
        if src_path.name == STATE_CACHE:
            return

        # 扩展名白名单
        if src_path.suffix.lower() in WATCH_EXTENSIONS:
            # 只有 active=True 才允许启动
            self.start_agent()

    def monitor_process_exit(self):
        """主循环中调用，监控子进程退出，不再自动重启"""
        if self.process and self.process.poll() is not None and self.active:
            code = self.process.returncode
            print(f"\nℹ️  子进程退出 (代码: {code})，不会自动重启。")
            self.active = False  # 禁止再次重启

# --- 3. 入口控制 ---
if __name__ == "__main__":
    if "__file__" in globals():
        SCRIPT_PATH = os.path.abspath(__file__)
    else:
        SCRIPT_PATH = os.path.abspath(sys.argv[0])

    if "--child" in sys.argv:
        try:
            AgentClass = get_agent_cls()
            asyncio.run(AgentClass().run())
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"\n💥 Agent 崩溃: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print(f"👀 监控已启动: {os.getcwd()}")
        print(f"   监听文件类型: {', '.join(WATCH_EXTENSIONS)}")
        print("   按 Ctrl+C 停止所有服务\n")
        
        handler = ReloadHandler(SCRIPT_PATH)
        observer = Observer()
        observer.schedule(handler, path=".", recursive=True)
        observer.start()

        def signal_handler(sig, frame):
            print("\n🛑 接收到停止信号，正在关闭...")
            observer.stop()
            if handler.process and handler.process.poll() is None:
                handler.process.terminate()
                try:
                    handler.process.wait(timeout=3)
                except:
                    handler.process.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while True:
                time.sleep(1)
                handler.monitor_process_exit()
                if not handler.active:
                    print("🛑 主进程检测到子进程已退出，正在关闭监控...")
                    break
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
            # 如果子进程还存在，安全终止
            if handler.process and handler.process.poll() is None:
                handler.process.terminate()
                try:
                    handler.process.wait(timeout=3)
                except:
                    handler.process.kill()
            sys.exit(0)