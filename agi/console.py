# filename: agent_cli.py
import asyncio
import uuid
import os
import sys
import time
import json
import mimetypes
import traceback
from typing import List, Any
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from prompt_toolkit.formatted_text import HTML 

# 假设这些是你的内部模块，请确保它们在 PYTHONPATH 中
from agi.agent.agent import stream_agent_async
from agi.agent.context import Context
from agi.apps.common import MessageContent, ImageURL, FileObject
from agi.api.media import process_multimodal_content
# --- 配置 ---
STATE_CACHE = ".cli_session.json"



console = Console()

class DeepAgentCLI:
    def __init__(self):
        self.load_session()
        self.session = PromptSession(completer=PathCompleter(expanduser=True))

    def load_session(self):
        if os.path.exists(STATE_CACHE):
            try:
                with open(STATE_CACHE, 'r', encoding='utf-8') as f:
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
                "conversation_id": self.conversation_id
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, STATE_CACHE)
        except Exception:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

    async def handle_stream(self, live):
        full_response = ""
        config = {"configurable": {"thread_id": self.thread_id}}
        context = Context(user_id=self.user_id, conversation_id=self.conversation_id)
        
        last_update_time = 0
        update_interval = 0.05 

        async for part in stream_agent_async(self.state, config=config, context=context, stream_mode=["messages"]):
            if isinstance(part, dict) and part.get("type") == "messages":
                data = part.get("data")
                if data and len(data) > 0:
                    content = getattr(data[0], "content", "")
                    full_response += str(content)
                    
                    now = time.time()
                    if now - last_update_time > update_interval:
                        live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
                        last_update_time = now
        
        if full_response:
            live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
        return full_response

    def _smart_parse(self, text: str):
        tokens = text.split()
        contents = []
        
        text_buffer = []

        def flush_text():
            """把缓存的文本一次性输出"""
            if text_buffer:
                contents.append(
                    MessageContent(type="text", text=" ".join(text_buffer))
                )
                text_buffer.clear()

        for t in tokens:
            # ---------- IMAGE ----------
            if t.startswith("img:"):
                flush_text()
                contents.append(
                    MessageContent(
                        type="image_url",
                        image_url=ImageURL(url=t[4:])
                    )
                )

            # ---------- FILE ----------
            elif t.startswith("file:"):
                flush_text()
                path = t[5:]
                mime, _ = mimetypes.guess_type(path)

                contents.append(
                    MessageContent(
                        type="file",
                        file=FileObject(
                            file_id=path,
                            mime_type=mime or "application/octet-stream"
                        )
                    )
                )

            # ---------- TEXT ----------
            else:
                text_buffer.append(t)

        # 收尾
        flush_text()

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

                human_message = process_multimodal_content(self._smart_parse(user_input))
                self.state["messages"].append(human_message)
                
                with Live(Spinner("dots", text="思考中..."), console=console, refresh_per_second=10) as live:
                    ans = await self.handle_stream(live)
                
                self.state["messages"].append({"role": "assistant", "content": ans})
                self._save_session()
                
            except (EOFError, KeyboardInterrupt): 
                break
            except Exception as e:
                traceback.print_exc()
                console.print(f"[red]发生错误: {e}[/red]")


if __name__ == "__main__":
    try:
        AgentClass = DeepAgentCLI()
        asyncio.run(AgentClass.run())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Agent 崩溃: {e}")
        traceback.print_exc()
        sys.exit(1)