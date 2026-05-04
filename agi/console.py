# filename: agent_cli.py
import asyncio
import uuid
import os
import io
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
from langgraph.graph.message import add_messages
# --- 配置 ---
STATE_CACHE = ".cli_session.json"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        
        # --- 计时与状态初始化 ---
        start_time = time.time()  # 记录流开始的绝对时间
        last_update_time = 0
        update_interval = 0.05
        
        stats_info = {"model": "N/A", "tokens": "In: 0 | Out: 0", "node": "N/A", "tps": 0.0}

        async for part in stream_agent_async(self.state, config=config, context=context, stream_mode=["messages"]):
            # 实时计算已经过去的时间
            current_elapsed = time.time() - start_time
            
            # --- 过滤 lc_source 为 summarization 的消息 ---
            if isinstance(part, dict) and part.get("type") == "messages":
                data = part.get("data")
                if data and len(data) > 0:
                    chunk = data[0]
                    
                    # 检查是否是 summarization 消息 (通过 additional_kwargs 或嵌套元数据)
                    is_summarization = False
                    
                    # 检查 chunk 的 additional_kwargs 是否有 lc_source
                    if hasattr(chunk, 'additional_kwargs'):
                        meta = getattr(chunk.additional_kwargs, 'lc_source', None) or \
                                getattr(chunk.additional_kwargs, 'get', lambda x: None)(None)
                        if meta == 'summarization':
                            is_summarization = True
                    
                    # 检查嵌套元数据 (第二个元素可能是元数据字典)
                    if not is_summarization and isinstance(data, tuple) and len(data) > 1:
                        nested_meta = data[1]
                        if isinstance(nested_meta, dict) and nested_meta.get('lc_source') == 'summarization':
                            is_summarization = True
                    
                    # 跳过 summarization 消息
                    if is_summarization:
                        continue
                    
                    # 提取正文并处理特殊符号兼容性
                    content = getattr(chunk, "content", "")
                    full_response += str(content).replace("→", "->")
                    
                    # 提取模型名称
                    metadata = getattr(chunk, "response_metadata", {})
                    if metadata.get("model_name"):
                        stats_info["model"] = metadata.get("model_name")
                    
                    # 提取 Token 统计并计算 TPS (每秒生成 Token 数)
                    usage = getattr(chunk, "usage_metadata", {})
                    if usage:
                        in_t = usage.get("input_tokens", 0)
                        out_t = usage.get("output_tokens", 0)
                        stats_info["tokens"] = f"In: {in_t} | Out: {out_t}"
                        if out_t > 0 and current_elapsed > 0:
                            stats_info["tps"] = out_t / current_elapsed

            # --- 2. 处理节点更新 (Graph Node) ---
            elif isinstance(part, dict) and "langgraph_node" in str(part):
                # 获取当前工作的 LangGraph 节点名
                nodes = [k for k in part.keys() if k not in ("__pregel_pull", "__pregel_push")]
                if nodes:
                    stats_info["node"] = nodes[0]

            # --- 3. 实时刷新 UI 界面 ---
            now = time.time()
            if now - last_update_time > update_interval:
                # 格式化副标题：[时长] 模型 | 节点 | Token 统计 | 速度
                time_display = f"[bold cyan]{current_elapsed:.1f}s[/bold cyan]"
                speed_display = f"[magenta]{stats_info['tps']:.1f} t/s[/magenta]"
                
                subtitle = (
                    f"{time_display} | {stats_info['model']} | "
                    f"Node: [yellow]{stats_info['node']}[/yellow] | "
                    f"{stats_info['tokens']} | {speed_display}"
                )
                
                live.update(
                    Panel(
                        Markdown(full_response), 
                        title="[bold blue]Agent Response[/bold blue]",
                        subtitle=subtitle,
                        subtitle_align="right",
                        border_style="blue",
                        padding=(0, 1)
                    )
                )
                last_update_time = now

        # --- 4. 结束后的最终渲染 (锁定绿色状态) ---
        final_duration = time.time() - start_time
        live.update(
            Panel(
                Markdown(full_response), 
                title="[bold green]Response Finished[/bold green]",
                subtitle=f"[bold white]Total: {final_duration:.2f}s[/bold white] | {stats_info['tokens']} | Avg: {stats_info['tps']:.1f} t/s",
                subtitle_align="right",
                border_style="green"
            )
        )
        
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
                self.state["messages"] = add_messages(self.state["messages"], [human_message])

                console.print(f"[bold blue]Agent:[/bold blue] ", end="")

                with Live(
                    Panel(Spinner("dots", text="思考中..."), title="Agent Response", border_style="blue"),
                    console=console, 
                    refresh_per_second=10,
                    transient=False 
                ) as live:
                    ans = await self.handle_stream(live)
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
