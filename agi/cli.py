import asyncio
from asyncio import sleep
import uuid
import os
import mimetypes
from typing import List, Dict, Any, Union
from prompt_toolkit import PromptSession
from langgraph.types import Overwrite
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from agi.agent.agent import stream_agent_async
from agi.agent.context import Context
from agi.apps.common import MessageContent,ImageURL,FileObject
from agi.api.media import process_multimodal_content
console = Console()

class DeepAgentCLI:
    def __init__(self):
        self.user_id = "admin"
        self.conversation_id = str(uuid.uuid4())
        self.thread_id = str(uuid.uuid4())
        self.state = {"messages": [], "user_id": self.user_id, "conversation_id": self.conversation_id}
        self.session = PromptSession(completer=PathCompleter(expanduser=True))

    async def handle_stream(self, live: Live):
        full_response = ""
        current_tool = None
        is_streaming_text = False
        
        config = {"configurable": {"thread_id": self.thread_id}}
        context = Context(user_id=self.user_id, conversation_id=self.conversation_id)

        try:
            async for part in stream_agent_async(
                self.state, 
                config=config, 
                context=context,
                stream_mode=["messages", "updates"]
            ):
                # --- 情况 A: 处理实时消息流 (打字机效果) ---
                if isinstance(part, dict) and part.get("type") == "messages":
                    data_tuple = part.get("data")
                    if data_tuple and len(data_tuple) >= 1:
                        chunk = data_tuple[0]
                        content = getattr(chunk, "content", "")
                        if content:
                            full_response += content
                            is_streaming_text = True
                            live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
                
                # --- 情况 B: 处理节点状态更新 (工具调用/完成) ---
                elif isinstance(part, dict) and part.get("type") == "updates":
                    updates_data = part.get("data", {})
                    
                    for node_name, node_output in updates_data.items():
                        if not isinstance(node_output, dict):
                            continue
                            
                        raw_messages = node_output.get("messages")
                        
                        # 🛠️ 关键修复：处理 Overwrite 对象
                        if raw_messages is None:
                            continue
                        
                        # 检查是否是 LangGraph 的 Overwrite 包装器
                        actual_messages = []
                        if hasattr(raw_messages, 'value'):
                            # 这是一个 Overwrite 对象，提取真实列表
                            actual_messages = raw_messages.value
                        elif isinstance(raw_messages, list):
                            # 普通列表
                            actual_messages = raw_messages
                        else:
                            # 未知类型，跳过
                            continue

                        if not actual_messages:
                            continue
                            
                        last_msg = actual_messages[-1]
                        
                        # 1. 检测工具调用
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                tool_name = tc.get('name', 'Unknown')
                                current_tool = tool_name
                                live.update(Panel(
                                    Spinner("dots", text=f"正在调用工具: [bold cyan]{tool_name}[/bold cyan]..."),
                                    title="Agent Action", border_style="yellow"
                                ))
                        
                        # 2. 检测工具返回
                        elif getattr(last_msg, "role", "") == "tool":
                            live.update(Panel(
                                f"✅ 工具 [bold cyan]{current_tool}[/bold cyan] 执行完毕，正在分析结果...",
                                title="Agent Action", border_style="green"
                            ))
                            await asyncio.sleep(0.3)
                        
                        # 3. 检测最终回复 (兜底)
                        elif getattr(last_msg, "role", "") == "assistant":
                            content = getattr(last_msg, "content", "")
                            if content and not is_streaming_text:
                                full_response = content
                                live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))

            return full_response

        except Exception as e:
            import traceback
            traceback.print_exc()
            console.print(f"\n[bold red]Stream Error: {e}[/bold red]")
            return f"错误: {e}"
    
    def _smart_parse(self, text: str) -> List[Any]:
        """解析输入，支持 img: 和 file: 标签"""
        tokens = text.split()
        contents = []
        text_parts = []
        for token in tokens:
            if token.startswith("img:"):
                path = os.path.expanduser(token.replace("img:", ""))
                contents.append(MessageContent(type="image_url", image_url=ImageURL(url=path)))
            elif token.startswith("file:"):
                path = os.path.expanduser(token.replace("file:", ""))
                mime, _ = mimetypes.guess_type(path)
                contents.append(MessageContent(type="file", file=FileObject(file_id=path, mime_type=mime)))
            else:
                text_parts.append(token)
        if text_parts:
            contents.append(MessageContent(type="text", text=" ".join(text_parts)))
        return contents

    async def main_loop(self):
        console.print(Panel.fit(
            "🚀 [bold green]Multi-Modal Tool-Enabled Agent CLI[/bold green]\n"
            "混合输入: [cyan]帮我看看这张图 img:1.jpg 并把结果存入 file:res.txt[/cyan]",
            border_style="green"
        ))

        while True:
            try:
                user_input = await self.session.prompt_async("\n👤 [bold yellow]You > [/bold yellow]")
                if not user_input.strip(): continue
                if user_input.lower() in ["exit", "quit", "q"]: break

                # 转换协议并更新状态
                content_models = self._smart_parse(user_input)
                processed_content, _ = process_multimodal_content(content_models)
                self.state["messages"].append({"role": "user", "content": processed_content})

                # 流式展示进度和结果
                with Live(Spinner("dots", text="Agent 准备中..."), console=console, refresh_per_second=10) as live:
                    final_text = await self.handle_stream(live)

                self.state["messages"].append({"role": "assistant", "content": final_text})

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

if __name__ == "__main__":
    asyncio.run(DeepAgentCLI().main_loop())