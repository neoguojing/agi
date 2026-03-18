import asyncio
import uuid
import os
import mimetypes
from typing import List, Dict, Any, Union
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from agi.agent.agent import stream_agent
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
        
        config = {"configurable": {"thread_id": self.thread_id}}
        context = Context(user_id=self.user_id, conversation_id=self.conversation_id)

        try:
            async for part in stream_agent(self.state, config=config, context=context):
                # 1. 解析节点更新 (Updates 模式通常能更早抓到工具调用)
                if isinstance(part, dict):
                    # 查找当前活跃的消息数据
                    node_data = part.get("agent") or part.get("chatbot") or part.get("tools")
                    
                    if node_data and "messages" in node_data:
                        last_msg = node_data["messages"][-1]
                        
                        # --- 处理工具调用阶段 ---
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                current_tool = tc['name']
                                # 在界面上显示正在调用的工具
                                live.update(Panel(
                                    Spinner("dots", text=f"正在调用工具: [bold cyan]{current_tool}[/bold cyan]..."),
                                    title="Agent Action", border_style="yellow"
                                ))
                        
                        # --- 处理工具返回结果阶段 ---
                        elif last_msg.role == "tool":
                            # 工具执行完毕，展示一下状态
                            live.update(Panel(
                                f"✅ 工具 [bold cyan]{current_tool}[/bold cyan] 执行完毕，正在整理结果...",
                                title="Agent Action", border_style="green"
                            ))
                            await asyncio.sleep(0.5) # 给用户一点感官停留时间

                        # --- 处理最终文本回复阶段 ---
                        elif last_msg.role == "assistant" and last_msg.content:
                            full_response = last_msg.content
                            live.update(Panel(Markdown(full_response), title="Agent Response", border_style="blue"))
                
            return full_response
        except Exception as e:
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
                with Live(Spinner("pulse", text="Agent 准备中..."), console=console, refresh_per_second=10) as live:
                    final_text = await self.handle_stream(live)

                self.state["messages"].append({"role": "assistant", "content": final_text})

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

if __name__ == "__main__":
    asyncio.run(DeepAgentCLI().main_loop())