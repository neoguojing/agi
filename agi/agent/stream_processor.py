from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

"""
Protocal:

1. for messages mode:
AIMessage response

{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(content=[], additional_kwargs={}, response_metadata={}, 
id='lc_run--019e814f-66eb-76d2-9616-b3d03216e20c', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], chunk_position='last'), {'ls_integration': 
'langchain_chat_model', 'lc_agent_name': 'main', 'versions': {'deepagents': '0.6.3'}, 'thread_id': '411bb209-a276-47cb-9e65-431516cacf9c', 'langgraph_step': 
824, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 
'model:0fa16542-1048-39a9-0cc2-c4dc7cc41262', 'checkpoint_ns': 'model:0fa16542-1048-39a9-0cc2-c4dc7cc41262', 'ls_provider': 'google_genai', 'ls_model_name': 
'gemini-3.1-flash-lite', 'ls_model_type': 'chat', 'ls_temperature': 1.0})}

ToolMessage response

{'type': 'messages', 'ns': (), 'data': (ToolMessage(content="Tool call failed in middleware 'DEFAULT' for 'check_async_task': KeyError: 
'stock-analyse-asubagent'", name='check_async_task', id='24a00531-bb8f-4455-88ea-52d25ef0ed4c', tool_call_id='64e0d2be-5cdb-4fff-b41b-1c11d4772337'), 
{'ls_integration': 'deepagents', 'lc_agent_name': 'main', 'versions': {'deepagents': '0.6.3'}, 'thread_id': '411bb209-a276-47cb-9e65-431516cacf9c', 
'langgraph_step': 827, 'langgraph_node': 'tools', 'langgraph_triggers': ('__pregel_push',), 'langgraph_path': ('__pregel_push', 0, False), 
'langgraph_checkpoint_ns': 'tools:c5fe63c6-eb73-7d49-d252-24e37d853e1a'})}


2.for updates mode：

AIMessage response：
{'type': 'updates', 'ns': (), 'data': {'model': {'messages': [AIMessage(content=[], additional_kwargs={'function_call': {'name': 'task', 
'arguments': '{"subagent_type": "web-search-expert", "description": "Retrieve the current stock prices for NVDA and TSLA. Please provide the output as a 
clean text string with the price for each ticker. If multiple sources exist, pick the most reliable recent market close or real-time price."}'}, 
'__gemini_function_call_thought_signatures__': {'5aad0492-3d92-4923-94d4-69a95934c3f6': 
'EjQKMgEMOdbHQLew3Q0gncQqHxD0+f9SIOP7xmoFjje9nqYg+ZQZ+p1mAWWCHv90RAg4Dh9x'}}, response_metadata={'finish_reason': 'STOP', 'model_name': 
'gemini-3.1-flash-lite', 'safety_ratings': [], 'model_provider': 'google_genai'}, name='main', id='lc_run--019e864f-874d-7f73-b0f1-a137e683f191-0', 
tool_calls=[{'name': 'task', 'args': {'subagent_type': 'web-search-expert', 'description': 'Retrieve the current stock prices for NVDA and TSLA. Please 
provide the output as a clean text string with the price for each ticker. If multiple sources exist, pick the most reliable recent market close or real-time 
price.'}, 'id': '5aad0492-3d92-4923-94d4-69a95934c3f6', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 12135, 'output_tokens':
72, 'total_tokens': 12207, 'input_token_details': {'cache_read': 4052}})]}}}

ToolMesage response:

{'type': 'updates', 'ns': (), 'data': {'tools': {'todos': [{'content': '将经过测试的策略集成至 FastAPI BackgroundTasks', 'status': 'pending'}], 
'messages': [ToolMessage(content="Updated todo list to [{'content': '在系统内查找与InvestmentMonitor相关的类或逻辑（项目范围搜索）', 'status': 'completed'}, 
{'content': '重新评估NVDA和TSLA在内的投资组合风险 profile (June 1)', 'status': 'completed'}, {'content': '定义重平衡策略 (sell/hedge/hold) based on 
analysis', 'status': 'in_progress'}, {'content': '将经过测试的策略集成至 FastAPI BackgroundTasks', 'status': 'pending'}]", name='write_todos', 
id='8d5538c6-b457-4209-a393-3b7d90fd5511', tool_call_id='2760e797-57dd-4615-a0da-393222926c47')]}}}

MiddleWare state:

{'type': 'updates', 'ns': (), 'data': {'ContextEngineeringMiddleware.after_model': None}}
{'type': 'updates', 'ns': (), 'data': {'TodoListMiddleware.after_model': None}}

3. for langgraph sdk

StreamPart(event='messages/partial', data=[{'content': [{'type': 'text', 'text': 
'您好', 'index': 0, 'extras': {'signature': 
'EjQKMgEMOdbHIog8QLuMANWBoql8fBF0vxk740NWbk0eFuCv+ktO+8/i/EDA0BRug9uj40AB'}}], 'additional_kwargs': {}, 'response_metadata': {'safety_ratings': [], 
'model_provider': 'google_genai', 'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite'}, 'type': 'ai', 'name': None, 'id': 
'lc_run--019e86f2-9afc-7610-9d31-07c1d4ca9918', 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 11565, 'output_tokens': 180, 
'total_tokens': 11745, 'input_token_details': {'cache_read': 0}}}], id=None)

"""

@dataclass
class StreamStats:
    model: str = "N/A"
    node: str = "N/A"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tps: float = 0.0
    tokens: str = "In: 0 | Out: 0 | Total: 0"

@dataclass
class StreamEvent:
    event_type: str  # "messages" or "updates"
    payload: Any
    stats: StreamStats
    preview: Optional[str] = None
    latest_ai_message: Optional[str] = None
    full_response: str = ""
    updates_trace_snippet: Optional[str] = None

class StreamProcessor:
    def __init__(self):
        self.stats = StreamStats()
        self.seen_messages: Set[Tuple[str, str]] = set()
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.updates_trace: List[str] = []
        self.max_updates_trace = 30
        self.latest_ai_message = ""
        self.full_response = ""
        self.updates_trace_snippet: Optional[str] = None

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

    def _is_ai_message(self, msg_type: str) -> bool:
        return "ai" in msg_type.lower() or "AIMessage" in msg_type

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

    def _update_stats_from_message(self, msg_data: Dict[str, Any]):
        metadata = msg_data.get("response_metadata") or {}
        usage = msg_data.get("usage_metadata") or {}
        additional_kwargs = msg_data.get("additional_kwargs", {})
        usage = usage or additional_kwargs

        model_name = metadata.get("model_name") or metadata or additional_kwargs.get("model_name")
        # wait, I'll just use the real logic
        model_name = metadata.get("model_name") or metadata.get("model") or additional_kwargs.get("model_name")
        if model_name:
            self.stats.model = str(model_name)

        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        total_t = usage.get("total_tokens")

        if isinstance(in_t, int):
            self.stats.input_tokens = max(self.stats.input_tokens, in_t)
        if isinstance(out_t, int):
            self.stats.output_tokens += max(out_t, 0)
        if isinstance(total_t, int):
            self.stats.total_tokens = max(
                self.stats.total_tokens,
                total_t,
                self.stats.input_tokens + self.stats.output_tokens,
            )
        else:
            self.stats.total_tokens = max(
                self.stats.total_tokens,
                self.stats.input_tokens + self.stats.output_tokens,
            )

        self.stats.tokens = (
            f"In: {self.stats.input_tokens} | "
            f"Out: {self.stats.output_tokens} | "
            f"Total: {self.stats.total_tokens}"
        )

    def _parse_messages_event(self, part: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data = part.get("data")
        if isinstance(data, (tuple, list)) and data:
            msg = data[0]
            meta = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
            return self._message_to_dict(msg), meta
        if isinstance(data, dict):
            msg = data.get("message") or data.get("chunk") or data.get("data")
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            if msg is not None:
                return self._message_to_dict(msg), meta
        return {}, {}

    def get_presentation_body(self) -> str:
        sections: List[str] = []
        if self.updates_trace_snippet:
            sections.append(self.updates_trace_snippet)

        if self.latest_ai_message:
            sections.append("\n---\n\n## Latest AIMessage\n" + self.latest_ai_message)
        elif self.full_response:
            sections.append("\n---\n\n## Streamed Response\n" + self.full_response)

        body = "\n\n".join(sections).strip()
        return body if body else "(waiting for updates...)"

    def get_subtitle(self, elapsed: float) -> str:
        return (
            f"[bold cyan]{elapsed:.1f}s[/bold cyan] | {self.stats.model} | "
            f"Node: [yellow]{self.stats.node}[/yellow] | {self.stats.tokens} | "
            f"[magenta]{self.stats.tps:.1f} t/s[/magenta]"
        )

    def process_part(self, part: Any) -> Optional[StreamEvent]:
        if not isinstance(part, dict):
            return None

        event_type = part.get("type")
        if event_type == "messages":
            msg_data, event_meta = self._parse_messages_event(part)
            if not msg_data:
                return None

            content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
            msg_type = self._message_type_name(msg_data)

            stream_node = event_meta.get("langgraph_node") or event_meta.get("lc_agent_name")
            if stream_node:
                self.stats.node = str(stream_node)

            if content:
                content_val = str(content).replace("→", "->")
                self.full_response += content_val
                if self._is_ai_message(msg_type):
                    self.latest_ai_message = (self.latest_ai_message + content_val) if "Chunk" in msg_type else content_val

            self._update_stats_from_message(msg_data)

            return StreamEvent(
                event_type="messages",
                payload=part,
                stats=self.stats,
                preview=self._extract_text_from_content_blocks(msg_data.get("content", ""))[:1000],
                latest_ai_message=self.latest_ai_message,
                full_response=self.full_response
            )

        elif event_type == "updates":
            updates = part.get("data", {})
            if isinstance(updates, dict):
                for node_name, node_payload in updates.items():
                    if isinstance(node_payload, dict) and "messages" in node_payload:
                        self.stats.node = str(node_name)
                        msgs = node_payload.get("messages", [])
                        if isinstance(msgs, (list, tuple)):
                            for raw_msg in msgs:
                                msg_data = self._message_to_dict(raw_msg)
                                msg_type = self._message_type_name(msg_data)
                                content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
                                if content:
                                    snippet = f"### Node `{node_name}`\n- **{msg_type}**: {content[:400]}"
                                    self.updates_trace_snippet = snippet
                                    if self._is_ai_message(msg_type):
                                        self.latest_ai_message = content
                                self._update_stats_from_message(msg_data)

            return StreamEvent(
                event_type="updates",
                payload=part,
                stats=self.stats,
                full_response=self.full_response,
                updates_trace_snippet=getattr(self, 'updates_trace_snippet', None)
            )

        elif event_type == "tool_calls":
            tool_call = part.get("data", {}).get("tool_call")
            if tool_call:
                self.updates_trace_snippet = f"### Tool Call\n- **{tool_call.get('name', 'N/A')}**"
                self.full_response += f"\n[Tool Call: {tool_call.get('name', 'N/A')}]"

            return StreamEvent(
                event_type="tool_calls",
                payload=part,
                stats=self.stats,
                full_response=self.full_response,
                updates_trace_snippet=self.updates_trace_snippet
            )

        return None
