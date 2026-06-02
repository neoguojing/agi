from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
Protocol:

1. for messages mode:
AIMessage response

{'type': 'messages', 'ns': (), 'data': (AIMessageChunk(...), {'langgraph_node': 'model', ...})}

ToolMessage response

{'type': 'messages', 'ns': (), 'data': (ToolMessage(...), {'langgraph_node': 'tools', ...})}

2. for updates mode:

AIMessage response:
{'type': 'updates', 'ns': (), 'data': {'model': {'messages': [AIMessage(...)]}}}

ToolMessage response:
{'type': 'updates', 'ns': (), 'data': {'tools': {'messages': [ToolMessage(...)]}}}

Middleware state:
{'type': 'updates', 'ns': (), 'data': {'ContextEngineeringMiddleware.after_model': None}}

3. for langgraph sdk

StreamPart(event='messages/partial', data=[{'content': [{'type': 'text', 'text': '您好'}], ...}], id=None)
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
    event_type: str
    payload: Any
    stats: StreamStats
    preview: Optional[str] = None
    latest_ai_message: Optional[str] = None
    full_response: str = ""
    updates_trace_snippet: Optional[str] = None


class StreamProcessor:
    """Normalize LangGraph/deepagents stream chunks into display-friendly state.

    展示目标：
    1. 最终输出是核心，永远单独放在顶部并持续更新；
    2. 中间过程作为辅助信息，展示工具调用、工具返回和状态更新；
    3. 不再把 latest_ai_message 作为独立展示区，避免和最终输出重复。

    `latest_ai_message` 仍保留为兼容字段，供旧调用方读取最近/最终的 AI 文本；
    UI 展示统一使用 `final_response` / `full_response`。
    """

    def __init__(self):
        self.stats = StreamStats()
        self.start_time = time.time()
        self.updates_trace: List[str] = []
        self.max_updates_trace = 30
        self.latest_ai_message = ""
        self.final_response = ""
        # full_response 是之前代码暴露给 StreamEvent 的字段，这里保持为最终输出别名。
        self.full_response = ""
        self.updates_trace_snippet: Optional[str] = None
        self._seen_update_items: set[Tuple[str, str, str]] = set()
        self._stream_delta_buffer = ""

    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        """Convert LangChain message objects, SDK dict messages, and chunks to dicts.

        LangGraph 本地流通常给出 `AIMessageChunk` / `ToolMessage` 对象；SDK
        `messages/partial` 往往直接给 dict。后续解析逻辑只处理 dict，因此入口处
        先把常见属性扁平化。这里不深拷贝 dict，以保留 SDK 原始字段。
        """
        if isinstance(msg, dict):
            return msg
        return {
            "type": type(msg).__name__,
            "content": getattr(msg, "content", ""),
            "response_metadata": getattr(msg, "response_metadata", {}) or {},
            "usage_metadata": getattr(msg, "usage_metadata", {}) or {},
            "additional_kwargs": getattr(msg, "additional_kwargs", {}) or {},
            "name": getattr(msg, "name", None),
            "id": getattr(msg, "id", None),
            "tool_calls": getattr(msg, "tool_calls", None),
            "invalid_tool_calls": getattr(msg, "invalid_tool_calls", None),
            "tool_call_chunks": getattr(msg, "tool_call_chunks", None),
            "tool_call_id": getattr(msg, "tool_call_id", None),
        }

    def _normalize_part(self, part: Any) -> Optional[Dict[str, Any]]:
        """Normalize all supported stream envelopes to `{type, data, raw_event}`.

        当前需要兼容三类输入：
        - deepagents/LangGraph 本地 dict：`{"type": "messages", "data": ...}`；
        - LangGraph SDK dict：`{"event": "messages/partial", "data": ...}`；
        - LangGraph SDK `StreamPart` 对象：有 `.event` / `.data` 属性。

        归一化后，`process_part` 只需根据 `type` 分派；`raw_event` 用来判断
        `messages/partial` 这类增量事件，避免把增量输出误当成完整 AIMessage。
        """
        if isinstance(part, dict):
            if "type" in part:
                normalized = dict(part)
                normalized.setdefault("raw_event", part.get("event") or part.get("type"))
                return normalized
            if "event" in part:
                return {
                    "type": self._event_to_type(part.get("event")),
                    "data": part.get("data"),
                    "raw_event": part.get("event"),
                }
            return None

        event = getattr(part, "event", None)
        data = getattr(part, "data", None)
        if event is None:
            return None
        return {
            "type": self._event_to_type(event),
            "data": data,
            "id": getattr(part, "id", None),
            "raw_event": event,
        }

    def _event_to_type(self, event: Any) -> str:
        """Map SDK event names to the coarse stream modes handled by this class."""
        name = str(event or "")
        if name.startswith("messages"):
            return "messages"
        if name.startswith("updates") or name == "values":
            return "updates"
        if name.startswith("tool"):
            return "tool_calls"
        return name

    def _message_type_name(self, msg_data: Dict[str, Any]) -> str:
        """Return a stable human-readable message type name."""
        raw_type = str(msg_data.get("type") or "Message")
        if "|" in raw_type:
            raw_type = raw_type.split("|")[0].strip()
        mapping = {
            "ai": "AIMessage",
            "human": "HumanMessage",
            "tool": "ToolMessage",
            "system": "SystemMessage",
        }
        return mapping.get(raw_type.lower(), raw_type)

    def _is_ai_message(self, msg_type: str) -> bool:
        """Whether the normalized message type should be treated as AI output."""
        return "ai" in msg_type.lower()

    def _extract_text_from_content_blocks(self, content: Any) -> str:
        """Extract displayable text from LangChain/OpenAI-style content blocks.

        Text blocks are concatenated as the answer body. Non-text multimodal blocks are
        represented with compact placeholders so the stream UI remains readable instead
        of dumping large nested payloads.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    block_type = item.get("type")
                    if block_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif block_type in {"image_url", "image", "file"}:
                        parts.append(f"[{block_type}]")
                    else:
                        parts.append(str(item))
                elif item is not None:
                    parts.append(str(item))
            return "".join(x for x in parts if x)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return str(content.get("text", ""))
            return str(content)
        return str(content or "").strip()

    def _iter_tool_calls(self, msg_data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Yield tool/function calls from modern and legacy message fields."""
        tool_calls = msg_data.get("tool_calls") or msg_data.get("tool_call_chunks") or []
        if isinstance(tool_calls, dict):
            yield tool_calls
        elif isinstance(tool_calls, (list, tuple)):
            for call in tool_calls:
                if isinstance(call, dict):
                    yield call

        function_call = (msg_data.get("additional_kwargs") or {}).get("function_call")
        if isinstance(function_call, dict):
            yield {"name": function_call.get("name"), "args": function_call.get("arguments")}

    def _tool_calls_summary(self, msg_data: Dict[str, Any]) -> str:
        """Build concise Markdown rows for tool calls attached to an AI message."""
        summaries = []
        for call in self._iter_tool_calls(msg_data):
            name = call.get("name") or call.get("function", {}).get("name") or "tool"
            args = call.get("args") or call.get("arguments") or ""
            args_text = str(args)
            if len(args_text) > 220:
                args_text = args_text[:220] + "..."
            summary = f"- 🛠️ `{name}`"
            if args_text:
                summary += f" — `{args_text}`"
            summaries.append(summary)
        return "\n".join(summaries)

    def _update_stats_from_message(self, msg_data: Dict[str, Any]):
        """Refresh model/token/TPS stats from a message if usage metadata exists.

        Usage metadata may arrive only on the final message. To avoid stats flickering
        backwards when chunks omit data, token values are kept as the maximum observed
        values in the current stream.
        """
        metadata = msg_data.get("response_metadata") or {}
        usage = msg_data.get("usage_metadata") or {}
        additional_kwargs = msg_data.get("additional_kwargs") or {}

        model_name = (
            metadata.get("model_name")
            or metadata.get("model")
            or additional_kwargs.get("model_name")
        )
        if model_name:
            self.stats.model = str(model_name)

        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        total_t = usage.get("total_tokens")

        if isinstance(in_t, int):
            self.stats.input_tokens = max(self.stats.input_tokens, in_t)
        if isinstance(out_t, int):
            self.stats.output_tokens = max(self.stats.output_tokens, out_t)
        if isinstance(total_t, int):
            self.stats.total_tokens = max(self.stats.total_tokens, total_t)
        else:
            self.stats.total_tokens = max(
                self.stats.total_tokens,
                self.stats.input_tokens + self.stats.output_tokens,
            )

        elapsed = max(time.time() - self.start_time, 1e-6)
        self.stats.tps = self.stats.output_tokens / elapsed if self.stats.output_tokens else 0.0
        self.stats.tokens = (
            f"In: {self.stats.input_tokens} | "
            f"Out: {self.stats.output_tokens} | "
            f"Total: {self.stats.total_tokens}"
        )

    def _parse_messages_event(self, part: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract `(message, metadata)` from a normalized `messages` event.

        Supported shapes:
        - local stream: `data=(message, metadata)`;
        - SDK partial: `data=[message]`;
        - defensive dict wrappers: `data={"message": ..., "metadata": ...}`.
        """
        data = part.get("data")
        if isinstance(data, (tuple, list)) and data:
            # deepagents messages mode uses (message, metadata); SDK messages/partial uses [message].
            msg = data[0]
            meta = (
                data[1]
                if len(data) > 1
                and isinstance(data[1], dict)
                and not self._looks_like_message_dict(data[1])
                else {}
            )
            return self._message_to_dict(msg), meta
        if isinstance(data, dict):
            msg = data.get("message") or data.get("chunk") or data.get("data")
            meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            if msg is not None:
                return self._message_to_dict(msg), meta
            if self._looks_like_message_dict(data):
                return self._message_to_dict(data), meta
        return {}, {}

    def _looks_like_message_dict(self, value: Dict[str, Any]) -> bool:
        """Heuristic that separates message dicts from metadata dicts."""
        return any(
            key in value
            for key in ("content", "tool_calls", "usage_metadata", "response_metadata")
        )

    def _append_update_trace(self, node_name: str, msg_type: str, text: str):
        """Append one deduplicated intermediate-process row.

        `updates` may replay the whole state on each event, so node/type/text are used as
        a stable de-dup key. The displayed trace keeps only the latest N rows to prevent
        long tool loops from making the Live panel unwieldy.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return
        key = (node_name, msg_type, clean_text[:500])
        if key in self._seen_update_items:
            return
        self._seen_update_items.add(key)
        self.updates_trace.append(self._format_trace_item(node_name, msg_type, clean_text))
        self.updates_trace = self.updates_trace[-self.max_updates_trace :]
        self.updates_trace_snippet = "\n".join(self.updates_trace)

    def _format_trace_item(self, node_name: str, msg_type: str, text: str) -> str:
        """Format a single intermediate step as compact Markdown."""
        label = self._display_message_label(msg_type)
        display_text = text.strip()
        if len(display_text) > 700:
            display_text = display_text[:700].rstrip() + "..."
        indented_text = "\n  ".join(display_text.splitlines())
        return f"- **{label}** · `{node_name}`\n\n  {indented_text}"

    def _display_message_label(self, msg_type: str) -> str:
        """Choose friendly labels/icons for message types in the process section."""
        lower = msg_type.lower()
        if "tool" in lower:
            return "🛠️ Tool"
        if "ai" in lower:
            return "🤖 Model"
        if "state" in lower:
            return "🔄 State"
        return f"📌 {msg_type}"

    def _is_stream_delta(self, msg_data: Dict[str, Any], raw_event: Any = None) -> bool:
        """Decide whether an AI message should be appended or replace the final output.

        Chunk events (`AIMessageChunk`, `messages/partial`) carry only the newly generated
        text and must be appended. Full `AIMessage` objects from `updates` usually contain
        the complete answer and should replace the current final output to avoid duplicate
        text when both `messages` and `updates` modes are enabled.
        """
        msg_type = self._message_type_name(msg_data)
        event_name = str(raw_event or "")
        return "Chunk" in msg_type or event_name.endswith("/partial")

    def _set_final_response(self, text: str, *, append: bool):
        """Update the primary final-answer buffer and compatibility aliases."""
        clean_text = text.replace("→", "->")
        if append:
            self._stream_delta_buffer += clean_text
            self.final_response = self._stream_delta_buffer
        else:
            self.final_response = clean_text
            self._stream_delta_buffer = clean_text
        self.full_response = self.final_response
        self.latest_ai_message = self.final_response

    def _handle_message(
        self,
        msg_data: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        node_name: Optional[str] = None,
        raw_event: Any = None,
    ) -> str:
        """Process one message and update stats/final-output state.

        Return value is a short textual preview for callers that want to know whether the
        Live panel should refresh. AI messages update the final answer buffer; tool calls
        and tool messages return text so `updates` can place them in the intermediate
        process section. The final answer itself is *not* separately shown as
        `latest_ai_message` because that duplicates the core output area.
        """
        meta = meta or {}
        stream_node = (
            node_name
            or meta.get("langgraph_node")
            or meta.get("lc_agent_name")
            or msg_data.get("name")
        )
        if stream_node:
            self.stats.node = str(stream_node)

        msg_type = self._message_type_name(msg_data)
        content = self._extract_text_from_content_blocks(msg_data.get("content", ""))
        tool_summary = self._tool_calls_summary(msg_data)

        preview = ""
        if content:
            preview = content.replace("→", "->")
            if self._is_ai_message(msg_type):
                self._set_final_response(preview, append=self._is_stream_delta(msg_data, raw_event))
        elif tool_summary:
            preview = tool_summary

        if tool_summary and content:
            preview = f"{preview}\n{tool_summary}"

        self._update_stats_from_message(msg_data)
        return preview

    def get_presentation_body(self) -> str:
        """Render the Live panel body as Markdown.

        The final answer appears first because it is the user's main objective. The
        intermediate-process section follows as context. `latest_ai_message` is not
        rendered as a separate block; it is only an internal/backwards-compatible alias of
        the same final answer text.
        """
        final_output = self.final_response.strip()
        process_output = (self.updates_trace_snippet or "").strip()
        sections: List[str] = ["## ✨ Final Output"]
        sections.append(final_output if final_output else "_Waiting for the model's final response..._")

        if process_output:
            sections.append("\n---\n\n## 🔎 Intermediate Process\n" + process_output)
        else:
            sections.append("\n---\n\n## 🔎 Intermediate Process\n_No intermediate events yet._")
        return "\n\n".join(sections).strip()

    def get_subtitle(self, elapsed: float) -> str:
        """Build a compact status line for the Rich panel subtitle."""
        self.stats.tps = (
            self.stats.output_tokens / max(elapsed, 1e-6)
            if self.stats.output_tokens
            else self.stats.tps
        )
        return (
            f"[bold cyan]{elapsed:.1f}s[/bold cyan] | {self.stats.model} | "
            f"Node: [yellow]{self.stats.node}[/yellow] | {self.stats.tokens} | "
            f"[magenta]{self.stats.tps:.1f} t/s[/magenta]"
        )

    def process_part(self, part: Any) -> Optional[StreamEvent]:
        """Consume one raw stream part and return a display event when state changed.

        This is the public entry point used by `console.py`:
        1. normalize the raw envelope;
        2. dispatch by coarse event type (`messages`, `updates`, `tool_calls`);
        3. update final-answer / intermediate-process / stats state;
        4. return a `StreamEvent` snapshot for optional downstream use.
        """
        normalized = self._normalize_part(part)
        if not normalized:
            return None

        event_type = normalized.get("type")
        raw_event = normalized.get("raw_event")
        if event_type == "messages":
            msg_data, event_meta = self._parse_messages_event(normalized)
            if not msg_data:
                return None
            preview = self._handle_message(msg_data, event_meta, raw_event=raw_event)
            return StreamEvent(
                event_type="messages",
                payload=part,
                stats=self.stats,
                preview=preview[:1000] if preview else None,
                latest_ai_message=self.latest_ai_message,
                full_response=self.full_response,
                updates_trace_snippet=self.updates_trace_snippet,
            )

        if event_type == "updates":
            updates = normalized.get("data", {})
            if isinstance(updates, dict):
                for node_name, node_payload in updates.items():
                    self.stats.node = str(node_name)
                    if isinstance(node_payload, dict) and "messages" in node_payload:
                        msgs = node_payload.get("messages", [])
                        if isinstance(msgs, (list, tuple)):
                            for raw_msg in msgs:
                                msg_data = self._message_to_dict(raw_msg)
                                msg_type = self._message_type_name(msg_data)
                                content = self._handle_message(
                                    msg_data,
                                    node_name=str(node_name),
                                    raw_event=raw_event,
                                )
                                if content and not self._is_ai_message(msg_type):
                                    self._append_update_trace(str(node_name), msg_type, content[:700])
                                elif content and self._tool_calls_summary(msg_data):
                                    self._append_update_trace(
                                        str(node_name),
                                        "Tool Call",
                                        self._tool_calls_summary(msg_data),
                                    )
                    elif node_payload is not None:
                        self._append_update_trace(str(node_name), "State", str(node_payload)[:700])
            return StreamEvent(
                event_type="updates",
                payload=part,
                stats=self.stats,
                latest_ai_message=self.latest_ai_message,
                full_response=self.full_response,
                updates_trace_snippet=self.updates_trace_snippet,
            )

        if event_type == "tool_calls":
            data = normalized.get("data") or {}
            tool_call = data.get("tool_call") if isinstance(data, dict) else None
            if tool_call:
                name = tool_call.get("name", "N/A")
                self._append_update_trace("tools", "Tool Call", f"`{name}`")
            return StreamEvent(
                event_type="tool_calls",
                payload=part,
                stats=self.stats,
                full_response=self.full_response,
                updates_trace_snippet=self.updates_trace_snippet,
            )

        return None
