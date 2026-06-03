from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
class EventLogItem:
    """对流信息进行高度定制的结构化关联"""
    category: str       # 'Node' | 'Tool' | 'Final_Output' | 'Stats'
    title: str          # 动态标题：不再是类别名，而是实际的 node 名或工具函数名 (例如: "agent_core", "web_search")
    detail: str         # 详细内容：小字附在后面的实际参数、输出文本或状态
    timestamp: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        """格式化输出：以实际名称为标题，详细内容作为下级缩进附在后面"""
        indented_detail = "\n  ".join(self.detail.strip().splitlines())
        if self.category in ("Tool", "Stats"):
            # 工具执行逻辑或统计明细，用代码块或缩进包裹以体现关联和层次
            return f"- **{self.title}**\n  ```python\n  {indented_detail}\n  ```"
        else:
            return f"- **{self.title}**\n  {indented_detail}"


@dataclass
class StreamEvent:
    event_type: str                   # 'messages' | 'updates' | 'sdk_partial'
    payload: Any                      # 原始报文
    stats: StreamStats                # 实时统计
    preview: Optional[str] = None     # 增量文本或工具名预览
    full_response: str = ""           # 迄今为止累计的完整模型文本
    is_delta: bool = False            # 是否为增量碎片
    structured_logs: List[EventLogItem] = field(default_factory=list) # 全量关联的事件日志链


class StreamProcessor:
    """深度结构化、按实际节点/工具命名的 LangGraph 流处理器"""

    def __init__(self):
        self.stats = StreamStats()
        self.start_time = time.time()
        
        # 核心状态机
        self.full_response = ""               
        self.log_sequence: List[EventLogItem] = [] 
        self._seen_update_items: set[Tuple[str, str, str]] = set()

        # 原始报文分类缓存
        self.buffers: Dict[str, List[Any]] = {
            "messages": [],
            "updates": [],
            "sdk_partial": [],
            "unknown": []
        }

    # ==========================================
    # 主路由入口
    # ==========================================
    def process_part(self, part: Any) -> Optional[StreamEvent]:
        if part is None:
            return None

        current_event: Optional[StreamEvent] = None

        if isinstance(part, dict):
            p_type = part.get("type")
            if p_type == "messages":
                self.buffers["messages"].append(part)
                current_event = self._handle_messages_mode(part)
            elif p_type == "updates":
                self.buffers["updates"].append(part)
                current_event = self._handle_updates_mode(part)
            elif "event" in part:
                current_event = self._dispatch_sdk_part(part.get("event"), part.get("data"), part)
        else:
            event = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if event is not None:
                current_event = self._dispatch_sdk_part(event, data, part)

        if current_event:
            # 伴随统计数据落盘，直接用 "System Stats" 替代类名大标题
            self._add_log_item(
                category="Stats",
                title="System Stats",
                detail=f"Active Node: {self.stats.node} | {self.stats.tokens} | {self.stats.tps:.1f} t/s"
            )
            current_event.structured_logs = list(self.log_sequence)
            return current_event

        self.buffers["unknown"].append(part)
        return None

    def _dispatch_sdk_part(self, event: str, data: Any, original_part: Any) -> Optional[StreamEvent]:
        if event == "messages/partial":
            self.buffers["sdk_partial"].append(original_part)
            return self._handle_sdk_partial_mode(data, original_part)
        return None

    # ==========================================
    # Protocol 1: Messages Mode 策略处理器
    # ==========================================
    def _handle_messages_mode(self, part: Dict[str, Any]) -> Optional[StreamEvent]:
        raw_data = part.get("data")
        if not isinstance(raw_data, (tuple, list)) or len(raw_data) < 1:
            return None
        
        msg_obj = raw_data[0]
        meta = raw_data[1] if len(raw_data) > 1 and isinstance(raw_data[1], dict) else {}
        
        # 提取或重置当前节点信息
        node_name = meta.get("langgraph_node", "model")
        self.stats.node = node_name

        msg_dict = self._message_to_dict(msg_obj)
        msg_type = msg_dict.get("type", "")
        is_delta = "Chunk" in msg_type
        
        content = self._extract_text(msg_dict.get("content"))
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        preview_text = ""
        
        if "ai" in msg_type.lower():
            if content:
                preview_text = content
                self.full_response = (self.full_response + content) if is_delta else content
                # 用实际的节点名（如 "agent_core"）作为标题，后面附输出明细
                self._add_log_item("Final_Output", node_name, self.full_response)
            if tool_calls:
                preview_text = tool_calls
                # 统一为 Tool 类别：标题直接是具体的工具名，详细内容附上调用传参 func(arg)
                for c in calls_list:
                    self._add_log_item("Tool", c['name'], f"-> Call: {c['name']}({c['args']})")
                    
        elif "tool" in msg_type.lower():
            preview_text = content
            # 统一为 Tool 类别：标题直接是具体的工具名，详细内容附上返回值
            tool_name = msg_dict.get("name") or "tool"
            self._add_log_item("Tool", tool_name, f"<= Return: {content}")
            
        self._update_stats(msg_dict)
        
        return StreamEvent(
            event_type="messages", platform_payload=part, stats=self.stats,
            preview=preview_text, full_response=self.full_response, is_delta=is_delta
        )

    # ==========================================
    # Protocol 2: Updates Mode 策略处理器
    # ==========================================
    def _handle_updates_mode(self, part: Dict[str, Any]) -> Optional[StreamEvent]:
        data_dict = part.get("data")
        if not isinstance(data_dict, dict):
            return None
            
        preview_text = ""
        for node_name, node_payload in data_dict.items():
            self.stats.node = str(node_name)
            
            if isinstance(node_payload, dict) and "messages" in node_payload:
                msgs = node_payload.get("messages", [])
                for raw_msg in msgs:
                    msg_dict = self._message_to_dict(raw_msg)
                    msg_type = msg_dict.get("type", "").lower()
                    content = self._extract_text(msg_dict.get("content"))
                    tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
                    
                    if "ai" in msg_type:
                        if content:
                            self.full_response = content
                            preview_text = content
                            # 用实际的节点名作为大标题
                            self._add_log_item("Final_Output", node_name, self.full_response)
                        if tool_calls:
                            preview_text = tool_calls
                            for c in calls_list:
                                self._add_log_item("Tool", c['name'], f"-> Call: {c['name']}({c['args']})")
                    elif "tool" in msg_type:
                        preview_text = content
                        tool_name = msg_dict.get("name") or "tool"
                        self._add_log_item("Tool", tool_name, f"<= Return: {content}")
                        
                    self._update_stats(msg_dict)
            else:
                # 状态/中间件更新，直接以中间件或节点本身作为标题
                state_desc = str(node_payload) if node_payload is not None else "Execution completed."
                self._add_log_item("Node", node_name, state_desc)
                preview_text = state_desc

        return StreamEvent(
            event_type="updates", payload=part, stats=self.stats,
            preview=preview_text, full_response=self.full_response, is_delta=False
        )

    # ==========================================
    # Protocol 3: LangGraph SDK Partial 策略处理器
    # ==========================================
    def _handle_sdk_partial_mode(self, data: Any, original_part: Any) -> Optional[StreamEvent]:
        if not isinstance(data, list) or len(data) < 1:
            return None
            
        msg_dict = data[0]
        node_name = "model"
        self.stats.node = node_name
        
        content = self._extract_text(msg_dict.get("content"))
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        preview_text = ""
        if content:
            preview_text = content
            self.full_response += content
            # 用实际的节点名作为大标题
            self._add_log_item("Final_Output", node_name, self.full_response)
            
        if tool_calls:
            preview_text = tool_calls
            for c in calls_list:
                self._add_log_item("Tool", c['name'], f"-> Call: {c['name']}({c['args']})")
            
        self._update_stats(msg_dict)
        
        return StreamEvent(
            event_type="sdk_partial", payload=original_part, stats=self.stats,
            preview=preview_text, full_response=self.full_response, is_delta=True
        )

    # ==========================================
    # 数据格式化辅助标准件
    # ==========================================
    def _parse_tool_calls_with_meta(self, msg_dict: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        tool_calls = msg_dict.get("tool_calls") or msg_dict.get("tool_call_chunks") or []
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not tool_calls:
            func_call = (msg_dict.get("additional_kwargs") or {}).get("function_call")
            if isinstance(func_call, dict):
                tool_calls = [{"name": func_call.get("name"), "args": func_call.get("arguments")}]

        if not isinstance(tool_calls, (list, tuple)):
            return "", []

        extracted_string_list = []
        structured_meta_list = []

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or call.get("function", {}).get("name") or "tool"
            args = call.get("args") or call.get("arguments") or ""
            
            if isinstance(args, dict):
                try: args_str = json.dumps(args, ensure_ascii=False)
                except Exception: args_str = str(args)
            else:
                args_str = str(args).strip()
                
            args_str = args_str.replace("\n", "").replace("  ", "")
            extracted_string_list.append(f"{name}({args_str})")
            structured_meta_list.append({"name": name, "args": args_str})
            
        return "\n".join(extracted_string_list), structured_meta_list

    def _add_log_item(self, category: str, title: str, detail: str):
        """去重与更新核心过滤器"""
        clean_detail = detail.strip()
        if not clean_detail:
            return
            
        # 大模型最终文本和状态统计：采取就地覆盖最新明细的处理
        if category in ("Final_Output", "Stats"):
            for item in reversed(self.log_sequence):
                if item.category == category and item.title == title:
                    item.detail = clean_detail
                    return
        
        # 工具和状态节点：如果对同一个工具进行“调用（Call）”和“返回（Return）”，
        # 我们让它们在同一个工具名标题下进行内容内聚追加，而不是开辟新的一行
        if category == "Tool":
            for item in reversed(self.log_sequence):
                if item.category == "Tool" and item.title == title:
                    # 如果详情已经包含该段子文本，直接跳过（解决协议重放）
                    if clean_detail[:100] in item.detail:
                        return
                    # 将调用参数和返回值序列追加到同一个工具节点的明细下
                    item.detail = f"{item.detail}\n{clean_detail}"
                    return

        # 兜底：对完全相同的事件进行去重
        dedup_key = (category, title, clean_detail[:300])
        if dedup_key in self._seen_update_items:
            return
        self._seen_update_items.add(dedup_key)
        
        # 实例化结构化节点
        self.log_sequence.append(EventLogItem(category=category, title=title, detail=clean_detail))
        if len(self.log_sequence) > 40:
            self.log_sequence.pop(0)

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            bits = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text": bits.append(str(item.get("text", "")))
                    elif item.get("type") in {"image_url", "image", "file"}: bits.append(f"[{item.get('type')}]")
                elif item is not None: bits.append(str(item))
            return "".join(bits)
        if isinstance(content, dict):
            return str(content.get("text", "")) if content.get("type") == "text" else str(content)
        return str(content or "").strip()

    def _message_to_dict(self, msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict): return msg
        return {
            "type": type(msg).__name__, "content": getattr(msg, "content", ""),
            "response_metadata": getattr(msg, "response_metadata", {}) or {},
            "usage_metadata": getattr(msg, "usage_metadata", {}) or {},
            "additional_kwargs": getattr(msg, "additional_kwargs", {}) or {},
            "tool_calls": getattr(msg, "tool_calls", None),
            "tool_call_chunks": getattr(msg, "tool_call_chunks", None),
        }

    def _update_stats(self, msg_dict: Dict[str, Any]):
        meta = msg_dict.get("response_metadata") or {}
        usage = msg_dict.get("usage_metadata") or {}
        model_name = meta.get("model_name") or meta.get("model")
        if model_name: self.stats.model = str(model_name)
        in_t = usage.get("input_tokens")
        out_t = usage.get("output_tokens")
        total_t = usage.get("total_tokens")
        if isinstance(in_t, int): self.stats.input_tokens = max(self.stats.input_tokens, in_t)
        if isinstance(out_t, int): self.stats.output_tokens = max(self.stats.output_tokens, out_t)
        if isinstance(total_t, int): self.stats.total_tokens = max(self.stats.total_tokens, total_t)
        elapsed = max(time.time() - self.start_time, 1e-6)
        if self.stats.output_tokens: self.stats.tps = self.stats.output_tokens / elapsed
        self.stats.tokens = f"In: {self.stats.input_tokens} | Out: {self.stats.output_tokens} | Total: {self.stats.total_tokens}"

    def get_presentation_markdown(self) -> str:
        """渲染输出：完全剔除了大分类标签，只呈现真实节点/工具与其名下绑定的明细"""
        return "\n\n".join(item.to_markdown() for item in self.log_sequence)