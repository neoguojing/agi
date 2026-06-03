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
    """纯粹的数据传输结构，供外部 UI / TUI 层按需渲染"""
    category: str       # 'Node' | 'Tool' | 'Final_Output'
    title: str          # 标准名称：模型节点名（如 agent），或格式化好的完整工具名 func(arg)
    detail: str         # 纯明细文本：大模型流式输出全量，或纯工具返回值字符串
    timestamp: float = field(default_factory=time.time)


@dataclass
class StreamEvent:
    event_type: str
    payload: Any
    stats: StreamStats
    is_delta: bool
    preview: str = ""
    full_response: str = ""
    structured_logs: List[EventLogItem] = field(default_factory=list)


class StreamProcessor:
    def __init__(self):
        self.stats = StreamStats()
        self.start_time = time.time()
        
        # 核心全量状态事实源
        self.full_response = ""               
        
        # 🛠️ 工具链上下文精准追踪注册表：
        # 键为具有唯一性的 tool_call_id（或工具名），值映射为格式化好的 "func(arg)" 完备形态
        self.active_tools_registry: Dict[str, str] = {}
        self.fallback_last_tool_name = ""

        self.log_sequence: List[EventLogItem] = [] 
        self.buffers: Dict[str, List[Any]] = {
            "messages": [], "updates": [], "sdk_partial": [], "unknown": []
        }

    def process_part(self, part: Any) -> Optional[StreamEvent]:
        """流式数据唯一入口：智能识别不同协议并完成数据的清洗过滤"""
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
            # 外部渲染层直接遍历此链表即可拿到当前生命周期内最干净的内容
            current_event.structured_logs = list(self.log_sequence)
            return current_event
        return None

    def _dispatch_sdk_part(self, event: str, data: Any, original_part: Any) -> Optional[StreamEvent]:
        if event == "messages/partial":
            self.buffers["sdk_partial"].append(original_part)
            return self._handle_sdk_partial_mode(data, original_part)
        return None

    # =========================================================================
    # Protocol 1: Messages Mode (包含核心 ToolMessage.artifact 深度穿透捕获)
    # =========================================================================
    def _handle_messages_mode(self, part: Dict[str, Any]) -> Optional[StreamEvent]:
        raw_data = part.get("data")
        if not isinstance(raw_data, (tuple, list)) or len(raw_data) < 1:
            return None
        
        msg_obj = raw_data[0]
        meta = raw_data[1] if len(raw_data) > 1 and isinstance(raw_data[1], dict) else {}
        node_name = meta.get("langgraph_node", "model")
        self.stats.node = node_name

        msg_dict = self._message_to_dict(msg_obj)
        msg_type = msg_dict.get("type", "")
        is_delta = "Chunk" in msg_type
        
        content = self._extract_text(msg_dict.get("content"))
        artifact = msg_dict.get("artifact")  # 🌟 捕获关键的高阶返回容器
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        if "ai" in msg_type.lower():
            if content:
                # 增量去重纠正：防止全量与增量混合导致的阶梯状堆叠
                if is_delta:
                    if self.full_response and content.startswith(self.full_response):
                        self.full_response = content
                    else:
                        self.full_response += content
                else:
                    self.full_response = content
                self._add_log_item("Final_Output", node_name, self.full_response)
                
            if tool_calls:
                for c in calls_list:
                    # 1. 构建符合要求的标准 func(arg) 格式大标题
                    tool_title = f"{c['name']}({c['args']})"
                    tool_call_id = str(c.get("id") or c['name'])
                    
                    # 2. 注入注册表，提供给接下来可能由于没有参数而面临失联的 ToolMessage
                    self.active_tools_registry[tool_call_id] = tool_title
                    self.fallback_last_tool_name = c['name']
                    
                    # 3. 初始投递，此时还没有返回值，detail 给空
                    self._add_log_item("Tool", tool_title, "")
                    
        elif "tool" in msg_type.lower() or msg_type == "ToolMessage":
            # 💡【权威修复】：拦截 ToolMessage 报文
            tool_name = str(msg_dict.get("name") or self.fallback_last_tool_name or "unknown")
            tool_call_id = str(msg_dict.get("tool_call_id") or tool_name)
            
            # 优先提取高级扩展字段 artifact 里的完整数据，如果没有才降级到 content 文本
            raw_result = artifact if artifact is not None else content
            result_str = self._raw_data_to_clean_str(raw_result)
            
            # 检索它应该归宿的那个带有全量参数的 func(arg) 卡片标题
            target_title = self.active_tools_registry.get(tool_call_id)
            if not target_title:
                target_title = self.active_tools_registry.get(tool_name) or f"{tool_name}(...)"
                
            self._add_log_item("Tool", target_title, result_str)
            
        self._update_stats(msg_dict)
        return StreamEvent(
            event_type="messages", payload=part, stats=self.stats,
            preview=content or tool_calls, full_response=self.full_response, is_delta=is_delta
        )

    # =========================================================================
    # Protocol 2: Updates Mode
    # =========================================================================
    def _handle_updates_mode(self, part: Dict[str, Any]) -> Optional[StreamEvent]:
        data_dict = part.get("data")
        if not isinstance(data_dict, dict):
            return None
            
        for node_name, node_payload in data_dict.items():
            self.stats.node = str(node_name)
            
            if isinstance(node_payload, dict) and "messages" in node_payload:
                msgs = node_payload.get("messages", [])
                for raw_msg in msgs:
                    msg_dict = self._message_to_dict(raw_msg)
                    msg_type = msg_dict.get("type", "").lower()
                    content = self._extract_text(msg_dict.get("content"))
                    artifact = msg_dict.get("artifact")
                    tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
                    
                    if "ai" in msg_type:
                        if content:
                            self.full_response = content
                            self._add_log_item("Final_Output", node_name, self.full_response)
                        if tool_calls:
                            for c in calls_list:
                                tool_title = f"{c['name']}({c['args']})"
                                tool_call_id = str(c.get("id") or c['name'])
                                self.active_tools_registry[tool_call_id] = tool_title
                                self.fallback_last_tool_name = c['name']
                                self._add_log_item("Tool", tool_title, "")
                    elif "tool" in msg_type:
                        tool_name = str(msg_dict.get("name") or self.fallback_last_tool_name or "unknown")
                        tool_call_id = str(msg_dict.get("tool_call_id") or tool_name)
                        
                        raw_result = artifact if artifact is not None else content
                        result_str = self._raw_data_to_clean_str(raw_result)
                        
                        target_title = self.active_tools_registry.get(tool_call_id) or self.active_tools_registry.get(tool_name) or f"{tool_name}(...)"
                        self._add_log_item("Tool", target_title, result_str)
            else:
                state_desc = str(node_payload) if node_payload is not None else "Completed."
                self._add_log_item("Node", node_name, state_desc)

        return StreamEvent(
            event_type="updates", payload=part, stats=self.stats,
            preview=self.full_response, full_response=self.full_response, is_delta=False
        )

    # =========================================================================
    # Protocol 3: LangGraph SDK Partial Mode
    # =========================================================================
    def _handle_sdk_partial_mode(self, data: Any, original_part: Any) -> Optional[StreamEvent]:
        if not isinstance(data, list) or len(data) < 1:
            return None
            
        msg_dict = data[0]
        node_name = "model"
        self.stats.node = node_name
        
        content = self._extract_text(msg_dict.get("content"))
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        if content:
            if self.full_response and content.startswith(self.full_response):
                self.full_response = content
            else:
                self.full_response += content
            self._add_log_item("Final_Output", node_name, self.full_response)
            
        if tool_calls:
            for c in calls_list:
                tool_title = f"{c['name']}({c['args']})"
                tool_call_id = str(c.get("id") or c['name'])
                self.active_tools_registry[tool_call_id] = tool_title
                self.fallback_last_tool_name = c['name']
                self._add_log_item("Tool", tool_title, "")
            
        self._update_stats(msg_dict)
        return StreamEvent(
            event_type="sdk_partial", payload=original_part, stats=self.stats,
            preview=content or tool_calls, full_response=self.full_response, is_delta=True
        )

    # =========================================================================
    # 核心分配过滤器：全部采用“原地全量最新快照清洗覆盖”，不再在内部追加 \n
    # =========================================================================
    def _add_log_item(self, category: str, title: str, detail: str):
        clean_detail = detail.strip()
        
        if category in ("Final_Output", "Node"):
            if not clean_detail:
                return
            for item in reversed(self.log_sequence):
                if item.category == category and item.title == title:
                    item.detail = clean_detail  
                    return

        if category == "Tool":
            for item in reversed(self.log_sequence):
                if item.category == "Tool" and item.title == title:
                    if clean_detail:
                        if clean_detail in item.detail:
                            return
                        # 数据层保持完全干净的覆写。外部 TUI 消费时读取这个最新的 detail
                        # 换行和样式控制完全由终端代码按需渲染，解耦业务与展现
                        item.detail = clean_detail
                    return

        self.log_sequence.append(EventLogItem(category=category, title=title, detail=clean_detail))
        if len(self.log_sequence) > 40:
            self.log_sequence.pop(0)

    # =========================================================================
    # 底层泛化抽取清洗工具组
    # =========================================================================
    def _raw_data_to_clean_str(self, data: Any) -> str:
        """安全地将复杂对象或基础返回转化为紧凑干净的纯字符串"""
        if data is None:
            return ""
        if isinstance(data, str):
            return data.strip()
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(data)

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
            call_id = call.get("id")
            
            if isinstance(args, dict):
                try: args_str = json.dumps(args, ensure_ascii=False)
                except Exception: args_str = str(args)
            else:
                args_str = str(args).strip()
                
            # 清洗参数中的剧烈换行以精简格式
            args_str = args_str.replace("\n", "").replace("  ", "")
            extracted_string_list.append(f"{name}({args_str})")
            
            meta_payload = {"name": name, "args": args_str}
            if call_id:
                meta_payload["id"] = str(call_id)
            structured_meta_list.append(meta_payload)
            
        return "\n".join(extracted_string_list), structured_meta_list

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
            "type": type(msg).__name__, 
            "content": getattr(msg, "content", ""),
            "name": getattr(msg, "name", None),
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "artifact": getattr(msg, "artifact", None),  # 🌟 射入基础映射反射
            "response_metadata": getattr(msg, "response_metadata", {}) or {},
            "usage_metadata": getattr(msg, "usage_metadata", {}) or {},
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