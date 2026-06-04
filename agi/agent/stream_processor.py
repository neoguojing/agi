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

StreamPart(event='messages/partial', data=[{'content': [], 'additional_kwargs': {'function_call': {'name': 'task', 'arguments': 
'{"description": "Provide a concise summary of today\'s (June 4, 2026) key news, focusing on the SpaceX IPO timeline, AI infrastructure sector
(especially compute deals like Anthropic-xAI), and significant macroeconomic indicators to inform investment strategy.", "subagent_type": 
"web-search-expert"}'}, '__gemini_function_call_thought_signatures__': {'6c2cc884-b789-4074-a991-8a16435ac020': 
'EjQKMgEMOdbHhTXRMo/SbNR3zL/f8VNkMdJ8Vw7fCwd2hgk1eqyD7Oik02GAQh3aAYqzlSRv'}}, 'response_metadata': {'safety_ratings': [], 'model_provider': 
'google_genai', 'finish_reason': 'STOP', 'model_name': 'gemini-3.1-flash-lite'}, 'type': 'ai', 'name': None, 'id': 
'lc_run--019e9075-7ffd-7e53-b038-4099630288bb', 'tool_calls': [{'name': 'task', 'args': {'description': "Provide a concise summary of today's 
(June 4, 2026) key news, focusing on the SpaceX IPO timeline, AI infrastructure sector (especially compute deals like Anthropic-xAI), and 
significant macroeconomic indicators to inform investment strategy.", 'subagent_type': 'web-search-expert'}, 'id': 
'6c2cc884-b789-4074-a991-8a16435ac020', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 14846, 
'output_tokens': 78, 'total_tokens': 14924, 'input_token_details': {'cache_read': 0}}}], id='1780540082776-0')

StreamPart(event='messages/complete', data=[{'content': 'There was a exception happend for get_weather_info:cannot unpack 
non-iterable NoneType object', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'Weather', 'id': 
'74fc328a-9de3-456d-8afd-b648d7f63629', 'tool_call_id': '574916fe-e725-4d36-a13d-4407f28b0d60', 'artifact': None, 'status': 'success'}], 
id='1780538915532-0')

StreamPart(event='metadata', data={'status': 'run_done', 'run_id': '019e9075-7c1c-7430-ad26-5e1c9799e8e3'}, id='1780540101193-0')
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
        
        # 工具链上下文精准追踪注册表
        self.active_tools_registry: Dict[str, str] = {}
        self.fallback_last_tool_name = ""

        self.log_sequence: List[EventLogItem] = [] 
        self.buffers: Dict[str, List[Any]] = {
            "messages": [], "updates": [], "sdk_partial": [], "metadata": [], "unknown": []
        }

    def reset_session_buffers(self):
        """🧹 深度清理上下文缓存与 Token 计数器，防止多轮对话间消息黏连与统计锁死"""
        self.full_response = ""
        self.active_tools_registry.clear()
        self.fallback_last_tool_name = ""
        self.start_time = time.time()
        
        # 💡 核心修复：每轮开始时必须重置计数器，确保 max() 不会被上一轮的老数据锁死
        self.stats.input_tokens = 0
        self.stats.output_tokens = 0
        self.stats.total_tokens = 0
        self.stats.tps = 0.0
        self.stats.tokens = "In: 0 | Out: 0 | Total: 0"

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
            # 兼容带有属性的对象格式 (如 StreamPart 对象)
            event = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if event is not None:
                current_event = self._dispatch_sdk_part(event, data, part)

        if current_event:
            current_event.structured_logs = list(self.log_sequence)
            return current_event
        return None

    def _dispatch_sdk_part(self, event: str, data: Any, original_part: Any) -> Optional[StreamEvent]:
        """SDK 专属事件分发中心"""
        if event in ("messages/partial", "messages/complete"):
            self.buffers["sdk_partial"].append(original_part)
            return self._handle_sdk_partial_mode(data, original_part, is_complete=(event == "messages/complete"))
            
        elif event == "messages/metadata":
            # 💡【生命周期对齐】：捕获到新回合元数据，启动双重保险式重置
            self.buffers["metadata"].append(original_part)
            self.reset_session_buffers()
            
            # 从深层字典中动态提取模型名称和当前激活的节点名
            if isinstance(data, dict) and len(data) > 0:
                inner_meta = list(data.values())[0].get("metadata", {})
                node_name = inner_meta.get("langgraph_node")
                model_name = inner_meta.get("ls_model_name")
                if node_name: self.stats.node = str(node_name)
                if model_name: self.stats.model = str(model_name)

            return StreamEvent(
                event_type="messages/metadata", payload=original_part, stats=self.stats, is_delta=False,
                full_response=self.full_response
            )
            
        elif event == "metadata":
            # 兼容旧版的全局元数据运行结束信号
            self.buffers["metadata"].append(original_part)
            if isinstance(data, dict) and data.get("status") == "run_done":
                self.reset_session_buffers()
                
            return StreamEvent(
                event_type="metadata", payload=original_part, stats=self.stats, is_delta=False,
                full_response=self.full_response
            )
        return None

    # =========================================================================
    # Protocol 3: LangGraph SDK Mode
    # =========================================================================
    def _handle_sdk_partial_mode(self, data: Any, original_part: Any, is_complete: bool = False) -> Optional[StreamEvent]:
        if not isinstance(data, list) or len(data) < 1:
            return None
            
        msg_dict = self._message_to_dict(data[0])
        node_name = "model"
        self.stats.node = node_name
        
        msg_type = msg_dict.get("type", "")
        # 💡 核心优化：在 messages/partial 事件中，只要未完结，统一视为 delta 增量流
        is_delta = "Chunk" in msg_type or (not is_complete and msg_type in ("AIMessageChunk", "ai"))
        
        content = self._extract_text(msg_dict.get("content"))
        artifact = msg_dict.get("artifact")
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        if "ai" in msg_type.lower():
            if content:
                # 💡 增量文本流去重防御：防止部分协议全量返回与增量返回混杂造成的阶梯状重叠
                if is_delta:
                    if self.full_response and content.startswith(self.full_response):
                        self.full_response = content
                    else:
                        # 兼容处理：若当前分块属于不带前缀的纯增量片段，则直接追加
                        if self.full_response and not content.startswith(self.full_response[:2]):
                            self.full_response += content
                        else:
                            self.full_response = content
                else:
                    self.full_response = content
                self._add_log_item("Final_Output", node_name, self.full_response)
                
            if tool_calls:
                for c in calls_list:
                    tool_title = f"{c['name']}({c['args']})"
                    tool_call_id = str(c.get("id") or c['name'])
                    
                    self.active_tools_registry[tool_call_id] = tool_title
                    self.fallback_last_tool_name = c['name']
                    
                    self._add_log_item("Tool", tool_title, "")
                    
        elif "tool" in msg_type.lower() or msg_type == "ToolMessage":
            tool_name = str(msg_dict.get("name") or self.fallback_last_tool_name or "unknown")
            tool_call_id = str(msg_dict.get("tool_call_id") or tool_name)
            
            raw_result = artifact if artifact is not None else content
            result_str = self._raw_data_to_clean_str(raw_result)
            
            target_title = self.active_tools_registry.get(tool_call_id)
            if not target_title:
                target_title = self.active_tools_registry.get(tool_name) or f"{tool_name}(...)"
                
            self._add_log_item("Tool", target_title, result_str)
            
        self._update_stats(msg_dict)
        return StreamEvent(
            event_type="sdk_partial", payload=original_part, stats=self.stats,
            preview=content or tool_calls, full_response=self.full_response, is_delta=is_delta
        )

    # =========================================================================
    # Protocol 1: Messages Mode
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
        artifact = msg_dict.get("artifact")
        tool_calls, calls_list = self._parse_tool_calls_with_meta(msg_dict)
        
        if "ai" in msg_type.lower():
            if content:
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
                    tool_title = f"{c['name']}({c['args']})"
                    tool_call_id = str(c.get("id") or c['name'])
                    self.active_tools_registry[tool_call_id] = tool_title
                    self.fallback_last_tool_name = c['name']
                    self._add_log_item("Tool", tool_title, "")
                    
        elif "tool" in msg_type.lower() or msg_type == "ToolMessage":
            tool_name = str(msg_dict.get("name") or self.fallback_last_tool_name or "unknown")
            tool_call_id = str(msg_dict.get("tool_call_id") or tool_name)
            raw_result = artifact if artifact is not None else content
            result_str = self._raw_data_to_clean_str(raw_result)
            
            target_title = self.active_tools_registry.get(tool_call_id)
            if not target_title:
                target_title = self.active_tools_registry.get(tool_name) or f"{tool_name}(...)"
            self._add_log_item("Tool", target_title, result_str)
            
        self._update_stats(msg_dict)
        return StreamEvent(event_type="messages", payload=part, stats=self.stats, is_delta=is_delta)

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

        return StreamEvent(event_type="updates", payload=part, stats=self.stats, is_delta=False)

    # =========================================================================
    # 核心分配过滤器：覆写最新快照状态
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
            # 提取出当前要处理的纯工具方法名，例如从 "Weather({"city_name": "北京"})" 提取出 "Weather"
            current_base_name = title.split("(")[0].strip()
            
            for item in reversed(self.log_sequence):
                if item.category == "Tool":
                    existing_base_name = item.title.split("(")[0].strip()
                    
                    # 💡【精准合并】：只要工具基础名字相同，就认为是同一个工具事件周期
                    if existing_base_name == current_base_name:
                        # 如果新的 title 更完整（比如带了详细参数），把旧的残缺标题优化覆写
                        if "(" in title and "..." not in title and ("..." in item.title or "(" not in item.title):
                            item.title = title
                            
                        if clean_detail:
                            if clean_detail in item.detail:
                                return
                            item.detail = clean_detail
                        return

        self.log_sequence.append(EventLogItem(category=category, title=title, detail=clean_detail))
        if len(self.log_sequence) > 40:
            self.log_sequence.pop(0)

    # =========================================================================
    # 泛化数据解析清洗工具箱
    # =========================================================================
    def _raw_data_to_clean_str(self, data: Any) -> str:
        if data is None: return ""
        if isinstance(data, str): return data.strip()
        try: return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception: return str(data)

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
            "artifact": getattr(msg, "artifact", None),
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
        
        if isinstance(in_t, int) and in_t > 0: self.stats.input_tokens = max(self.stats.input_tokens, in_t)
        if isinstance(out_t, int) and out_t > 0: self.stats.output_tokens = max(self.stats.output_tokens, out_t)
        if isinstance(total_t, int) and total_t > 0: self.stats.total_tokens = max(self.stats.total_tokens, total_t)
        
        elapsed = max(time.time() - self.start_time, 1e-6)
        if self.stats.output_tokens: self.stats.tps = self.stats.output_tokens / elapsed
        self.stats.tokens = f"In: {self.stats.input_tokens} | Out: {self.stats.output_tokens} | Total: {self.stats.total_tokens}"