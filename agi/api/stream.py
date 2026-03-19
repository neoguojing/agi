import json
import uuid
import time
from typing import Dict, Any, AsyncGenerator, List, Union
from agi.agent.agent import stream_agent_async
from langgraph.types import Overwrite

def _extract_messages(raw_messages: Any) -> List:
    """
    安全提取 messages 列表，处理 LangGraph V2 的 Overwrite 包装器。
    """
    if raw_messages is None:
        return []
    
    # 核心修复：如果是 Overwrite 对象，提取其 .value 属性
    if isinstance(raw_messages, Overwrite):
        return raw_messages.value if raw_messages.value else []
    
    # 如果已经是列表，直接返回
    if isinstance(raw_messages, list):
        return raw_messages
    
    # 其他未知类型，返回空列表以防崩溃
    return []

async def stream_response(state: Dict[str, Any]) -> AsyncGenerator[str, None]:
    index = 0
    has_sent_role = False

    async for part in stream_agent_async(state):
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "agi-model",
            "choices": [{
                "index": index,
                "delta": {},
                "finish_reason": None,
            }]
        }
        
        delta = {}
        finish_reason = None
        should_yield = False

        # =========================
        # 1. 处理 LLM Token 流 (messages)
        # =========================
        if part.get("type") == "messages":
            data_tuple = part.get("data")
            if not data_tuple or len(data_tuple) < 2:
                continue
                
            msg, meta = data_tuple[0], data_tuple[1]
            content = getattr(msg, "content", "")
            
            # 过滤空内容
            if content:
                if not has_sent_role:
                    delta["role"] = "assistant"
                    has_sent_role = True
                delta["content"] = content
                should_yield = True

            # 检测结束标志 (done_reason 通常在最后一个空 content 的 meta 中)
            if meta:
                reason = meta.get("done_reason") or meta.get("finish_reason")
                if reason:
                    finish_reason = reason
                    should_yield = True

        # =========================
        # 2. 处理状态更新 (updates) - 重点修复 Overwrite
        # =========================
        elif part.get("type") == "updates":
            updates_data = part.get("data", {})
            
            for node_name, node_data in updates_data.items():
                if not isinstance(node_data, dict):
                    continue
                
                # A. 处理中断 (__interrupt__)
                interrupt = node_data.get("__interrupt__")
                if interrupt:
                    interrupt_value = ""
                    if isinstance(interrupt, list) and len(interrupt) > 0:
                        item = interrupt[0]
                        interrupt_value = str(item.value) if hasattr(item, 'value') else str(item)
                    
                    if interrupt_value:
                        if not has_sent_role:
                            delta["role"] = "assistant"
                            has_sent_role = True
                        delta["content"] = f"\n[系统中断]: {interrupt_value}\n"
                        finish_reason = "stop"
                        should_yield = True
                    break

                # B. 处理消息更新 (messages) - 这里最容易遇到 Overwrite
                raw_msgs = node_data.get("messages")
                if raw_msgs is not None:
                    # 【关键修复】使用 helper 函数解包 Overwrite
                    messages_list = _extract_messages(raw_msgs)
                    
                    if messages_list:
                        last_msg = messages_list[-1]
                        content = getattr(last_msg, "content", "")
                        
                        # 如果是工具调用后的完整消息回显，或者需要通知前端状态变化
                        # 注意：通常 SSE 主要靠 'messages' type 流式传输，
                        # 'updates' 里的 messages 往往是完整的最终消息。
                        # 如果你需要在这里也推送内容，可以取消下面的注释：
                        
                        # if content and not has_sent_role: 
                        #     delta["role"] = "assistant"
                        #     delta["content"] = content
                        #     has_sent_role = True
                        #     should_yield = True
                        
                        # 或者，你可以在这里检测 tool_calls 并发送自定义事件
                        tool_calls = getattr(last_msg, "tool_calls", [])
                        if tool_calls:
                            # 示例：发送工具调用通知（非标准 OpenAI 格式，需前端支持）
                            # 这里仅做演示，通常工具调用会在 'messages' 流中以 chunk 形式出现
                            pass 

        # =========================
        # 3. 发送数据
        # =========================
        if should_yield:
            if delta:
                chunk["choices"][0]["delta"] = delta
            if finish_reason:
                chunk["choices"][0]["finish_reason"] = finish_reason
            
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            index += 1

    yield "data: [DONE]\n\n"