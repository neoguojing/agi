import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union
from collections.abc import Awaitable
from deepagents.backends.protocol import BackendProtocol
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

# 框架标准类型依赖
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    ToolCallRequest,
)

from agi.config import (
    CONTEXT_TOOL_ARG_LENGTH,
    CONTEXT_TOOL_TRUNCATION_TEXT,
    CONTEXT_MESSAGES_TO_KEEP,
    CONTEXT_MAX_TOOL_OUTPUT_LENGTH,
    CONTEXT_PREVIEW_TOOL_TEXT_LENGTH,
)

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


class ToolContextMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """专注处理 Tool 调用的上下文管理中间件
    
    1. awrap_model_call: 全量扫描历史窗口，物理裁剪“所有”工具的过长 args 和模型的 thinking，缩短上下文。
    2. awrap_tool_call: 运行期不做截断。返回过长时，保留一部分文本预览 + 写入文件名，并引导 LLM 使用 read 工具读取。
    """

    def __init__(self, backend: Optional[BackendProtocol] = None):
        super().__init__()
        self.backend = backend
        self.tools = []

    def _get_backend(self, runtime: Any) -> Optional[BackendProtocol]:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    # =====================================================================
    # 核心逻辑 1：Model 侧 —— 裁剪“所有”工具的历史 args 以及思考过程 (Thinking)
    # =====================================================================
    def _truncate_tool_call(self,tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args", {})
        truncated_args = {}
        modified = False
        for key, value in args.items():
            if isinstance(value, str) and len(value) > CONTEXT_TOOL_ARG_LENGTH:
                truncated_args[key] = value[:20] + CONTEXT_TOOL_TRUNCATION_TEXT
                modified = True
            else:
                truncated_args[key] = value
        if modified:
            return {**tool_call, "args": truncated_args}
        return tool_call

    def _truncate_args(self,messages: List[AnyMessage]) -> tuple[List[AnyMessage], bool]:
        cutoff_index = len(messages) - CONTEXT_MESSAGES_TO_KEEP
        if cutoff_index <= 0:
            return messages, False

        truncated_messages = []
        modified = False
        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
                truncated_tool_calls = []
                msg_modified = False
                for tool_call in msg.tool_calls:
                    truncated_call = self._truncate_tool_call(tool_call)
                    if truncated_call != tool_call:
                        msg_modified = True
                    truncated_tool_calls.append(truncated_call)

                if msg_modified:
                    truncated_msg = msg.model_copy()
                    truncated_msg.tool_calls = truncated_tool_calls
                    truncated_messages.append(truncated_msg)
                    modified = True
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)
        return truncated_messages, modified

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> Union[ModelResponse[ResponseT], AIMessage, Any]:
        """异步模型拦截：发起 LLM 请求前，全量瘦身历史消息"""
        if request.messages and isinstance(request.messages, list):
            optimized_messages,modified = self._truncate_args(request.messages)
            if modified:
                request = request.override(messages=optimized_messages)
        
        return await handler(request)

    # =====================================================================
    # 核心逻辑 2：Tool 侧 —— 运行期不改参数，结果过长时截取文本预览并提示 Read 工具
    # =====================================================================
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Union[ToolMessage, Command[Any]]]],
    ) -> Union[ToolMessage, Command[Any]]:
        """异步工具拦截：保持完整入参运行，过长出参进行分流并提供 LLM 行动引导"""
        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_call_id = request.tool_call.get("id", "call_default_id")
        config = getattr(request.runtime.config, "configurable")
        thread_id =  config.get("thread_id","default")

        # 1. 运行期绝对不能改入参请求，直接交付底层 Handler 执行
        raw_result = await handler(request)

        if isinstance(raw_result, Command):
            return raw_result

        # 2. 归一化提取工具返回的 Message 结构
        if isinstance(raw_result, ToolMessage):
            final_message = raw_result
        else:
            final_message = ToolMessage(content=str(raw_result), tool_call_id=tool_call_id, name=tool_name)

        content_str = final_message.content if isinstance(final_message.content, str) else json.dumps(final_message.content, ensure_ascii=False)

        # 3. 如果结果没有超长，直接返回给大模型
        if len(content_str) <= CONTEXT_MAX_TOOL_OUTPUT_LENGTH:
            return final_message

        # 4. 结果超长，触发转储离线文件逻辑
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"{thread_id}_{tool_name}_{timestamp}.txt"
        file_path = f"/conversation_history/{file_name}"
        backend_instance = self._get_backend(request.runtime)

        try:
            # 异步或同步写入存储后端
            await backend_instance.awrite(file_path, content_str.encode("utf-8"))

            logger.info(f"Successfully dumped massive output for tool '{tool_name}' to {file_path}")

            # 5. 提取部分文本作为 Preview
            preview_text = content_str[:CONTEXT_PREVIEW_TOOL_TEXT_LENGTH]

            # 6. 【核心点】构建提示 LLM 具有明确 Call to Action 的返回话术
            guided_content = (
                f"--- TOOL OUTPUT PREVIEW ({tool_name}) ---\n"
                f"{preview_text}\n"
                f"... [Remaining {len(content_str) - CONTEXT_PREVIEW_TOOL_TEXT_LENGTH} characters hidden due to context length limits] ...\n\n"
                f"⚠️ SYSTEM NOTICE TO LLM:\n"
                f"The complete output of this tool is too large and has been safely saved to a backend file: `{file_name}`.\n"
                f"If the preview above does not contain all the details you need, please explicitly invoke your file reading tool "
                f"(e.g., `read_file(path='{file_name}')`) to inspect the remaining content."
            )

            return ToolMessage(
                content=guided_content,
                tool_call_id=final_message.tool_call_id,
                # artifact 内保留完整原始镜像和具体路径，方便下游代码级节点无感消费数据
                artifact={"full_output_path": file_path, "original_content": final_message.content, "file_name": file_name},
                status=final_message.status,
                name=final_message.name
            )

        except Exception as e:
            logger.error(f"Failed to offload massive tool output to storage: {e}")
            # 本地 IO 异常降级兜底：只能强制截断
            fallback_text = (
                f"{content_str[:CONTEXT_MAX_TOOL_OUTPUT_LENGTH]}\n"
                f"... [Content truncated due to storage IO failure: {str(e)}]"
            )
            return ToolMessage(
                content=fallback_text,
                tool_call_id=final_message.tool_call_id,
                status="error",
                name=tool_name
            )