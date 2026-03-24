from datetime import datetime
import shlex
from typing import Any, Callable, Awaitable, Annotated

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict
from agi.utils.common import append_to_system_message


# =========================
# State 定义
# =========================

class FileData(TypedDict, total=False):
    content: list[str]
    created_at: str
    modified_at: str
    type: str


class FfmpegState(AgentState):
    files: Annotated[NotRequired[dict[str, FileData]], "_file_data_reducer"]


# =========================
# Middleware
# =========================

class FfmpegMiddleware(AgentMiddleware[FfmpegState, Any, Any]):

    state_schema = FfmpegState

    backend: Any
    tools: list
    _custom_system_prompt: str | None = None

    def __init__(self, *, backend: Any):
        self.backend = backend
        self.tools = self._create_tools()

    # -------------------------
    # Backend 解析
    # -------------------------

    def _get_backend(self, runtime: ToolRuntime) -> Any:
        return runtime.context.get("backend") or self.backend

    # -------------------------
    # 文件状态
    # -------------------------

    def _build_file_state(self, path: str, type_: str = "video") -> dict[str, FileData]:
        now = datetime.utcnow().isoformat()
        return {
            path: {
                "content": [],  # 可扩展为 hash 或缩略图路径
                "created_at": now,
                "modified_at": now,
                "type": type_,
            }
        }

    # -------------------------
    # 文件存在与上传
    # -------------------------

    async def _ensure_file_exists(self, backend, path: str, runtime: ToolRuntime) -> bool:
        # 先在容器/沙箱检查文件
        res = await backend.aexecute(f"test -f {shlex.quote(path)} && echo OK")
        exists = "OK" in res.output
        if exists:
            return True

        # 再从 state 中查找文件内容
        file_data = runtime.state.files.get(path) if runtime.state.files else None
        if file_data:
            # 自动上传到 sandbox/backend
            content = "\n".join(file_data.get("content", []))
            upload_res = await backend.upload_files(path, content)
            if upload_res.error:
                return False
            return True

        return False

    # -------------------------
    # Tools 创建
    # -------------------------

    def _create_tools(self):
        return [
            self._create_video_cut_tool(),
            self._create_video_snapshot_tool(),
        ]

    # 🎬 video_cut
    def _create_video_cut_tool(self):

        async def async_video_cut(
            input_path: str,
            output_path: str,
            start: float,
            duration: float,
            runtime: ToolRuntime[None, FfmpegState],
        ) -> Command | str:

            backend = self._get_backend(runtime)

            # 文件存在检查 + 上传
            if not await self._ensure_file_exists(backend, input_path, runtime):
                return f"❌ file not found: {input_path}"

            cmd = (
                f"ffmpeg -y -i {shlex.quote(input_path)} "
                f"-ss {start} -t {duration} "
                f"-c copy {shlex.quote(output_path)}"
            )
            result = await backend.aexecute(cmd)
            if result.exit_code != 0:
                return f"❌ ffmpeg error:\n{result.output}"

            # 构建文件状态
            files_update = self._build_file_state(output_path)
            return Command(
                update={
                    "files": files_update,
                    "messages": [
                        ToolMessage(
                            content=f"✅ video cut: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        def sync_video_cut(*args, **kwargs):
            raise RuntimeError("Use async version: call `async_video_cut` instead")

        return StructuredTool.from_function(
            name="video_cut",
            description="Cut a segment from a video",
            func=sync_video_cut,
            coroutine=async_video_cut,
        )

    # 🎬 video_snapshot
    def _create_video_snapshot_tool(self):

        async def async_snapshot(
            input_path: str,
            time_sec: float,
            output_path: str,
            runtime: ToolRuntime[None, FfmpegState],
        ) -> Command | str:

            backend = self._get_backend(runtime)

            if not await self._ensure_file_exists(backend, input_path, runtime):
                return f"❌ file not found: {input_path}"

            cmd = (
                f"ffmpeg -y -i {shlex.quote(input_path)} "
                f"-ss {time_sec} -vframes 1 {shlex.quote(output_path)}"
            )
            result = await backend.aexecute(cmd)
            if result.exit_code != 0:
                return f"❌ ffmpeg error:\n{result.output}"

            files_update = self._build_file_state(output_path, type_="snapshot")
            return Command(
                update={
                    "files": files_update,
                    "messages": [
                        ToolMessage(
                            content=f"✅ snapshot: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        def sync_snapshot(*args, **kwargs):
            raise RuntimeError("Use async version: call `async_snapshot` instead")

        return StructuredTool.from_function(
            name="video_snapshot",
            description="Extract frame from video",
            func=sync_snapshot,
            coroutine=async_snapshot,
        )

    # -------------------------
    # Model Hook
    # -------------------------

    def wrap_model_call(self, request: ModelRequest, handler):
        # 过滤工具，仅保留视频工具 + 原工具
        available_tools = []
        tool_names = [getattr(t, "name", None) or t.get("name") for t in self.tools]
        for tool in request.tools:
            name = getattr(tool, "name", None) or tool.get("name")
            if name in tool_names:
                available_tools.append(tool)
            else:
                available_tools.append(tool)  # 非视频工具保持不变

        request = request.override(tools=available_tools)

        # 系统 prompt
        system_prompt = self._custom_system_prompt or f"You can use the following video tools: {', '.join(tool_names)}"
        new_system_message = append_to_system_message(request.system_message, system_prompt)
        request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler):
        # 异步直接复用同步逻辑生成 request
        modified_request = self.wrap_model_call(request, lambda r: r)
        return await handler(modified_request)