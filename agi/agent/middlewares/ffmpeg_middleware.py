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
from agi.agent.sandbox.docker import DockerSandbox

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

    backend: DockerSandbox
    tools: list
    _custom_system_prompt: str | None = None

    def __init__(self, *, backend: DockerSandbox ):
        self.backend = backend
        self.tools = self._create_tools()

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
            self._create_video_concat_tool(),
            self._create_video_resize_tool(),
            self._create_video_crop_tool(),
            self._create_video_add_text_tool(),
            self._create_video_overlay_tool(),
        ]

    # 🎬 video_cut
    def _create_video_cut_tool(self):
        tool_description = (
            "Trim a video precisely with re-encoding. "
            "More accurate than fast cut but slower."
        )

        async def async_video_trim(
            runtime: ToolRuntime[None, FfmpegState],

            input_path: Annotated[str, "Input video path"],
            output_path: Annotated[str, "Output video path"],
            start: Annotated[float, "Start time in seconds"],
            end: Annotated[float, "End time in seconds"],
        ) -> Command | str:

            cmd = (
                f"ffmpeg -y -i {input_path} "
                f"-ss {start} -to {end} "
                f"-c:v libx264 -c:a aac {output_path}"
            )

            result = await self.backend.aexecute(cmd)
            if result.exit_code != 0:
                return result.output

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ trim video from {start}s → {end},output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_cut",
            description=tool_description,
            coroutine=async_video_trim,
        )
    # 🎬 video_snapshot
    def _create_video_snapshot_tool(self):
        tool_description = (
            "Extract a single frame (snapshot) from a video at a specific time. "
            "Useful for thumbnails or preview images. Time is in seconds."
        )

        async def async_snapshot(
            runtime: ToolRuntime[None, FfmpegState],

            input_path: Annotated[
                str,
                "Absolute path to the input video file."
            ],

            time_sec: Annotated[
                float,
                "Timestamp in seconds where the frame should be extracted."
            ],

            output_path: Annotated[
                str,
                "Absolute path for the output image (e.g., .jpg or .png)."
            ],
        ) -> Command | str:

            backend = self.backend

            if not await self._ensure_file_exists(backend, input_path, runtime):
                return f"❌ file not found: {input_path}"

            cmd = (
                f"ffmpeg -y -i {shlex.quote(input_path)} "
                f"-ss {time_sec} -vframes 1 "
                f"{shlex.quote(output_path)}"
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
                            content=f"✅ snapshot created at {time_sec}s → {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )

        def sync_snapshot(*args, **kwargs):
            raise RuntimeError("Use async version")

        return StructuredTool.from_function(
            name="video_snapshot",
            description=tool_description,
            func=sync_snapshot,
            coroutine=async_snapshot,
        )
    
    def _create_video_concat_tool(self):
        tool_description = (
            "Concatenate multiple video files into one. "
            "All videos must have same codec, resolution, and format."
        )

        async def async_video_concat(
            runtime: ToolRuntime[None, FfmpegState],
            input_paths: Annotated[
                list[str],
                "List of absolute paths to input video files in order."
            ],
            output_path: Annotated[
                str,
                "Absolute path for the merged output video."
            ],
        ) -> Command | str:

            backend = self.backend

            # 构建 concat list 文件
            list_file = "/tmp/concat_list.txt"
            content = "\n".join([f"file '{p}'" for p in input_paths])
            await backend.awrite(list_file, content)

            cmd = (
                f"ffmpeg -y -f concat -safe 0 -i {list_file} "
                f"-c copy {output_path}"
            )

            result = await backend.aexecute(cmd)
            if result.exit_code != 0:
                return f"❌ ffmpeg error:\n{result.output}"

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ concatenate success,output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_concat",
            description=tool_description,
            coroutine=async_video_concat,
        )
    
    def _create_video_resize_tool(self):
        tool_description = "Resize video resolution."

        async def async_resize(
            runtime: ToolRuntime[None, FfmpegState],

            input_path: Annotated[str, "Input video path"],
            output_path: Annotated[str, "Output video path"],
            width: Annotated[int, "Target width in pixels"],
            height: Annotated[int, "Target height in pixels"],
        ):

            cmd = (
                f"ffmpeg -y -i {input_path} "
                f"-vf scale={width}:{height} {output_path}"
            )

            result = await self.backend.aexecute(cmd)
            if result.exit_code != 0:
                return result.output

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ resize video to width:{width},height:{height},output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_resize",
            description=tool_description,
            coroutine=async_resize,
        )
    
    def _create_video_crop_tool(self):
        tool_description = "Crop a region from video."

        async def async_crop(
            runtime: ToolRuntime[None, FfmpegState],

            input_path: Annotated[str, "Input video"],
            output_path: Annotated[str, "Output video"],
            x: Annotated[int, "Top-left x coordinate"],
            y: Annotated[int, "Top-left y coordinate"],
            width: Annotated[int, "Crop width"],
            height: Annotated[int, "Crop height"],
        ):

            cmd = (
                f"ffmpeg -y -i {input_path} "
                f"-vf crop={width}:{height}:{x}:{y} "
                f"{output_path}"
            )

            result = await self.backend.aexecute(cmd)
            if result.exit_code != 0:
                return result.output

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ crop video to width:{width},height:{height},output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_crop",
            description=tool_description,
            coroutine=async_crop,
        )
    
    def _create_video_add_text_tool(self):

        tool_description = "Overlay text on video."

        async def async_text(
            runtime: ToolRuntime[None, FfmpegState],

            input_path: Annotated[str, "Input video"],
            output_path: Annotated[str, "Output video"],
            text: Annotated[str, "Text content"],
            x: Annotated[int, "X position"],
            y: Annotated[int, "Y position"],
        ):

            cmd = (
                f"ffmpeg -y -i {input_path} "
                f"-vf drawtext=text='{text}':x={x}:y={y}:fontsize=24:fontcolor=white "
                f"{output_path}"
            )

            result = await self.backend.aexecute(cmd)
            if result.exit_code != 0:
                return result.output

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ Overlay text on video:{text},output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_add_text",
            description=tool_description,
            coroutine=async_text,
        )
    
    def _create_video_overlay_tool(self):

        tool_description = "Overlay an image watermark on video."

        async def async_overlay(
            runtime: ToolRuntime[None, FfmpegState],

            video_path: Annotated[str, "Input video"],
            image_path: Annotated[str, "Overlay image"],
            output_path: Annotated[str, "Output video"],
        ):

            cmd = (
                f"ffmpeg -y -i {video_path} -i {image_path} "
                f"-filter_complex overlay=10:10 "
                f"{output_path}"
            )

            result = await self.backend.aexecute(cmd)
            if result.exit_code != 0:
                return result.output

            return Command(update={
                "files": self._build_file_state(output_path),
                "messages": [
                        ToolMessage(
                            content=f"✅ Overlay an image watermark:{image_path},output path: {output_path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                })
        
        return StructuredTool.from_function(
            name="video_watermark",
            description=tool_description,
            coroutine=async_overlay,
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