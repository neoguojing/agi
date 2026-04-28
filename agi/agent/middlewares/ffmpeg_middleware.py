import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import shlex
from typing import Any, Annotated
import asyncio

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict
from agi.utils.common import append_to_system_message
from agi.agent.sandbox.docker import DockerSandbox
from agi.agent.prompt import get_middleware_prompt

logger = logging.getLogger(__name__)


class FileData(TypedDict, total=False):
    content: list[str]
    created_at: str
    modified_at: str
    type: str
    status: str
    container_path: str
    local_path: str
    operation: str


def _file_data_reducer(
    left: dict[str, FileData] | None,
    right: dict[str, FileData | None],
) -> dict[str, FileData]:
    """Merge file updates and support deletions via None markers."""
    if left is None:
        return {key: value for key, value in right.items() if value is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _last_operation_reducer(left: dict[str, Any] | None, right: dict[str, Any]) -> dict[str, Any]:
    """Keep the latest operation payload."""
    _ = left
    return right


class FfmpegState(AgentState):
    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    last_operation: Annotated[NotRequired[dict[str, Any]], _last_operation_reducer]


# =========================
# Middleware
# =========================

# Configuration constants
MAX_EXECUTE_TIMEOUT = 300  # seconds - max timeout for FFmpeg operations
MAX_VIDEO_SIZE_MB = 2048  # 2GB limit for uploaded files
DEFAULT_OUTPUT_DIR = "/workspace/outputs"  # default output path in container


class FfmpegMiddleware(AgentMiddleware[FfmpegState, Any, Any]):

    state_schema = FfmpegState

    backend: DockerSandbox
    tools: list
    _custom_system_prompt: str | None = None

    def __init__(self, *, backend: DockerSandbox, output_dir: str = DEFAULT_OUTPUT_DIR):
        """Initialize FFmpeg middleware.

        Args:
            backend: Docker sandbox backend
            output_dir: Output directory path in container (not used for host)
        """
        self.backend = backend
        self.tools = self._create_tools()

    def _resolve_user_id(self, runtime: ToolRuntime[None, FfmpegState] | None = None) -> str:
        """Extract user_id from runtime context or config."""
        if runtime is not None:
            context = getattr(runtime, "context", None)
            if context and context.user_id:
                return str(context.user_id)
            config = getattr(runtime, "config", {}) or {}
            if isinstance(config, dict) and "configurable" in config:
                configurable = config["configurable"]
                if configurable.get("user_id"):
                    return str(configurable["user_id"])
        raise ValueError("user_id is required for ffmpeg tools")

    def _backend_for_runtime(self, runtime: ToolRuntime[None, FfmpegState]) -> Any:
        """Get backend instance for the given runtime."""
        return self.backend.for_user(self._resolve_user_id(runtime))

    def _validate_path(self, path: str) -> str:
        """Validate and normalize path, return empty string if invalid."""
        if not path:
            return ""
        normalized = str(Path(path).expanduser().resolve())
        if not normalized.startswith("/") and not Path(normalized).is_absolute():
            logger.warning(f"Non-absolute path rejected: {path}")
        return normalized

    def _validate_ffmpeg_command(self, cmd: str) -> tuple[bool, str | None]:
        """Validate FFmpeg command for safety and constraints.

        Args:
            cmd: FFmpeg command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for dangerous shell patterns
        dangerous_patterns = ["rm -rf", "mkfs", "dd if=/dev/zero", "nc ", "curl ", "wget "]
        for pattern in dangerous_patterns:
            if pattern in cmd.lower():
                return False, f"Blocked dangerous pattern: {pattern}"
        # Check command length (4KB is generous for ffmpeg)
        if len(cmd) > 4096:
            return False, f"Command exceeds limit: {len(cmd)} > 4096 chars"
        return True, None

    async def _run_ffmpeg(self, backend: Any, cmd: str, *, timeout: int = MAX_EXECUTE_TIMEOUT) -> str | None:
        """Execute FFmpeg command with error handling.

        Args:
            backend: Docker backend
            cmd: FFmpeg command
            timeout: Execution timeout in seconds

        Returns:
            Error message on failure, None on success
        """
        # Validate command before execution
        is_valid, error = self._validate_ffmpeg_command(cmd)
        if not is_valid:
            logger.warning(f"Command rejected: {error}")
            return error

        try:
            result = await backend.aexecute(cmd, timeout=timeout)
            if result.exit_code != 0:
                error_msg = (result.output or "").strip() or f"Exit code {result.exit_code}"

                if "No such file or directory" in error_msg:
                    logger.warning(f"FFmpeg binary missing: {cmd[:100]}")
                elif "Invalid data found when processing input" in error_msg:
                    logger.warning(f"Invalid input file: {cmd[:100]}")
                elif "timeout" in error_msg.lower():
                    logger.warning(f"FFmpeg timeout: {cmd[:100]}")
                elif result.exit_code == 1:
                    logger.warning(f"FFmpeg encoding error: {cmd[:100]}")
                return error_msg
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Command timeout: {cmd[:100]}")
            return f"Timed out after {timeout}s"
        except FileNotFoundError:
            logger.error(f"FFmpeg binary not found", exc_info=True)
            return "FFmpeg binary missing in container"
        except PermissionError:
            logger.error(f"Permission denied: {cmd[:100]}", exc_info=True)
            return "Permission denied"
        except BrokenPipeError:
            logger.error(f"Pipe broken: {cmd[:100]}", exc_info=True)
            return "Execution interrupted"
        except Exception as e:
            logger.error(f"FFmpeg error: {e}", exc_info=True)
            return str(e)

    def _tool_success(
        self,
        runtime: ToolRuntime[None, FfmpegState],
        *,
        action: str,
        output_path: str,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> Command:
        payload = {
            "action": action,
            "status": "success",
            "output_path": output_path,
            "next_step": f"Call video_download with container_path={output_path}",
        }
        if extra:
            payload.update(extra)
        return self._build_state_command(
            runtime,
            message=message,
            files_update=self._build_file_state(output_path, status="processed"),
            last_operation=payload,
        )

    def _build_state_command(
        self,
        runtime: ToolRuntime[None, FfmpegState],
        *,
        message: str,
        files_update: dict[str, FileData],
        last_operation: dict[str, Any],
    ) -> Command:
        return Command(
            update={
                "files": files_update,
                "last_operation": last_operation,
                "messages": [ToolMessage(content=message, tool_call_id=runtime.tool_call_id)],
            }
        )

    # -------------------------
    # 文件状态管理
    # -------------------------

    def _build_file_state(
        self,
        path: str,
        *,
        type_: str = "video",
        status: str = "ready",
        local_path: str | None = None,
    ) -> dict[str, FileData]:
        """Build file state dictionary for state management."""
        now = datetime.utcnow().isoformat()
        return {
            path: {
                "content": [],
                "created_at": now,
                "modified_at": now,
                "type": type_,
                "status": status,
                "container_path": path,
                "local_path": local_path or "",
            }
        }

    def _generate_tool_name(self, func_name: str) -> str:
        """Generate clean tool name from function name."""
        return func_name.replace("async_", "").replace("_", "_")

    # -------------------------
    # 文件存在与上传
    # -------------------------

    async def _ensure_file_exists(
        self, backend: Any, path: str, runtime: ToolRuntime
    ) -> tuple[bool, str | None]:
        """Check if file exists in container or state, auto-upload if in state.

        Args:
            backend: Docker backend
            path: File path to check
            runtime: ToolRuntime context

        Returns:
            Tuple of (exists, error)
        """
        # Validate path first
        validated_path = self._validate_path(path)
        if not validated_path:
            return False, f"Invalid path: {path}"

        # Check in container filesystem
        try:
            res = await backend.aexecute(f"test -f {shlex.quote(validated_path)} && echo OK", timeout=10)
            exists = "OK" in res.output
            if exists:
                logger.debug(f"File exists in container: {validated_path}")
                return True, None
        except Exception as e:
            logger.debug(f"Cannot check container filesystem for {validated_path}: {e}")

        # Check state for file content
        files_state = runtime.state.files if hasattr(runtime.state, "files") else None
        if files_state and validated_path in files_state:
            try:
                file_data = files_state[validated_path]
                content = "\n".join(file_data.get("content", []))
                if content:  # Only upload if content exists
                    upload_res = await backend.aupload_files([(validated_path, content.encode("utf-8"))])
                    if upload_res and upload_res[0].error:
                        logger.warning(f"Failed to upload from state: {upload_res[0].error}")
                        return False, upload_res[0].error
                    logger.debug(f"Auto-uploaded from state: {validated_path}")
                    return True, None
            except Exception as e:
                logger.debug(f"Failed to get file data from state: {e}")

        return False, None

    # -------------------------
    # Tools 创建
    # -------------------------

    def _create_tools(self):
        """Create and return all FFmpeg processing tools."""
        tools = [
            self._create_video_upload_tool(),
            self._create_video_download_tool(),
            self._create_video_cut_tool(),
            self._create_video_snapshot_tool(),
            self._create_video_concat_tool(),
            self._create_video_resize_tool(),
            self._create_video_crop_tool(),
            self._create_video_add_text_tool(),
            self._create_video_overlay_tool(),
        ]
        logger.info(f"Initialized {len(tools)} FFmpeg tools")
        return tools

    def _create_video_upload_tool(self):
        tool_description = (
            "Upload local files to Docker container for FFmpeg processing.\n"
            "Specify absolute paths. Use container_path in subsequent FFmpeg tools.\n"
        )

        async def async_video_upload(
            runtime: ToolRuntime[None, FfmpegState],
            local_path: Annotated[str, "Host local file path to upload. Must be absolute."],
        ) -> Command | str:
            backend = self._backend_for_runtime(runtime)

            # Validate and normalize path
            source_path = self._validate_path(local_path)
            if not source_path:
                return f"❌ Invalid path: {local_path}"

            try:
                # Check if source file exists
                if not Path(source_path).exists() or not Path(source_path).is_file():
                    return f"❌ local file not found: {local_path}"

                # Check file size before upload
                file_size = Path(source_path).stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb > MAX_VIDEO_SIZE_MB:
                    return f"❌ File too large: {file_size_mb:.1f}MB exceeds limit of {MAX_VIDEO_SIZE_MB}MB"

                logger.info(f"Uploading file: {source_path} ({file_size_mb:.1f}MB)")

                # Upload file
                payload = Path(source_path).read_bytes()
                upload_res = await backend.aupload_files([(source_path, payload)])

                if upload_res and upload_res[0].error:
                    error_msg = upload_res[0].error
                    if "readonly" in error_msg.lower() or "exists" in error_msg.lower():
                        logger.warning(f"Upload error (file may already exist): {error_msg}")
                    return f"❌ upload failed: {error_msg}"

                target_container_path = upload_res[0].path
                logger.info(f"Successfully uploaded: {source_path} -> {target_container_path}")

                return self._build_state_command(
                    runtime,
                    message=f"✅ uploaded: {source_path} -> {target_container_path}",
                    files_update=self._build_file_state(
                        target_container_path,
                        status="uploaded",
                        local_path=str(source_path),
                    ),
                    last_operation={
                        "action": "video_upload",
                        "status": "success",
                        "local_path": str(source_path),
                        "container_path": target_container_path,
                        "file_size": file_size,
                        "file_size_mb": round(file_size_mb, 2),
                        "next_step": "Run FFmpeg tool with this container_path as input.",
                    },
                )
            except PermissionError as e:
                return f"❌ Permission denied: {source_path}. Check file permissions."
            except Exception as e:
                logger.error(f"Upload failed for {source_path}: {e}", exc_info=True)
                return f"❌ Upload failed: {str(e)}"

        return StructuredTool.from_function(
            name="video_upload",
            description=tool_description,
            coroutine=async_video_upload,
        )

    def _create_video_download_tool(self):
        tool_description = (
            "Download processed files from Docker container to host.\n"
            "Retrieve results after FFmpeg processing completes.\n"
        )

        async def async_video_download(
            runtime: ToolRuntime[None, FfmpegState],
            container_path: Annotated[str, "File path in container"],
        ) -> Command | str:
            backend = self._backend_for_runtime(runtime)

            # Validate path
            validated_path = self._validate_path(container_path)
            if not validated_path:
                return f"❌ Invalid container_path: {container_path}"

            try:
                download_res = await backend.adownload_files([validated_path])
                if not download_res:
                    return f"❌ download failed: no response for {container_path}"

                first = download_res[0]
                if first.error:
                    error_msg = first.error
                    if "not found" in error_msg.lower():
                        logger.warning(f"File not found in container: {validated_path}")
                    return f"❌ download failed: {error_msg}"

                local_path = first.path
                logger.info(f"Downloaded: {validated_path} -> {local_path}")

                # Update file status
                return self._build_state_command(
                    runtime,
                    message=f"✅ downloaded: {validated_path} -> {local_path}",
                    files_update=self._build_file_state(
                        validated_path,
                        status="downloaded",
                        local_path=local_path,
                    ),
                    last_operation={
                        "action": "video_download",
                        "status": "success",
                        "container_path": validated_path,
                        "local_path": local_path,
                    },
                )
            except Exception as e:
                logger.error(f"Download failed for {validated_path}: {e}", exc_info=True)
                return f"❌ Download failed: {str(e)}"

        return StructuredTool.from_function(
            name="video_download",
            description=tool_description,
            coroutine=async_video_download,
        )

    def _create_video_cut_tool(self):
        tool_description = (
            "Trim video by specifying start/end timestamps.\n"
            "Re-encodes to libx264 for quality.\n"
            "Parameters: input_path, output_path, start_sec, end_sec\n"
        )

        async def async_video_trim(
            runtime: ToolRuntime[None, FfmpegState],
            input_path: Annotated[str, "Input video path (must exist in container)"],
            output_path: Annotated[str, "Output video path"],
            start: Annotated[float, "Start time in seconds"],
            end: Annotated[float, "End time in seconds"],
        ) -> Command | str:
            backend = self._backend_for_runtime(runtime)

            # Validate paths
            validated_input = self._validate_path(input_path)
            validated_output = self._validate_path(output_path)
            if not validated_input or not validated_output:
                return f"❌ Invalid path: input={input_path}, output={output_path}"

            # Check start < end
            if start >= end:
                return f"❌ start ({start}s) must be less than end ({end}s)"

            # Generate unique output path if not provided
            if not Path(validated_output).name.startswith(("0", "1", "2")):
                base = Path(validated_input).stem
                validated_output = f"{base}_trim_{start:.0f}-{end:.0f}.mp4"

            # Build FFmpeg command
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_input)} "
                f"-ss {start} -to {end} "
                f"-c:v libx264 -c:a aac -preset medium -crf 23 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Running FFmpeg cut: {validated_input} [{start:.1f}s-{end:.1f}s] -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                # Extract line number from error
                lines = error.split("\n")
                error_preview = "\n".join(lines[:3])
                return f"❌ FFmpeg cut failed at line 3:\n{error_preview}"

            # Update state
            return self._tool_success(
                runtime,
                action="video_cut",
                output_path=validated_output,
                message=f"✅ trimmed video from {start:.1f}s → {end:.1f}s, output: {validated_output}",
                extra={"input_path": validated_input, "start": start, "end": end},
            )

        return StructuredTool.from_function(
            name="video_cut",
            description=tool_description,
            coroutine=async_video_trim,
        )
    # 🎬 video_snapshot
    def _create_video_snapshot_tool(self):
        tool_description = (
            "Extract a single frame as image (thumbnail).\n"
            "Parameters: input_path, time_sec, output_path\n"
            "Auto-generates .jpg if output_path has no extension.\n"
        )

        async def async_snapshot(
            runtime: ToolRuntime[None, FfmpegState],
            input_path: Annotated[str, "Absolute path to the input video file (must exist)"],
            time_sec: Annotated[float, "Timestamp in seconds (0 for first frame)"],
            output_path: Annotated[str, "Output image path (.jpg or .png)"],
        ) -> Command | str:
            backend = self._backend_for_runtime(runtime)

            # Validate paths
            validated_input = self._validate_path(input_path)
            validated_output = self._validate_path(output_path)
            if not validated_input or not validated_output:
                return f"❌ Invalid path: input={input_path}, output={output_path}"

            if time_sec < 0:
                return f"❌ time_sec must be >= 0, got {time_sec}"

            # Generate unique output path if not provided
            if ".jpg" not in validated_output.lower() and ".png" not in validated_output.lower():
                validated_output = f"{Path(validated_input).stem}_frame_{time_sec:.0f}.jpg"

            # Check if file exists
            exists, error = await self._ensure_file_exists(backend, validated_input, runtime)
            if not exists:
                if error:
                    return error
                return f"❌ file not found: {validated_input}"

            # Build FFmpeg command (fast extraction with libx264)
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_input)} "
                f"-ss {time_sec} -vframes 1 "
                f"-c:v libx264 -pix_fmt yuv420p "
                f"-preset medium -crf 28 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Creating snapshot at {time_sec}s: {validated_input} -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:4])
                return f"❌ snapshot failed:\n{error_preview}"

            return self._build_state_command(
                runtime,
                message=f"✅ snapshot created at {time_sec:.1f}s → {validated_output}",
                files_update=self._build_file_state(
                    validated_output,
                    type_="snapshot",
                ),
                last_operation={
                    "action": "video_snapshot",
                    "status": "success",
                    "input_path": validated_input,
                    "output_path": validated_output,
                    "time_sec": time_sec,
                    "next_step": f"Call video_download with container_path={validated_output}",
                },
            )

        return StructuredTool.from_function(
            name="video_snapshot",
            description=tool_description,
            coroutine=async_snapshot,
        )
    
    def _create_video_concat_tool(self):
        tool_description = (
            "Concatenate multiple videos using copy stream.\n"
            "Videos must have identical codec/resolution/fps.\n"
            "Parameters: input_paths (list), output_path\n"
        )

        async def async_video_concat(
            runtime: ToolRuntime[None, FfmpegState],
            input_paths: Annotated[list[str], "List of absolute paths to input video files in order. Must exist."],
            output_path: Annotated[str, "Absolute path for the merged output video."],
        ) -> Command | str:
            backend = self._backend_for_runtime(runtime)

            # Validate paths
            validated_output = self._validate_path(output_path)
            if not validated_output:
                return f"❌ Invalid output_path: {output_path}"

            if not isinstance(input_paths, list) or len(input_paths) < 2:
                return f"❌ input_paths must be a list with at least 2 files, got {len(input_paths) if input_paths else 0}"

            # Validate all input paths and check existence
            validated_inputs = []
            for idx, path in enumerate(input_paths):
                validated = self._validate_path(path)
                if not validated:
                    return f"❌ Invalid input_path at index {idx}: {path}"

                exists, error = await self._ensure_file_exists(backend, validated, runtime)
                if not exists:
                    if error:
                        return error
                    return f"❌ file not found at index {idx}: {validated}"

                validated_inputs.append(validated)

            # Build concat list file
            list_file = "/tmp/concat_list_$(date +%s).txt"
            list_content = "\n".join([f"file '{p}'" for p in validated_inputs])

            try:
                await backend.awrite(list_file, list_content)
            except Exception as e:
                logger.error(f"Failed to write concat list: {e}")
                return f"❌ Failed to create concat list: {str(e)}"

            # Build FFmpeg command
            cmd = f"ffmpeg -y -f concat -safe 0 -i {list_file} -c copy -preset medium {shlex.quote(validated_output)}"

            logger.info(f"Concatenating {len(validated_inputs)} videos: {validated_inputs} -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:5])
                # Check for codec mismatch
                if "different size" in error.lower() or "different codecs" in error.lower():
                    return f"❌ Video format mismatch:\n{error_preview}\n\nTip: Re-encode all videos to same codec/resolution first."
                return f"❌ concat failed:\n{error_preview}"

            # Calculate total duration (approximate)
            total_files = len(validated_inputs)
            return self._tool_success(
                runtime,
                action="video_concat",
                output_path=validated_output,
                message=f"✅ concatenated {total_files} videos, output: {validated_output}",
                extra={"input_paths": validated_inputs, "count": total_files},
            )

        return StructuredTool.from_function(
            name="video_concat",
            description=tool_description,
            coroutine=async_video_concat,
        )
    
    def _create_video_resize_tool(self):
        tool_description = (
            "Resize video to exact width x height.\n"
            "Re-encodes to libx264. Parameters: input_path, width, height, output_path\n"
        )

        async def async_resize(
            runtime: ToolRuntime[None, FfmpegState],
            input_path: Annotated[str, "Input video path (must exist)"],
            output_path: Annotated[str, "Output video path"],
            width: Annotated[int, "Target width in pixels (must be > 0)"],
            height: Annotated[int, "Target height in pixels (must be > 0)"],
        ):
            backend = self._backend_for_runtime(runtime)

            # Validate inputs
            validated_input = self._validate_path(input_path)
            validated_output = self._validate_path(output_path)
            if not validated_input or not validated_output:
                return f"❌ Invalid path: input={input_path}, output={output_path}"

            if width <= 0 or height <= 0:
                return f"❌ width={width} and height={height} must be > 0"

            # Generate unique output path
            if not validated_output:
                base = Path(validated_input).stem
                validated_output = f"{base}_resize_{width}x{height}.mp4"

            # Build FFmpeg command
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_input)} "
                f"-vf scale={width}:{height} "
                f"-c:v libx264 -preset medium -crf 23 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Resizing video: {validated_input} {width}x{height} -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:4])
                return f"❌ resize failed:\n{error_preview}"

            return self._tool_success(
                runtime,
                action="video_resize",
                output_path=validated_output,
                message=f"✅ resized to {width}x{height}, output: {validated_output}",
                extra={"input_path": validated_input, "size": {"width": width, "height": height}},
            )

        return StructuredTool.from_function(
            name="video_resize",
            description=tool_description,
            coroutine=async_resize,
        )
    
    def _create_video_crop_tool(self):
        tool_description = (
            "Crop video to specified width x height region.\n"
            "Crops from top-left at (x, y). Parameters: input_path, output_path, x, y, width, height\n"
        )

        async def async_crop(
            runtime: ToolRuntime[None, FfmpegState],
            input_path: Annotated[str, "Input video (must exist)"],
            output_path: Annotated[str, "Output video"],
            x: Annotated[int, "Top-left x coordinate (0 for left edge)"],
            y: Annotated[int, "Top-left y coordinate (0 for top edge)"],
            width: Annotated[int, "Crop width (positive)"],
            height: Annotated[int, "Crop height (positive)"],
        ):
            backend = self._backend_for_runtime(runtime)

            # Validate inputs
            validated_input = self._validate_path(input_path)
            validated_output = self._validate_path(output_path)
            if not validated_input or not validated_output:
                return f"❌ Invalid path: input={input_path}, output={output_path}"

            if width <= 0 or height <= 0:
                return f"❌ width={width} and height={height} must be > 0"

            # Generate unique output path
            validated_output = validated_output or f"{Path(validated_input).stem}_crop_{width}x{height}.mp4"

            # Build FFmpeg crop command
            # Note: crop filter format is crop=w:h:x:y
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_input)} "
                f"-vf crop={width}:{height}:{x}:{y} "
                f"-c:v libx264 -preset medium -crf 23 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Cropping video: {validated_input} [{width}x{height} at {x},{y}] -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:4])
                return f"❌ crop failed:\n{error_preview}"

            return self._tool_success(
                runtime,
                action="video_crop",
                output_path=validated_output,
                message=f"✅ cropped to {width}x{height} at ({x},{y}), output: {validated_output}",
                extra={"input_path": validated_input, "crop": {"x": x, "y": y, "width": width, "height": height}},
            )

        return StructuredTool.from_function(
            name="video_crop",
            description=tool_description,
            coroutine=async_crop,
        )
    
    def _create_video_add_text_tool(self):

        tool_description = (
            "Add text overlay to video.\n"
            "Creates static text at (x, y) position. Escapes single quotes in text.\n"
            "Parameters: input_path, output_path, text, x, y\n"
        )

        async def async_text(
            runtime: ToolRuntime[None, FfmpegState],
            input_path: Annotated[str, "Input video (must exist)"],
            output_path: Annotated[str, "Output video"],
            text: Annotated[str, "Text content (escape single quotes using '')"],
            x: Annotated[int, "X position (0 = left edge)"],
            y: Annotated[int, "Y position (0 = top edge)"],
        ):
            backend = self._backend_for_runtime(runtime)

            # Validate inputs
            validated_input = self._validate_path(input_path)
            validated_output = self._validate_path(output_path)
            if not validated_input or not validated_output:
                return f"❌ Invalid path: input={input_path}, output={output_path}"

            # Escape single quotes in text
            escaped_text = text.replace("'", r"\'")

            # Generate unique output path
            validated_output = validated_output or f"{Path(validated_input).stem}_text.mp4"

            # Build FFmpeg drawtext command
            # Note: y=0 puts text at top, increase for lower position
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_input)} "
                f"-vf drawtext=text='{escaped_text}':x={x}:y={y}:fontsize=24:fontcolor=white@black "
                f"-c:v libx264 -preset medium -crf 23 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Adding text to video: {validated_input} [{text[:30]}...] -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:4])
                return f"❌ text overlay failed:\n{error_preview}"

            return self._tool_success(
                runtime,
                action="video_add_text",
                output_path=validated_output,
                message=f"✅ text overlay added: [{text[:30]}...] output: {validated_output}",
                extra={"input_path": validated_input, "text": text},
            )

        return StructuredTool.from_function(
            name="video_add_text",
            description=tool_description,
            coroutine=async_text,
        )
    
    def _create_video_overlay_tool(self):
        tool_description = (
            "Overlay image watermark on video.\n"
            "Parameters: video_path, image_path, output_path\n"
            "Auto-generates .mp4 output with watermark.\n"
        )

        async def async_overlay(
            runtime: ToolRuntime[None, FfmpegState],
            video_path: Annotated[str, "Input video (must exist)"],
            image_path: Annotated[str, "Overlay image (must exist)"],
            output_path: Annotated[str, "Output video"],
        ):
            backend = self._backend_for_runtime(runtime)

            # Validate inputs
            validated_video = self._validate_path(video_path)
            validated_image = self._validate_path(image_path)
            validated_output = self._validate_path(output_path)

            if not validated_video or not validated_image:
                return f"❌ Invalid path: video={video_path}, image={image_path}"

            # Check if both files exist
            exists_video, err = await self._ensure_file_exists(backend, validated_video, runtime)
            exists_image, err = await self._ensure_file_exists(backend, validated_image, runtime)

            if not exists_video or not exists_image:
                if err:
                    return err
                missing = ""
                if not exists_video:
                    missing = f"video not found: {validated_video}\n"
                if not exists_image:
                    missing += f"image not found: {validated_image}"
                return f"❌ missing files:\n{missing}"

            # Generate unique output path
            validated_output = validated_output or f"{Path(validated_video).stem}_watermark.mp4"

            # Build FFmpeg overlay command
            # overlay=offset_x:offset_y (image must be smaller or equal to video)
            cmd = (
                f"ffmpeg -y -i {shlex.quote(validated_video)} -i {shlex.quote(validated_image)} "
                f"-filter_complex overlay=10:10 "
                f"-c:v libx264 -preset medium -crf 23 "
                f"{shlex.quote(validated_output)}"
            )

            logger.info(f"Overlaying image on video: {validated_video} + {validated_image} -> {validated_output}")

            error = await self._run_ffmpeg(backend, cmd)
            if error:
                lines = error.split("\n")
                error_preview = "\n".join(lines[:5])
                # Check for resolution mismatch
                if "size of first" in error.lower():
                    return f"❌ overlay failed: video and image size mismatch:\n{error_preview}\n\nTip: Resize image to match video resolution first."
                return f"❌ overlay failed:\n{error_preview}"

            return self._tool_success(
                runtime,
                action="video_watermark",
                output_path=validated_output,
                message=f"✅ watermark added: {validated_image} on {validated_video}, output: {validated_output}",
                extra={"video_path": validated_video, "image_path": validated_image},
            )

        return StructuredTool.from_function(
            name="video_watermark",
            description=tool_description,
            coroutine=async_overlay,
        )

    # =============== System Hooks ===============

    def wrap_model_call(self, request: ModelRequest, handler):
        """Wrap model call to add FFmpeg tools to the request."""
        # Get tool names
        tool_names = [getattr(t, "name", None) or t.get("name") for t in self.tools]

        # System prompt
        default_prompt = (
            f"You can use the following FFmpeg video processing tools:\n\n"
            f"1. {tool_names[0]} (video_upload) - Upload a local file to container\n"
            f"2. {tool_names[1]} (video_download) - Download processed file from container\n"
            f"3. {tool_names[2]} (video_cut) - Trim a video (start → end)\n"
            f"4. {tool_names[3]} (video_snapshot) - Extract a frame as image\n"
            f"5. {tool_names[4]} (video_concat) - Concatenate multiple videos\n"
            f"6. {tool_names[5]} (video_resize) - Resize video resolution\n"
            f"7. {tool_names[6]} (video_crop) - Crop video region\n"
            f"8. {tool_names[7]} (video_add_text) - Add text overlay\n"
            f"9. {tool_names[8]} (video_watermark) - Overlay image watermark\n\n"
            f"## Usage Guidelines\n\n"
            f"- Always upload files using {tool_names[0]} before processing\n"
            f"- Use absolute paths for all file operations\n"
            f"- FFmpeg operations may take time depending on file size\n"
            f"- Download results using {tool_names[1]} after processing\n\n"
            f"{get_middleware_prompt('ffmpeg')}"
        )
        system_prompt = self._custom_system_prompt or default_prompt
        new_system_message = append_to_system_message(request.system_message, system_prompt)
        request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Async version of wrap_model_call."""
        modified_request = self.wrap_model_call(request, lambda r: r)
        return await handler(modified_request)
