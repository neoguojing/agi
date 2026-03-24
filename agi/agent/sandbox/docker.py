import subprocess
import uuid
import os
import shutil
from typing import List
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileUploadResponse,
    FileDownloadResponse,
)
from pathlib import Path
from deepagents.backends.sandbox import BaseSandbox  # 你之前的基类

class DockerSandbox(BaseSandbox):
    """Stateful Docker sandbox implementation."""

    def __init__(self, image: str = "python:3.11-slim", workspace: str | None = None):
        """
        Args:
            image: Docker image to use
            workspace: Optional host path to mount for persistence
        """
        self._image = image
        self._id = f"docker_sandbox_{uuid.uuid4().hex[:8]}"
        self._workspace_host = workspace or f"/tmp/{self._id}_workspace"
        self._workspace_container = "/workspace"
        os.makedirs(self._workspace_host, exist_ok=True)
        self._container_running = False
        self._start_container()

    @property
    def id(self) -> str:
        return self._id

    def _start_container(self):
        """Start the Docker container if not running."""
        cmd = [
            "docker", "run", "-d",
            "--name", self._id,
            "-v", f"{self._workspace_host}:{self._workspace_container}",
            "-w", self._workspace_container,
            "--rm",  # remove on stop
            self._image,
            "tail", "-f", "/dev/null"  # keep container alive
        ]
        subprocess.run(cmd, check=True)
        self._container_running = True

    def _exec_docker(self, command: str, timeout: int | None = None) -> ExecuteResponse:
        """Execute command inside the container."""
        docker_cmd = ["docker", "exec", self._id, "bash", "-c", command]
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
                truncated=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(output="Timeout", exit_code=124, truncated=True)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._exec_docker(command, timeout=timeout)

    def upload_files(self, files: List[tuple[str, bytes]]) -> List[FileUploadResponse]:
        responses = []
        for file_name, content in files:
            host_path = Path(self._workspace_host) / file_name
            host_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                host_path.write_bytes(content)
                responses.append(FileUploadResponse(path=str(file_name)))
            except Exception as e:
                responses.append(FileUploadResponse(path=str(file_name), error=str(e)))
        return responses

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """返回文件完整宿主机路径，不加载内容到内存"""
        responses = []
        workspace = Path(self._workspace_host)  # 容器挂载到宿主机的目录
        for file_name in paths:
            host_path = (workspace / file_name).resolve()  # 获取绝对路径
            try:
                if not host_path.exists():
                    raise FileNotFoundError(f"{host_path} not found")
                # path = 完整路径, content = None
                responses.append(FileDownloadResponse(path=str(host_path), content=None))
            except Exception as e:
                responses.append(FileDownloadResponse(path=str(host_path), content=None, error=str(e)))
        return responses

    def close(self):
        """Stop and remove container."""
        if self._container_running:
            subprocess.run(["docker", "stop", self._id], capture_output=True)
            self._container_running = False
        # Optional: cleanup workspace if not mounted externally
        if self._workspace_host.startswith("/tmp/") and Path(self._workspace_host).exists():
            shutil.rmtree(self._workspace_host)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()