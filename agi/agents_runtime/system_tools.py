from __future__ import annotations

import os
import subprocess
from typing import Any

from .sandbox_runtime import DockerSandboxManager


def execute_shell(command: str, *, timeout: int = 120, cwd: str | None = None) -> dict[str, Any]:
    """在宿主机执行 shell 命令（仅开发环境使用）。"""
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


# global sandbox manager (sandbox-as-tool pattern)
_SANDBOX_MANAGER = DockerSandboxManager(execute_shell)


def run_in_sandbox(command: str, *, image: str = "python:3.11", timeout: int = 300) -> dict[str, Any]:
    """通过 docker 启动一次性隔离环境执行命令（兼容旧接口）。"""
    docker_cmd = (
        f"docker run --rm --network=none {image} "
        f"/bin/sh -lc {subprocess.list2cmdline([command])}"
    )
    return execute_shell(docker_cmd, timeout=timeout)


def sandbox_execute(command: str, *, thread_id: str = "default", timeout: int = 300) -> dict[str, Any]:
    return _SANDBOX_MANAGER.execute(thread_id, command, timeout=timeout)


def sandbox_upload_file(path: str, content_base64: str, *, thread_id: str = "default") -> dict[str, Any]:
    import base64

    content = base64.b64decode(content_base64.encode("utf-8"))
    result = _SANDBOX_MANAGER.upload_files(thread_id, [(path, content)])
    return result[0]


def sandbox_download_file(path: str, *, thread_id: str = "default") -> dict[str, Any]:
    result = _SANDBOX_MANAGER.download_files(thread_id, [path])
    item = result[0]
    if item["content"] is not None:
        import base64

        item["content_base64"] = base64.b64encode(item["content"]).decode("utf-8")
        del item["content"]
    return item


def sandbox_shutdown(*, thread_id: str = "default") -> dict[str, Any]:
    return _SANDBOX_MANAGER.shutdown(thread_id)


def docker_build_toolchain(context_dir: str, tag: str, *, dockerfile: str = "Dockerfile", timeout: int = 1800) -> dict[str, Any]:
    if not os.path.isdir(context_dir):
        return {"error": f"context_dir does not exist: {context_dir}"}
    cmd = f"docker build -f {dockerfile} -t {tag} {context_dir}"
    return execute_shell(cmd, timeout=timeout)
