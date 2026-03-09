from __future__ import annotations

import os
import subprocess
from typing import Any


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


def run_in_sandbox(command: str, *, image: str = "python:3.11", timeout: int = 300) -> dict[str, Any]:
    """通过 docker 启动一次性隔离环境执行命令。"""
    docker_cmd = (
        f"docker run --rm --network=none {image} "
        f"/bin/sh -lc {subprocess.list2cmdline([command])}"
    )
    return execute_shell(docker_cmd, timeout=timeout)


def docker_build_toolchain(context_dir: str, tag: str, *, dockerfile: str = "Dockerfile", timeout: int = 1800) -> dict[str, Any]:
    if not os.path.isdir(context_dir):
        return {"error": f"context_dir does not exist: {context_dir}"}
    cmd = f"docker build -f {dockerfile} -t {tag} {context_dir}"
    return execute_shell(cmd, timeout=timeout)
