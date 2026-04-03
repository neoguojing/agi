import atexit
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

logger = logging.getLogger(__name__)


@dataclass
class _DockerSession:
    user_id: str
    container_id: str
    workspace_host: str
    created_at: float
    last_active_at: float


class _UserScopedDockerSandbox(BaseSandbox):
    """A user-scoped sandbox view backed by DockerSandbox session manager."""

    def __init__(self, manager: "DockerSandbox", user_id: str):
        self._manager = manager
        self._user_id = user_id

    @property
    def id(self) -> str:
        return self._manager.get_container_id(self._user_id)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._manager.execute_for_user(self._user_id, command, timeout=timeout)

    def upload_files(self, files: List[tuple[str, bytes]]) -> List[FileUploadResponse]:
        return self._manager.upload_files_for_user(self._user_id, files)

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        return self._manager.download_files_for_user(self._user_id, paths)


class DockerSandbox(BaseSandbox):
    """Multi-tenant Docker sandbox manager.

    - One container per user_id.
    - Lazy session creation (no container starts at init).
    - Idle-session TTL cleanup to avoid leaked containers.
    """

    def __init__(
        self,
        image: str = "eswardudi/python-ffmpeg:3.13.3",
        workspace_root: str = "/tmp/docker_sandbox_workspaces",
        session_ttl: int = 1800,
        cleanup_interval: int = 60,
    ):
        self._image = image
        self._workspace_container = "/workspace"
        self._workspace_root = Path(workspace_root)
        self._workspace_root.mkdir(parents=True, exist_ok=True)

        self._session_ttl = max(60, int(session_ttl))
        self._cleanup_interval = max(10, int(cleanup_interval))

        self._sessions: Dict[str, _DockerSession] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        atexit.register(self.close)

    @property
    def id(self) -> str:
        raise RuntimeError("DockerSandbox id is user-scoped. Use for_user(user_id).id instead.")

    def for_user(self, user_id: str | None) -> BaseSandbox:
        resolved_user_id = self._normalize_user_id(user_id)
        return _UserScopedDockerSandbox(self, resolved_user_id)

    def get_container_id(self, user_id: str | None) -> str:
        session = self._ensure_session(self._normalize_user_id(user_id))
        return session.container_id

    def execute_for_user(self, user_id: str, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        session = self._ensure_session(self._normalize_user_id(user_id))
        docker_cmd = ["docker", "exec", session.container_id, "bash", "-c", command]
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            self._touch_session(session.user_id)
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
                truncated=False,
            )
        except subprocess.TimeoutExpired:
            self._touch_session(session.user_id)
            return ExecuteResponse(output="Timeout", exit_code=124, truncated=True)

    def upload_files_for_user(self, user_id: str, files: List[tuple[str, bytes]]) -> List[FileUploadResponse]:
        session = self._ensure_session(self._normalize_user_id(user_id))
        responses: List[FileUploadResponse] = []

        for host_path, content in files:
            host_path = self._to_host_path(session, host_path)
            container_path = self.to_container_path(session, host_path)
            host_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                host_path.write_bytes(content)
                responses.append(FileUploadResponse(path=str(container_path)))
            except Exception as e:
                responses.append(FileUploadResponse(path=str(container_path), error=str(e)))

        self._touch_session(session.user_id)
        return responses

    def download_files_for_user(self, user_id: str, paths: List[str]) -> List[FileDownloadResponse]:
        session = self._ensure_session(self._normalize_user_id(user_id))
        responses: List[FileDownloadResponse] = []

        for container_path in paths:
            host_path = self._to_host_path(session, container_path)
            try:
                if not host_path.exists():
                    raise FileNotFoundError(f"{host_path} not found")
                responses.append(FileDownloadResponse(path=str(host_path), content=None))
            except Exception as e:
                responses.append(FileDownloadResponse(path=str(host_path), content=None, error=str(e)))

        self._touch_session(session.user_id)
        return responses

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise RuntimeError("DockerSandbox is user-scoped. Use for_user(user_id).execute(...)")

    def upload_files(self, files: List[tuple[str, bytes]]) -> List[FileUploadResponse]:
        raise RuntimeError("DockerSandbox is user-scoped. Use for_user(user_id).upload_files(...)")

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        raise RuntimeError("DockerSandbox is user-scoped. Use for_user(user_id).download_files(...)")

    def close_user_session(self, user_id: str | None) -> None:
        resolved_user_id = self._normalize_user_id(user_id)
        with self._lock:
            session = self._sessions.pop(resolved_user_id, None)
        if not session:
            return

        self._stop_container(session.container_id)
        self._remove_workspace(session.workspace_host)
        logger.info("Docker sandbox session closed: user_id=%s container=%s", resolved_user_id, session.container_id)

    def close(self) -> None:
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            self._stop_container(session.container_id)
            self._remove_workspace(session.workspace_host)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # internal helpers
    def _normalize_user_id(self, user_id: str | None) -> str:
        value = (user_id or "").strip()
        if not value:
            raise ValueError("user_id is required for DockerSandbox")
        return value

    def _ensure_session(self, user_id: str) -> _DockerSession:
        with self._lock:
            existing = self._sessions.get(user_id)
            if existing:
                if not self._is_container_running(existing.container_id):
                    logger.warning(
                        "Docker sandbox container not running, recreating: user_id=%s container=%s",
                        user_id,
                        existing.container_id,
                    )
                    self._sessions.pop(user_id, None)
                    self._remove_workspace(existing.workspace_host)
                else:
                    existing.last_active_at = time.time()
                    return existing

            session = self._start_session(user_id)
            self._sessions[user_id] = session
            return session

    def _touch_session(self, user_id: str) -> None:
        with self._lock:
            session = self._sessions.get(user_id)
            if session:
                session.last_active_at = time.time()

    def _start_session(self, user_id: str) -> _DockerSession:
        safe_user = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_id)[:32] or "default"
        container_id = f"docker_sandbox_{safe_user}"
        workspace_host = self._workspace_root / f"{container_id}_workspace"
        workspace_host.mkdir(parents=True, exist_ok=True)

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_id,
            "-v",
            f"{workspace_host}:{self._workspace_container}",
            "-w",
            self._workspace_container,
            "--rm",
            self._image,
            "tail",
            "-f",
            "/dev/null",
        ]
        subprocess.run(cmd, check=True)
        now = time.time()
        logger.info("Docker sandbox session started: user_id=%s container=%s", user_id, container_id)
        return _DockerSession(
            user_id=user_id,
            container_id=container_id,
            workspace_host=str(workspace_host),
            created_at=now,
            last_active_at=now,
        )

    def _stop_container(self, container_id: str) -> None:
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def _is_container_running(self, container_id: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _remove_workspace(self, workspace_host: str) -> None:
        path = Path(workspace_host)
        if path.exists() and str(path).startswith(str(self._workspace_root)):
            shutil.rmtree(path, ignore_errors=True)

    def _to_host_path(self, session: _DockerSession, path: str) -> Path:
        normalized = str(path or "").strip()
        if not normalized:
            raise ValueError("container path cannot be empty")

        # 只保留文件名（去掉所有路径前缀）
        filename = Path(normalized).name
        if not filename:
            raise ValueError(f"invalid path, no filename: {path}")

        # 拼接到宿主机 workspace
        return (Path(session.workspace_host) / filename).resolve()
    
    def to_container_path(self, session: _DockerSession, host_path: Path) -> str:
        host_path = Path(host_path).resolve()
        workspace_host = Path(session.workspace_host).resolve()

        # 校验必须在 workspace 内
        try:
            host_path.relative_to(workspace_host)
        except ValueError:
            raise ValueError(f"path {host_path} is outside workspace {workspace_host}")

        # 只保留文件名
        filename = host_path.name
        if not filename:
            raise ValueError(f"invalid path, no filename: {host_path}")

        return f"{self._workspace_container}/{filename}"

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(self._cleanup_interval):
            self._cleanup_idle_sessions()

    def _cleanup_idle_sessions(self) -> None:
        now = time.time()
        expired: list[_DockerSession] = []

        with self._lock:
            for user_id, session in list(self._sessions.items()):
                if now - session.last_active_at > self._session_ttl:
                    expired.append(session)
                    self._sessions.pop(user_id, None)

        for session in expired:
            logger.info(
                "Docker sandbox session expired: user_id=%s container=%s idle=%ss",
                session.user_id,
                session.container_id,
                int(now - session.last_active_at),
            )
            self._stop_container(session.container_id)
            self._remove_workspace(session.workspace_host)
