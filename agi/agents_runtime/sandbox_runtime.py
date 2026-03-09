from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


Runner = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class SandboxSession:
    thread_id: str
    container_id: str
    image: str
    created_at: str
    expires_at: str | None = None


class DockerSandboxManager:
    """基于 Docker 的轻量 sandbox 会话管理（sandbox-as-tool 模式）。"""

    def __init__(self, runner: Runner, *, image: str = "python:3.11", network_disabled: bool = True, ttl_seconds: int = 3600) -> None:
        self.runner = runner
        self.image = image
        self.network_disabled = network_disabled
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, SandboxSession] = {}

    def get_or_create(self, thread_id: str) -> SandboxSession:
        existing = self._sessions.get(thread_id)
        if existing:
            return existing

        net_opt = "--network=none" if self.network_disabled else ""
        result = self.runner(f"docker run -d {net_opt} {self.image} sleep infinity", timeout=120)
        if result.get("exit_code") != 0:
            raise RuntimeError(result.get("stderr", "failed to start sandbox"))

        container_id = (result.get("stdout") or "").strip()
        now = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(now.timestamp() + self.ttl_seconds, tz=timezone.utc).isoformat()
        sess = SandboxSession(
            thread_id=thread_id,
            container_id=container_id,
            image=self.image,
            created_at=now.isoformat(),
            expires_at=expires_at,
        )
        self._sessions[thread_id] = sess
        return sess

    def execute(self, thread_id: str, command: str, *, timeout: int = 300) -> dict[str, Any]:
        sess = self.get_or_create(thread_id)
        return self.runner(f"docker exec {sess.container_id} /bin/sh -lc {self._quote(command)}", timeout=timeout)

    def upload_files(self, thread_id: str, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        sess = self.get_or_create(thread_id)
        out: list[dict[str, Any]] = []
        for path, content in files:
            encoded = base64.b64encode(content).decode("utf-8")
            cmd = (
                "python - <<'PY'\n"
                "import base64, pathlib\n"
                f"p = pathlib.Path({path!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                f"p.write_bytes(base64.b64decode({encoded!r}))\n"
                "print('ok')\n"
                "PY"
            )
            res = self.runner(f"docker exec {sess.container_id} /bin/sh -lc {self._quote(cmd)}", timeout=120)
            out.append({"path": path, **res})
        return out

    def download_files(self, thread_id: str, paths: list[str]) -> list[dict[str, Any]]:
        sess = self.get_or_create(thread_id)
        out: list[dict[str, Any]] = []
        for path in paths:
            cmd = (
                "python - <<'PY'\n"
                "import base64, pathlib, sys\n"
                f"p = pathlib.Path({path!r})\n"
                "if not p.exists():\n"
                "  print('ERR:NOT_FOUND')\n"
                "  sys.exit(2)\n"
                "print(base64.b64encode(p.read_bytes()).decode('utf-8'))\n"
                "PY"
            )
            res = self.runner(f"docker exec {sess.container_id} /bin/sh -lc {self._quote(cmd)}", timeout=120)
            content = None
            error = None
            if res.get("exit_code") == 0:
                try:
                    content = base64.b64decode((res.get("stdout") or "").strip())
                except Exception:  # noqa: BLE001
                    error = "decode_failed"
            else:
                error = res.get("stderr") or res.get("stdout")
            out.append({"path": path, "content": content, "error": error})
        return out

    def shutdown(self, thread_id: str) -> dict[str, Any]:
        sess = self._sessions.get(thread_id)
        if not sess:
            return {"exit_code": 0, "stdout": "already_stopped", "stderr": ""}
        res = self.runner(f"docker rm -f {sess.container_id}", timeout=60)
        self._sessions.pop(thread_id, None)
        return res

    @staticmethod
    def _quote(s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"
