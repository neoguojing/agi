from agi.agents_runtime.sandbox_runtime import DockerSandboxManager


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        if command.startswith("docker run -d"):
            return {"exit_code": 0, "stdout": "cid-123\n", "stderr": ""}
        if command.startswith("docker rm -f"):
            return {"exit_code": 0, "stdout": "removed", "stderr": ""}
        if "print(base64.b64encode" in command:
            return {"exit_code": 0, "stdout": "aGVsbG8=\n", "stderr": ""}
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}


def test_docker_sandbox_manager_lifecycle_and_file_io():
    runner = FakeRunner()
    mgr = DockerSandboxManager(runner)

    sess = mgr.get_or_create("t1")
    assert sess.container_id == "cid-123"

    exec_out = mgr.execute("t1", "echo hi")
    assert exec_out["exit_code"] == 0

    up = mgr.upload_files("t1", [("/tmp/a.txt", b"hello")])
    assert up[0]["path"] == "/tmp/a.txt"

    down = mgr.download_files("t1", ["/tmp/a.txt"])
    assert down[0]["content"] == b"hello"

    stop = mgr.shutdown("t1")
    assert stop["exit_code"] == 0
