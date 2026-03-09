from agi.agents_runtime.system_tools import execute_shell


def test_execute_shell_runs_command():
    result = execute_shell("echo hello")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
