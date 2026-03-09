from agi.agents_runtime.harness import TodoManager
from agi.agents_runtime.types import AgentRuntimeConfig


def test_todo_manager_write_update_list():
    mgr = TodoManager()
    mgr.write_todos(
        "s1",
        [
            {"id": "1", "content": "collect", "status": "completed"},
            {"id": "2", "content": "answer", "status": "in_progress"},
        ],
    )
    assert len(mgr.list_todos("s1")) == 2

    changed = mgr.update_status("s1", "2", "completed")
    assert changed is True
    assert mgr.list_todos("s1")[1].status == "completed"


def test_runtime_config_harness_fields():
    config = AgentRuntimeConfig()
    assert config.memory_files == []
    assert config.interrupt_on == {}
