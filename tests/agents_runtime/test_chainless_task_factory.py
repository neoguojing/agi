import importlib

module = importlib.import_module("agi.tasks.runtime.task_factory")
TaskFactory = module.TaskFactory
TASK_RAG = module.TASK_RAG
TASK_WEB_SEARCH = module.TASK_WEB_SEARCH
TASK_DOC_CHAT = module.TASK_DOC_CHAT


def test_task_factory_no_legacy_chain_import():
    source = open(module.__file__, "r", encoding="utf-8").read()
    assert "agi.tasks.chat.chains" not in source


def test_tool_runnable_tasks_are_available():
    assert TaskFactory.create_task(TASK_RAG) is not None
    assert TaskFactory.create_task(TASK_WEB_SEARCH) is not None
    assert TaskFactory.create_task(TASK_DOC_CHAT) is not None
