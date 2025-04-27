from agi.tasks.define import State
from agi.tasks.task_factory import (
    TaskFactory,
    TASK_DOC_CHAT,
    )
from langchain_core.runnables import  RunnableConfig
from langgraph.types import StreamWriter

# NODE
# 文档对话
def doc_chat_node(state: State,config: RunnableConfig,writer: StreamWriter):
    chain = TaskFactory.create_task(TASK_DOC_CHAT)
    if state["citations"]:
        writer({"citations":state["citations"],"docs":state["docs"]})
    return chain.invoke(state,config=config)

