from agi.tasks.db_builder import db_graph
from agi.tasks.rag_web import rag_graph,collection_manager
from agi.tasks.define import State
import asyncio
import pytest
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage 

@pytest.mark.asyncio
async def test_db_graph():
    config={"configurable": {"conversation_id": "1",
                                "thread_id": "dbtest"}}
    
    state = State()
    state['user_id'] = "dbtest"
    state['collection_name'] = "dbtest"
    state['file_path'] = "tests/test.pdf"
    ret = await db_graph.ainvoke(state,config=config)
    print(ret)
    colects = collection_manager.list_collections(tenant="dbtest")
    print(colects)
    docs = collection_manager.get_documents(collection_name="dbtest",tenant="dbtest")
    print(ret)



@pytest.mark.asyncio
async def test_rag():
    config={"configurable": {"conversation_id": "2","thread_id": "dbtest"}}
    input = State(
        messages=[HumanMessage(content="总结该文档")],
        collection_names = ["dbtest"],
        user_id = "dbtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    print(ret)
        