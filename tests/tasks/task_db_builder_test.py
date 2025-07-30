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
    assert ret is not None, "ainvoke 返回值为空"
    colects = collection_manager.list_collections(tenant="dbtest")
    assert len(colects) >= 2, "应包含至少两个集合"
    indexs = collection_manager.get_documents(collection_name="index",tenant="dbtest")
    assert len(indexs) >= 1, "应包含至少1个索引"
    docs = collection_manager.get_documents(collection_name="dbtest",tenant="dbtest")
    assert len(docs) >= 2, "应包含至少2个文档"



@pytest.mark.asyncio
async def test_summary_rag():
    config={"configurable": {"conversation_id": "2","thread_id": "dbtest"}}
    input = State(
        messages=[HumanMessage(content="总结该文档")],
        collection_names = ["dbtest"],
        user_id = "dbtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    assert isinstance(ret,dict)
    assert len(ret.get("docs")) >= 1, "应包含至少1个文档"

@pytest.mark.asyncio
async def test_rag():
    config={"configurable": {"conversation_id": "3","thread_id": "dbtest"}}
    input = State(
        messages=[HumanMessage(content="NTP3000Plus")],
        collection_names = ["dbtest"],
        user_id = "dbtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    print(ret)
    assert isinstance(ret,dict)
    assert len(ret.get("docs")) >= 1, "应包含至少1个文档"
    assert len(ret.get("index_search_result")) >= 1, "应包含至少1个文档"

        