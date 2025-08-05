from agi.tasks.db_builder import db_graph
from agi.tasks.rag_web import rag_graph,collection_manager
from agi.tasks.define import State
from agi.utils.nlp import TextProcessor
import asyncio
import pytest
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage 

@pytest.mark.asyncio
async def test_db_graph():
    config={"configurable": {"conversation_id": "1",
                                "thread_id": "ragtest"}}
    
    state = State()
    state['user_id'] = "ragtest"
    state['collection_name'] = "ragtest"
    state['file_path'] = "tests/2412.19437v1.pdf"
    ret = await db_graph.ainvoke(state,config=config)
    assert ret is not None, "ainvoke 返回值为空"
    colects = collection_manager.list_collections(tenant="ragtest")
    assert len(colects) >= 2, "应包含至少两个集合"
    indexs = collection_manager.get_documents(collection_name="index",tenant="ragtest")
    assert len(indexs) >= 1, "应包含至少1个索引"
    docs = collection_manager.get_documents(collection_name="ragtest",tenant="ragtest")
    assert len(docs) >= 2, "应包含至少2个文档"



@pytest.mark.asyncio
async def test_summary_rag():
    config={"configurable": {"conversation_id": "2","thread_id": "ragtest"}}
    input = State(
        messages=[HumanMessage(content="总结该文档")],
        collection_names = ["ragtest"],
        user_id = "ragtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    assert isinstance(ret,dict)
    assert ret.get("docs") is None, "应包含至少1个文档"

@pytest.mark.asyncio
async def test_rag():
    config={"configurable": {"conversation_id": "3","thread_id": "ragtest"}}
    input = State(
        # messages=[HumanMessage(content="NTP3000Plus")],
        messages=[HumanMessage(content="Ablation Studies for Multi-Token Prediction 讲了什么？")],
        collection_names = ["ragtest"],
        user_id = "ragtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    print(ret)
    assert isinstance(ret,dict)
    assert ret.get("docs") is None, "应包含至少1个文档"
    assert ret.get("index_search_result") is None, "应包含至少1个文档"





        