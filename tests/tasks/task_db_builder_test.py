from agi.tasks.db_builder import db_graph
from agi.tasks.rag_web import rag_graph,collection_manager
from agi.tasks.define import State
from agi.utils.nlp import TextProcessor
import asyncio
import pytest
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage 


@pytest.mark.asyncio
async def test_web():
    config={"configurable": {"thread_id": "ragtest"}}
    input = State(
        messages=[HumanMessage(content="今天的科技新闻")],
        user_id = "ragtest"
    )
    ret = await rag_graph.ainvoke(input,config=config)
    print(ret)
    assert isinstance(ret,dict)






        