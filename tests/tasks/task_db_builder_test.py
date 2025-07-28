from agi.tasks.db_builder import db_graph
from agi.tasks.define import State
import asyncio
import pytest

@pytest.mark.asyncio
async def test_db_graph():
    config={"configurable": {"conversation_id": "",
                                "thread_id": "dbtest"}}
    
    state = State()
    state['user_id'] = "dbtest"
    state['collection_name'] = "dbtest"
    state['file_path'] = "tests/test.pdf"
    ret = await db_graph.ainvoke(state,config=config)
    print(ret)
