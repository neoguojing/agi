import unittest
from agi.tasks.db_builder import db_graph
from agi.tasks.define import State
import asyncio
import pytest
@pytest.mark.asyncio
class TestDBGraph(unittest.TestCase):
    def setUp(self):        
        pass

    async def test_db_graph(self):
        config={"configurable": {"user_id": "dbtest", "conversation_id": "",
                                 "thread_id": "dbtest"}}
        
        state = State()
        state['user_id'] = "dbtest"
        state['collection_name'] = "dbtest"
        state['file_path'] = "tests/test.pdf"
        ret = await db_graph.ainvoke(state,config=config)
        print(ret)
