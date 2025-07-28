import unittest
from agi.tasks.db_builder import db_graph
from agi.tasks.define import State

class TestDBGraph(unittest.TestCase):
    def setUp(self):        
        pass

    def test_db_graph(self):
        state = State()
        state['user_id'] = "dbtest"
        state['collection_name'] = "dbtest"
        state['file_path'] = "tests/test.pdf"
        ret = db_graph.invoke(state)
        print(ret)
