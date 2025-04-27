import unittest
from agi.utils.search_engine import SearchEngineSelector


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.search_engine = SearchEngineSelector()
    def test_search_engine(self):
        ret = self.search_engine.invoke("中美关税战最新情况")
        print(ret)
        self.assertIsNotNone(ret)
        self.assertGreater(len(ret),0)


if __name__ == "__main__":
    unittest.main()
