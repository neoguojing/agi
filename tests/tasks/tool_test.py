import unittest
from agi.tasks.tools import (
    wikidata,
    wikipedia,
    pythonREPL,
)
class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def test_yahoo_news(self):
        from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

        tool = YahooFinanceNewsTool()
        resp = tool.invoke("NVDA")
        print(resp)


    def test_youtube(self):
        from langchain_community.tools import YouTubeSearchTool

        tool = YouTubeSearchTool()
        resp = tool.invoke("llm")
        print(resp)

    def test_wikidata(self):

        tool = wikidata()
        resp = tool.invoke("Alan Turing")
        print(resp)

    def test_wikipedia(self):

        tool = wikipedia()
        resp = tool.invoke("yao min")
        print(resp)

    def test_pythonREPL(self):

        tool = pythonREPL()
        resp = tool.invoke("print(1+1)")
        print(resp)