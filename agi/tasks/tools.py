import os
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool
from agi.utils.weather import get_weather_info
from agi.utils.search_engine import SearchEngineSelector
from agi.utils.stock_market import get_stock
from agi.utils.scrape import WebScraper
from agi.tasks.task_factory import TaskFactory,TASK_IMAGE_GEN,TASK_MULTI_MODEL
from agi.config import log
from agi.tasks.define import AskHuman
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import YouTubeSearchTool

arxiv = ArxivAPIWrapper()

def wikipedia():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    runner = WikipediaQueryRun(api_wrapper=api_wrapper)
    return runner

# TEST FAIED
def wikidata():
    from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

    runner = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
    return runner

def pythonREPL():
    from langchain_core.tools import Tool
    from langchain_experimental.utilities import PythonREPL
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    return repl_tool

image_gen_tool = TaskFactory.create_task(TASK_IMAGE_GEN).as_tool(
    name = "",
    description=""
)
image_gen_tool.return_direct = True
print(image_gen_tool.args_schema.model_json_schema())

image_recog_tool = TaskFactory.create_task(TASK_MULTI_MODEL).as_tool(
    name = "",
    description=""
)
image_recog_tool.return_direct = True
print(image_recog_tool.args_schema.model_json_schema())

tools = [
    AskHuman,
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    SearchEngineSelector(),
    # TEST FAIED
    # YahooFinanceNewsTool(),
    YouTubeSearchTool(),
    WebScraper(),
    wikipedia(),
    # wikidata(),
    pythonREPL(),
    get_weather_info,
    get_stock,

    image_gen_tool,
    image_recog_tool
]
