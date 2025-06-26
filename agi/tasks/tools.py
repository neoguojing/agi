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
from pydantic import BaseModel,Field
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

# 一个tool的定义,和模型绑定,让模型决定是否调用该工具
# 不会实际实现该函数
class AskHuman(BaseModel):
    """
    This model is used when an automated system or agent determines that 
    human intervention is required. It represents a question that the agent 
    wants to ask the human to proceed with a task, resolve ambiguity, or 
    make a decision that requires human judgment.

    Typical use cases include:
    - Uncertain model predictions or low-confidence outcomes
    - Missing information that requires user input
    - Safety-critical or policy-sensitive decisions
    """

    question: str = Field(
        description="A question the system wants to ask the human for clarification, confirmation, or additional input"
    )

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
