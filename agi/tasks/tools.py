import os
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool
from agi.utils.weather import get_weather_info
from agi.utils.search_engine import SearchEngineSelector
from agi.utils.stock_market import get_stock
from agi.config import log
from pydantic import BaseModel,Field

arxiv = ArxivAPIWrapper()

def wikipedia():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100,lang="cn")
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return tool

def wikidata():
    from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

    wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())
    return wikidata

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
    wikipedia(),
    # wikidata(),
    pythonREPL(),
    get_weather_info,
    get_stock
]
