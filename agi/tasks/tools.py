import os
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool
from langchain_core.messages import AIMessage
from agi.llms.model_factory import ModelFactory
from agi.tasks.multi_model_app import create_text2image_chain,create_llm_task
from agi.utils.weather import get_weather_info
from agi.utils.search_engine import SearchEngineSelector
from agi.config import log


arxiv = ArxivAPIWrapper()


def wikipedia():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
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

tools = [
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    SearchEngineSelector(),
    wikipedia(),
    wikidata(),
    pythonREPL(),
    get_weather_info
]
