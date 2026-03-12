import os
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool
from agi.agent.tools.weather import get_weather_info
from agi.agent.tools.stock_market import get_stock
from agi.tasks.runtime.task_factory import TaskFactory,TASK_IMAGE_GEN,TASK_MULTI_MODEL
from agi.tasks.rag_web import rag_as_subgraph
from agi.tasks.define import State
from agi.config import log
from agi.tasks.define import AskHuman
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import YouTubeSearchTool
from langchain_core.messages import HumanMessage

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
    name = "image_gen",
    description="A tool used to generate images based on text descriptions."
)
image_gen_tool.return_direct = True
log.info(image_gen_tool.args_schema.model_json_schema())

image_recog_tool = TaskFactory.create_task(TASK_MULTI_MODEL).as_tool(
    name = "image_recog",
    description="A tool used to recognize the content of an image and return relevant descriptions or labels."
)
image_recog_tool.return_direct = True
log.info(image_recog_tool.args_schema.model_json_schema())

@tool(return_direct=True)
async def search(query: str):
    """Useful for when you need to answer questions about current events. Not for weather、stock."""
    config={"configurable": {"conversation_id": "agent","thread_id": "agent"}}
    state = State(
        messages=[HumanMessage(content=query)],
        user_id = "agent",
        feature="web"
    )
    return await rag_as_subgraph.ainvoke(state,config=config)

@tool(return_direct=True)
async def web_scrape(query: str):
    """Web scraper that takes one or more URLs as input and extracts web page content such as text, links, and metadata."""
    config={"configurable": {"conversation_id": "agent","thread_id": "agent"}}
    state = State(
        messages=[HumanMessage(content=query)],
        user_id = "agent",
        feature="scrape"
    )
    return await rag_as_subgraph.ainvoke(state,config=config)

tools = [
    AskHuman,
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    # SearchEngineSelector(),
    search,
    # TEST FAIED
    # YahooFinanceNewsTool(),
    YouTubeSearchTool(),
    web_scrape,
    wikipedia(),
    # wikidata(),
    pythonREPL(),
    get_weather_info,
    get_stock,

    image_gen_tool,
    image_recog_tool
]
