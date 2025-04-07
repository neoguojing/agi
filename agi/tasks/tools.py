import os
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_core.messages import AIMessage
from dataclasses import dataclass,asdict
import requests
import json
from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import stock_code_prompt
from agi.tasks.multi_model_app import create_text2image_chain,create_llm_task
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

search = DuckDuckGoSearchRun()
arxiv = ArxivAPIWrapper()

@tool("image generate", return_direct=True)
def image_gen(input:str) ->str:
    """Useful for when you need to generate or draw a picture by input text.
    Text to image diffusion model capable of generating photo-realistic images given any text input."""
    def parse(ai_message: AIMessage) -> any:
        """Parse the AI message."""
        return ai_message.content
    
    llm = create_llm_task()
    chain = create_text2image_chain(llm) | parse
    return chain.invoke({"text":input})

# @tool("speech or audio generate", return_direct=True)
# def text2speech(input:str) ->str:
#     """Useful for when you need to transfer text to speech or audio.Speech to speech translation.Speech to text translation.Text to speech translation.Text to text translation.Automatic speech recognition."""
#     task = TaskFactory.create_task(TASK_SPEECH)
#     return task.run(input)


@dataclass
class StockData:
    open: str
    high: str
    low: str
    close: str
    volume: str

@tool("stock or trade info", return_direct=False)
def get_stock(input:str,topk=5) ->str:
    # """Useful for get one stock trade info; input must be the stock code"""
    """Useful for takeing the stock symbol or ticker as input and retrieves relevant trading data for that stock"""

    translate = ModelFactory.get_model("llama3")
    stock_code = stock_code_prompt(input)
    llm_out = translate.invoke(stock_code)
    input = parse_stock_code(llm_out.content)
    
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': input,
        'apikey': '1JXIUYN26HYID5Y9'
    }
    log.debug(params)
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except json.JSONDecodeError:
        log.error("JSON data is invalid.")
        return "JSON data is invalid."

    if "Time Series (Daily)" not in data:
        return "JSON data does not contain 'Time Series (Daily)'."

    time_series_data = []
    for date, values in data["Time Series (Daily)"].items():
        stock_data = StockData(
            open=values["1. open"],
            high=values["2. high"],
            low=values["3. low"],
            close=values["4. close"],
            volume=values["5. volume"]
        )
        time_series_data.append((date, stock_data))

    serializable_data = {date: asdict(stock_data) for date, stock_data in time_series_data[:topk]}
    return json.dumps(serializable_data)

def get_stock_code(input:str):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'SYMBOL_SEARCH',
        'keywords': input,
        'apikey': '1JXIUYN26HYID5Y9'
    }
    log.debug(params)
    r = requests.get(url, params=params)

def parse_stock_code(input:str):
    # Split the sentence into words
    words = input.split()
    # Loop through the words to find the stock symbol
    for word in words:
        if word.isupper():
            stock_symbol = word
            break
    return stock_symbol.rstrip('.')

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
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    image_gen,
    wikipedia(),
    wikidata(),
    pythonREPL()
    # text2speech,
    # get_stock,
]
