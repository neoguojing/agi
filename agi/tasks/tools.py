import os
from langchain.tools import tool
from langchain_community.utilities import WolframAlphaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities import AlphaVantageAPIWrapper
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from dataclasses import dataclass,asdict
import requests
import json
from agi.llms.model_factory import ModelFactory
from agi.tasks.prompt import stock_code_prompt
from agi.tasks.task_factory import TaskFactory,TASK_IMAGE_GEN,TASK_SPEECH

os.environ['WOLFRAM_ALPHA_APPID'] = 'QTJAQT-UPJ2R3KP89'
os.environ["ALPHAVANTAGE_API_KEY"] = '1JXIUYN26HYID5Y9'

search = DuckDuckGoSearchRun()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
alpha_vantage = AlphaVantageAPIWrapper()


@tool("image generate", return_direct=True)
def image_gen(input:str) ->str:
    """Useful for when you need to generate or draw a picture by input text.
    Text to image diffusion model capable of generating photo-realistic images given any text input."""
    task = TaskFactory.create_task(TASK_IMAGE_GEN)
    return task.run(input)

@tool("speech or audio generate", return_direct=True)
def text2speech(input:str) ->str:
    """Useful for when you need to transfer text to speech or audio.Speech to speech translation.Speech to text translation.Text to speech translation.Text to text translation.Automatic speech recognition."""
    task = TaskFactory.create_task(TASK_SPEECH)
    return task.run(input)


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
    print(params)
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except json.JSONDecodeError:
        print("JSON data is invalid.")
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
    print(params)
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

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Math",
        func=WolframAlpha.run,
        description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life."
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
            Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
            on arxiv.org."
    ),
    Tool(
        name="alphaVantage",
        func=alpha_vantage.run,
        description ="Alpha Vantage is a platform useful for provides financial market data and related services. It offers a wide range \
              of financial data, including stock market data, cryptocurrency data, and forex data. Developers can access real-time and \
                historical market data through Alpha Vantage, enabling them to perform technical analysis, modeling, and develop financial\
                applications."
    ),
    image_gen,
    text2speech,
    get_stock,
]