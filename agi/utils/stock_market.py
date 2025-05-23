from langchain.tools import tool
from agi.config import log,ALPHAVANTAGE_API_KEY
import requests
import json
from dataclasses import dataclass,asdict

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
    """Useful for takeing the stock symbol or ticker as input and retrieves relevant trading data for that stock.
    Input should be stock symbol or ticker symbol."""
    
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': input,
        'apikey': ALPHAVANTAGE_API_KEY
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


