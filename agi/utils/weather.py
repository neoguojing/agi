import time
from datetime import datetime, timedelta
import requests
from typing import Any, Dict, Optional
import json
import os
import difflib
from agi.config import CACHE_DIR
import jieba.posseg as pseg
from langchain.tools import tool

class WeatherAPIError(Exception):
    """自定义异常：天气接口调用或解析失败"""
    pass

# 获取省级行政单位列表
def get_province_list():
    url = "http://www.nmc.cn/rest/province"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        provinces = json.loads(response.text)
        return provinces
    except requests.RequestException as e:
        print(f"获取省份列表失败: {e}")
        return []

# 获取某个省份下的城市列表
def get_city_list(province_code):
    url = f"http://www.nmc.cn/rest/province/{province_code}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        cities = json.loads(response.text)
        return cities
    except requests.RequestException as e:
        print(f"获取 {province_code} 的城市列表失败: {e}")
        return []

# 构建搜索列表，包含省份和城市信息
def build_search_list():
    file_path = f'{CACHE_DIR}/search_list.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            search_list = json.load(f)
            ts = search_list["time"]
            ts_time = datetime.fromtimestamp(ts / 1000)
            now = datetime.now()
            # 比较是否早于当前时间 10 天
            ten_days_ago = now - timedelta(days=10)
            if ts_time > ten_days_ago:
                return search_list
    
    provinces = get_province_list()
    search_list = {}
    
    for province in provinces:
        province_name = province['name']
        province_code = province['code']
        province_url = province['url']
        # 添加省份信息
        search_list[province_name] = {
            'code': province_code,
            'url': province_url,
            'type': 'province'
        }
        
        # 获取该省份下的城市列表
        cities = get_city_list(province_code)
        for city in cities:
            city_name = city['city']
            city_code = city['code']
            city_url = city['url']
            # 添加城市信息
            search_list[city_name] = {
                'code': city_code,
                'url': city_url,
                'type': 'city'
            }
    search_list["time"] = int(time.time() * 1000)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(search_list, f, indent=4, ensure_ascii=False)
        return search_list

# 构建搜索列表（建议在程序启动时构建一次并缓存）
search_list = build_search_list()

# 模糊匹配城市名称，返回最佳匹配的代码和 URL
def find_best_match(input_name):
    
    words = pseg.cut(input_name)
    locations = []
    for word, flag in words:
        if flag == 'ns':  # 'ns'表示地名
            locations.append(word)
    if locations:
        input_name = locations[0]
    names = search_list.keys()
    # 使用 difflib 进行模糊匹配，cutoff=0.6 表示至少 60% 相似度
    best_matches = difflib.get_close_matches(input_name, names, n=1, cutoff=0.6)
    if best_matches:
        match_name = best_matches[0]
        item = search_list.get(match_name,None)
        return match_name,item['code'], item['url']
    return None,None, None

def get_nmc_weather(station_id: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    查询国家气象中心（NMC）天气实况和预报数据。

    :param station_id: 站点 ID，例如 "bwUMl"
    :param timeout: 请求超时时间（秒）
    :return: 包含 'real', 'predict', 'air' 等子字段的字典
    :raises WeatherAPIError: 接口调用失败或返回数据格式不符时抛出
    """
    # 构造 URL，最后的时间戳参数防止缓存
    ts = int(time.time() * 1000)
    url = f"http://www.nmc.cn/rest/weather"
    params = {
        "stationid": station_id,
        "_": ts
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise WeatherAPIError(f"请求天气接口失败: {e}")

    try:
        payload = resp.json()
    except ValueError:
        raise WeatherAPIError("响应不是合法的 JSON")

    # 检查返回码
    if payload.get("code") != 0 or payload.get("msg") != "success":
        raise WeatherAPIError(f"接口返回异常: {payload.get('msg')} (code={payload.get('code')})")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise WeatherAPIError("接口返回的数据格式不正确")

    # 提取常用部分
    result: Dict[str, Any] = {}
    real = data.get("real", {})
    predict = data.get("predict", {})
    air = data.get("air", {})

    # 实况部分
    result["station"] = real.get("station", {})
    result["publish_time"] = real.get("publish_time")
    result["temperature"] = real.get("weather", {}).get("temperature")
    result["humidity"] = real.get("weather", {}).get("humidity")
    result["pressure"] = real.get("weather", {}).get("airpressure")
    result["weather_info"] = real.get("weather", {}).get("info")
    result["wind"] = real.get("wind", {})

    # 未来预报（detail 列表）
    result["forecast"] = predict.get("detail", [])

    # 空气质量
    if isinstance(air,dict):
        result["aqi"] = air.get("aqi")
        result["air_quality"] = air.get("text")

    return result

# 主函数：根据城市名获取代码和天气预报 URL
@tool("Weather", return_direct=False)
def get_weather_info(city_name):
    """Utility to fetch real-time weather conditions for specified locations."""
    _,city_code, _ = find_best_match(city_name)
    return get_nmc_weather(city_code)
    