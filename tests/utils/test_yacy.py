import pytest
import httpx

BASE_URL = "http://localhost:8090/yacysearch.json"

@pytest.mark.asyncio
async def test_yacy_search_basic_async():
    params = {
        'query': 'ChatGPT',
        'startRecord': 0,
        'maximumRecords': 5,
        'default': 'web',
        'sort': 'rank',
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)

    assert response.status_code == 200

    data = response.json()
    print(data)
    assert 'searchResult' in data
    assert 'results' in data['searchResult']

    results = data['searchResult']['results']
    assert isinstance(results, list)
    assert len(results) <= 5

    for item in results:
        assert 'title' in item
        assert 'url' in item
        assert 'snippet' in item
