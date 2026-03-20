import pytest
import asyncio
from pathlib import Path

from .browser_backend import StatefulBrowserBackend


@pytest.fixture(scope="module")
async def browser():
    backend = StatefulBrowserBackend(
        storage_dir="./test_output",
        headless=True
    )
    await backend.initialize()
    yield backend
    await backend.close()


# =========================
# 1. 基础生命周期
# =========================

@pytest.mark.asyncio
async def test_initialize(browser):
    assert browser._browser is not None
    assert browser._context is not None
    assert browser._page is not None


# =========================
# 2. 页面导航
# =========================

@pytest.mark.asyncio
async def test_navigate(browser):
    page_info = await browser.navigate("https://docs.langchain.com/oss/python/deepagents/overview")

    assert page_info.url.startswith("https://docs.langchain.com/oss/python/deepagents/overview")
    assert page_info.title is not None
    assert "Example Domain" in page_info.text
    assert page_info.screenshot_path is not None
    assert Path(page_info.screenshot_path).exists()


# =========================
# 3. 查找元素
# =========================

@pytest.mark.asyncio
async def test_find_elements(browser):
    await browser.navigate("https://docs.langchain.com/oss/python/deepagents/overview")

    elements = await browser.find_elements("h1")

    assert len(elements) > 0
    assert "Example Domain" in elements[0].text


# =========================
# 4. 截图
# =========================

@pytest.mark.asyncio
async def test_screenshot(browser):
    await browser.navigate("https://docs.langchain.com/oss/python/deepagents/overview")

    path = await browser.get_screenshot()

    assert path != ""
    assert Path(path).exists()


# =========================
# 5. 表单交互（fill）
# =========================

@pytest.mark.asyncio
async def test_fill(browser):
    await browser.navigate("https://www.google.com")

    result = await browser.fill("textarea[name='q']", "playwright")

    assert result.metadata.get("error") is None


# =========================
# 6. 点击交互（click）
# =========================

@pytest.mark.asyncio
async def test_click(browser):
    await browser.navigate("https://docs.langchain.com/oss/python/deepagents/overview")

    result = await browser.click("a")

    # example.com 点击后会跳转 IANA
    assert result.url != ""