import asyncio
import base64
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

# 引入我们重构后的后端
from agi.web.browser_backend import StatefulBrowserBackend, PageInfo

logger = logging.getLogger(__name__)

# =========================
# Browser Tool Descriptions
# =========================

BROWSER_NAVIGATE_TOOL_DESCRIPTION = """
Navigates the browser to a specific URL.

Assume this tool can access most public websites. If the User provides a URL, assume it is valid unless known otherwise.
This tool maintains the current browser session state (cookies, local storage, history).

Usage:
- **Stateful Navigation**: Calling this tool updates the current page context. Subsequent tools (click, fill, screenshot) will operate on this new page.
- **Wait Strategy**: Automatically waits for the 'domcontentloaded' event and network idle to ensure page stability before returning.
- **Error Handling**: If navigation fails (e.g., 404, timeout, DNS error), an error message is returned. Do not retry immediately; check the URL or error details first.
- **Anti-Bot Measures**: The tool automatically applies random delays and rotates User-Agent headers to reduce blocking risks.
- **Content Preview**: Returns a summary of the page (title, URL, first ~500 chars of text). It does NOT return the full HTML to prevent context overflow.
- **Full Content Access**: To read the full page content, you MUST call `browser_extract` after navigating.

Important:
- Always call `browser_navigate` before attempting to interact with a new website.
- If you need to go back to a previous page, you must navigate to that URL again (back/forward buttons are not exposed as tools).
"""

BROWSER_CLICK_TOOL_DESCRIPTION = """
Clicks an element on the current page using a CSS selector.

Prerequisites:
- You MUST have called `browser_navigate` (or another interaction tool) previously to load a page.
- Ensure the element exists and is visible. If unsure, use `browser_find` first to verify the selector.

Usage:
- **Selector Syntax**: Accepts standard CSS selectors (e.g., `#id`, `.class`, `button[type='submit']`, `a[href*='login']`).
- **Interaction Flow**: 
  1. Identify the element (visually or via `browser_find`).
  2. Call `browser_click(selector)`.
  3. The tool waits for the page to stabilize (network idle or load event) after the click, handling potential page navigations or modals.
- **Error Handling**: Returns an error if the element is not found, hidden, or obscured. If this happens, try refining your selector or scrolling (not directly supported, try clicking a visible parent).
- **State Update**: Updates the current page context. If the click triggers a navigation, the new page becomes the active context.

Tips:
- For links, prefer specific attributes: `a[href='/desired-path']`.
- For buttons, use text or role: `button:has-text('Submit')` or `input[type='submit']`.
- If a click doesn't seem to work, try `browser_screenshot` to verify the current state.
"""

BROWSER_FILL_TOOL_DESCRIPTION = """
Fills a text input field on the current page with the provided text.

Prerequisites:
- You MUST have called `browser_navigate` previously to load a page.
- The target element must be an `<input>`, `<textarea>`, or `[contenteditable]` element.

Usage:
- **Selector Syntax**: Accepts standard CSS selectors. Be specific to avoid filling the wrong field (e.g., `input#email`, `textarea[name='comment']`).
- **Clearing Behavior**: This tool automatically clears the existing content of the field before typing the new text.
- **Interaction Flow**: 
  1. Identify the input field.
  2. Call `browser_fill(selector, text)`.
  3. The tool simulates human-like typing speed internally.
- **Error Handling**: Returns an error if the element is not found or is not an editable field.

Tips:
- Combine with `browser_click` for submitting forms: `browser_fill(...)` then `browser_click('button[type=submit']`.
- For sensitive data (passwords), the tool handles them securely in memory, but avoid logging them in your reasoning.
"""

BROWSER_EXTRACT_TOOL_DESCRIPTION = """
Extracts the main text/HTML content from the current page.

**CRITICAL FOR LARGE PAGES**: This tool implements automatic content eviction to prevent context overflow.

Usage:
- **Automatic Truncation**: 
  - If the page content is small (< 15,000 tokens), the full content is returned.
  - If the page content is large, ONLY a preview (first ~15,000 tokens) is returned in the `content_preview` field.
  - **IMPORTANT**: If truncated, the response will include a `full_content_path` (e.g., `/tmp/browser_dumps/xyz.html`).
- **Reading Large Files**: If `full_content_path` is provided, you MUST use the `read_file` tool with pagination to read the rest of the content:
  - `read_file(path=full_content_path, limit=100)` to start.
  - `read_file(path=full_content_path, offset=100, limit=200)` to continue.
- **Content Format**: Returns raw HTML by default. Use `read_file` on the saved path if you need to parse specific sections with grep or regex.
- **OCR Fallback**: If the page appears blocked (empty HTML) or is an image-heavy captcha, this tool may automatically trigger OCR and return extracted text in the `ocr_text` field.

Prerequisites:
- You MUST have called `browser_navigate` previously.

Tips:
- Do NOT attempt to read the full HTML of a complex site (like Wikipedia or news sites) directly in one go. Rely on the eviction mechanism.
- If you only need specific data (e.g., "all prices"), consider using `browser_find` with a specific selector instead of extracting the whole page.
"""

BROWSER_SCREENSHOT_TOOL_DESCRIPTION = """
Captures a full-page screenshot of the current browser view.

Returns a **multimodal image content block** that the model can "see".

Usage:
- **Visual Debugging**: Essential for verifying layout, checking if elements are visible, or diagnosing why a `browser_click` or `browser_fill` failed.
- **Image Tasks**: Use this tool when the user asks "what does this page look like?" or "is there a banner?".
- **OCR Context**: If text extraction fails, a screenshot often provides the necessary context for the model to read text visually.
- **No Parameters**: Captures the entire scrollable page (full_page=True).

Prerequisites:
- You MUST have called `browser_navigate` previously.

Tips:
- Call `browser_screenshot` speculatively if you are unsure about the current page state.
- Images are returned inline; do not try to save them manually unless specifically requested by the user.
- Combining `browser_screenshot` with `browser_extract` provides both visual and textual understanding of the page.
"""

BROWSER_FIND_TOOL_DESCRIPTION = """
Finds elements on the current page matching a CSS selector.

Usage:
- **Discovery**: Use this to verify if an element exists before trying to `browser_click` or `browser_fill`.
- **Exploration**: Great for finding all links (`a`), buttons (`button`), or specific items (`.product-card`).
- **Output Limit**: Returns metadata (text, attributes) for the first 10 matches to prevent output flooding.
- **Selector Syntax**: Standard CSS selectors. Supports complex queries like `div.container > p.intro`.

Prerequisites:
- You MUST have called `browser_navigate` previously.

Tips:
- If `browser_find` returns 0 matches, your selector is likely incorrect or the element hasn't loaded yet (though the tool waits for network idle).
- Use `browser_find("a")` to list available links if you are unsure where to navigate next.
- Use `browser_find("input")` to discover form fields.
"""

class BrowserMiddleware(AgentMiddleware):
    """
    工业级浏览器中间件，配合 StatefulBrowserBackend 使用。
    
    特性:
    - 真正的会话保持 (Stateful)
    - 纯异步 (No asyncio.run)
    - 智能重试与反爬
    - 大内容自动截断 (Eviction)
    - OCR 降级支持
    """

    def __init__(
        self,
        backend: StatefulBrowserBackend,
        ocr_engine: Optional[Any] = None,
        max_retries: int = 3,
        enable_ocr_fallback: bool = True,
        content_token_limit: int = 15000,
        eviction_handler: Optional[Callable[[str], str]] = None,
    ):
        super().__init__()
        self.backend = backend
        self.ocr = ocr_engine
        self.max_retries = max_retries
        self.enable_ocr = enable_ocr_fallback
        self.content_limit = content_token_limit
        self.eviction_handler = eviction_handler
        
        self.tools = self.get_tools()
        # 简单的状态追踪
        self._last_result: Optional[PageInfo] = None

    # =========================
    # Tool Implementations (Async)
    # =========================

    async def _tool_navigate(self, url: str) -> Dict[str, Any]:
        """导航到 URL"""
        result = await self._execute_with_retry("navigate", url=url)
        return self._format_result(result)

    async def _tool_click(self, selector: str) -> Dict[str, Any]:
        """点击元素"""
        result = await self._execute_with_retry("click", selector=selector)
        return self._format_result(result)

    async def _tool_fill(self, selector: str, text: str) -> Dict[str, Any]:
        """填充文本"""
        result = await self._execute_with_retry("fill", selector=selector, text=text)
        return self._format_result(result)

    async def _tool_screenshot(self) -> Dict[str, Any]:
        """截图"""
        try:
            img_b64 = await self.backend.get_screenshot()
            if not img_b64:
                return {"error": "Failed to take screenshot"}
            
            # 确保有页面加载后再返回 URL 信息
            url_info = ""
            if self._last_result and self._last_result.url:
                url_info = f" for {self._last_result.url}"
            
            return {
                "type": "image",
                "image_data": img_b64,
                "text": f"Screenshot captured{url_info}"
            }
        except Exception as e:
            return {"error": str(e)}

    async def _tool_extract(self) -> Dict[str, Any]:
        """提取内容 (带 Eviction 逻辑)"""
        if not self._last_result:
            return {"error": "No page loaded. Please navigate first."}
        
        html = self._last_result.html or ""
        if not html:
            return {"error": "Page content is empty.", "url": self._last_result.url}

        response = {
            "url": self._last_result.url,
            "title": self._last_result.title,
        }

        # Eviction Logic
        if len(html) > self.content_limit:
            preview = html[:self.content_limit] + "\n... [Content Truncated]"
            response["content_preview"] = preview
            response["is_truncated"] = True
            
            if self.eviction_handler:
                file_path = self.eviction_handler(html)
                response["message"] = f"Content too large. Full HTML saved to: {file_path}. Use read_file to access."
                response["full_content_path"] = file_path
            else:
                response["message"] = "Content too large. Only preview shown."
        else:
            response["content"] = html
            response["is_truncated"] = False

        return response

    async def _tool_find(self, selector: str) -> Dict[str, Any]:
        """查找元素"""
        matches = await self.backend.find_elements(selector)
        return {
            "count": len(matches),
            "matches": [{"text": m.text, "attrs": m.attributes} for m in matches[:10]] # 限制返回数量
        }

    # =========================
    # Execution Logic
    # =========================

    async def _execute_with_retry(self, action: str, **kwargs) -> PageInfo:
        last_error = None
        self._last_result = None

        for attempt in range(self.max_retries):
            try:
                # Anti-bot delay
                if attempt > 0:
                    delay = random.uniform(1.0, 3.0)
                    logger.info(f"Retry {attempt+1}: waiting {delay}s")
                    await asyncio.sleep(delay)
                
                # Headers logic could be added via context options in backend
                
                start = time.time()
                if action == "navigate":
                    result = await self.backend.navigate(kwargs["url"])
                elif action == "click":
                    result = await self.backend.click(kwargs["selector"])
                elif action == "fill":
                    result = await self.backend.fill(kwargs["selector"], kwargs["text"])
                else:
                    raise ValueError(f"Unknown action: {action}")
                
                logger.info(f"{action} completed in {time.time()-start:.2f}s")

                # Update state
                self._last_result = result

                # Check for errors in result
                if result.metadata and result.metadata.get("error"):
                    raise Exception(result.metadata["error"])

                # OCR Fallback check
                if self.enable_ocr and (not result.html or len(result.html) < 100):
                    logger.warning("Page content empty/suspicious. Triggering OCR...")
                    img_b64 = await self.backend.get_screenshot()
                    if img_b64 and self.ocr:
                        img_bytes = base64.b64decode(img_b64)
                        ocr_text = await self.ocr.parse(img_bytes)
                        result.text = str(ocr_text)
                        result.screenshot_path = img_b64 # 确保截图也存在于结果中
                        result.metadata["ocr_applied"] = True

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # Final failure
        return PageInfo(
            url=kwargs.get("url", "unknown"),
            title=None,
            html=None,
            text=None,
            screenshot_path=None,
            metadata={"error": f"Failed after {self.max_retries} retries: {last_error}"}
        )

    def _format_result(self, result: PageInfo) -> Dict[str, Any]:
        if result.metadata and result.metadata.get("error"):
            return {"status": "error", "error": result.metadata["error"]}
        
        # 安全地获取文本内容，优先使用 text 属性，如果没有则尝试从 html 提取
        content = getattr(result, 'text', None) or ""
        if not content and hasattr(result, 'html'):
            content = result.html[:500] + "..." if len(result.html) > 500 else result.html
        
        return {
            "status": "success",
            "url": getattr(result, 'url', ''),
            "title": getattr(result, 'title', ''),
            "content_preview": content,
            "metadata": getattr(result, 'metadata', {})
        }

    # =========================
    # Tool Registration
    # =========================

    def get_tools(self) -> List:
        """
        返回配置好详细描述的工具列表。
        LLM 将通过这些 description 学习如何正确使用浏览器。
        """
        return [
            StructuredTool.from_function(
                coroutine=self._tool_navigate,
                name="browser_navigate",
                description=BROWSER_NAVIGATE_TOOL_DESCRIPTION,  # <--- 这里使用
                response_format="content_and_artifact" # 可选：如果需要结构化返回
            ),
            StructuredTool.from_function(
                coroutine=self._tool_click,
                name="browser_click",
                description=BROWSER_CLICK_TOOL_DESCRIPTION, # <--- 这里使用
            ),
            StructuredTool.from_function(
                coroutine=self._tool_fill,
                name="browser_fill",
                description=BROWSER_FILL_TOOL_DESCRIPTION, # <--- 这里使用
            ),
            StructuredTool.from_function(
                coroutine=self._tool_extract,
                name="browser_extract",
                description=BROWSER_EXTRACT_TOOL_DESCRIPTION, # <--- 这里使用 (最重要)
            ),
            StructuredTool.from_function(
                coroutine=self._tool_screenshot,
                name="browser_screenshot",
                description=BROWSER_SCREENSHOT_TOOL_DESCRIPTION, # <--- 这里使用
            ),
            StructuredTool.from_function(
                coroutine=self._tool_find,
                name="browser_find",
                description=BROWSER_FIND_TOOL_DESCRIPTION, # <--- 这里使用
            ),
        ]

    # =========================
    # Middleware Hooks (Optional)
    # =========================
    # 如果需要拦截其他非浏览器工具或做全局日志，可在此实现
    # 但核心逻辑已封装在 Tool 中
