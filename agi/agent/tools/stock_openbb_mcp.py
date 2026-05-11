import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPToolDiscoveryDemo:

    def __init__(self):

        self.tools = {}

    # ============================================================
    # Pretty Print
    # ============================================================

    async def print_json(self, title, data):

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        try:
            print(
                json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )
            )
        except Exception:
            print(data)

    # ============================================================
    # Load Tools From Session
    # ============================================================

    async def load_tools(self, session):

        print("\n正在从 session 获取 tools...\n")

        result = await session.list_tools()

        print(f"********************{result}")
        tools = result.tools

        self.tools = {
            tool.name: tool
            for tool in tools
        }

        print(f"当前 session 可用工具数量: {len(tools)}\n")

        for idx, tool in enumerate(tools, 1):

            print(f"[{idx}] {tool.name}")

            if tool.description:
                print(tool.description[:150])

            print("-" * 80)

    # ============================================================
    # Call Tool Helper
    # ============================================================

    async def call_tool(
            self,
            session,
            tool_name,
            arguments=None
    ):

        if arguments is None:
            arguments = {}

        print("\n")
        print("=" * 80)
        print(f"CALL TOOL: {tool_name}")
        print("=" * 80)

        print("\nArguments:")
        print(json.dumps(arguments, indent=2, ensure_ascii=False))

        result = await session.call_tool(
            tool_name,
            arguments
        )

        await self.print_json(
            f"RESULT: {tool_name}",
            result
        )

        return result

    # ============================================================
    # Parse MCP Text Result
    # ============================================================

    def parse_text_result(self, result):

        content = getattr(result, "content", None)

        if not content:
            return ""

        texts = []

        for item in content:

            text = getattr(item, "text", None)

            if text:
                texts.append(text)

        return "\n".join(texts)

    # ============================================================
    # Verify Active Tool
    # ============================================================

    async def verify_equity_tools(self, session):

        result = await self.call_tool(
            session,
            "available_tools",
            {
                "category": "equity"
            }
        )

        raw_text = self.parse_text_result(result)

        if not raw_text:
            print("\n没有返回文本内容")
            return

        try:

            parsed = json.loads(raw_text)

            print("\n")
            print("=" * 80)
            print("ACTIVE EQUITY TOOLS")
            print("=" * 80)

            active_tools = []

            for item in parsed:

                if item.get("active") is True:

                    active_tools.append(item["name"])

            print(f"\n激活工具数量: {len(active_tools)}\n")

            for tool_name in active_tools:
                print("✓", tool_name)

        except Exception as e:

            print("\n解析 available_tools 返回失败:")
            print(e)


# ============================================================
# MAIN
# ============================================================

async def main():

    client = MultiServerMCPClient(
        {
            "stock": {
                "transport": "http",
                "url": "http://localhost:8001/mcp",
            }
        }
    )

    runtime = MCPToolDiscoveryDemo()

    # ============================================================
    # Explicit Session
    # ============================================================

    async with client.session("stock") as session:

        print("\n")
        print("=" * 80)
        print("MCP SESSION STARTED")
        print("=" * 80)

        # --------------------------------------------------------
        # 1. 获取当前 session tools
        # --------------------------------------------------------

        await runtime.load_tools(session)

        # --------------------------------------------------------
        # 2. 获取 category
        # --------------------------------------------------------

        await runtime.call_tool(
            session,
            "available_categories"
        )

        # --------------------------------------------------------
        # 3. 查看 equity tools
        # --------------------------------------------------------

        await runtime.call_tool(
            session,
            "available_tools",
            {
                "category": "equity"
            }
        )

        # --------------------------------------------------------
        # 4. 激活 equity_calendar_dividend
        # --------------------------------------------------------

        await runtime.call_tool(
            session,
            "activate_tools",
            {
                "tool_names": [
                    "equity_calendar_dividend"
                ]
            }
        )

        # --------------------------------------------------------
        # 5. 再次从 session 获取 tools
        # --------------------------------------------------------

        print("\n")
        print("=" * 80)
        print("REFRESH SESSION TOOLS")
        print("=" * 80)

        await runtime.load_tools(session)

        # --------------------------------------------------------
        # 6. 验证 equity tool 状态
        # --------------------------------------------------------

        await runtime.verify_equity_tools(session)

        print("\n")
        print("=" * 80)
        print("MCP SESSION FINISHED")
        print("=" * 80)


if __name__ == "__main__":

    asyncio.run(main())