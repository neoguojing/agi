import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPNonSessionDiscoveryDemo:

    def __init__(self):

        self.client = MultiServerMCPClient(
            {
                "stock": {
                    "transport": "http",
                    "url": "http://localhost:8001/mcp",
                }
            }
        )

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
    # Non-session Tool Loading
    # ============================================================

    async def load_tools(self):

        print("\n")
        print("=" * 80)
        print("LOAD TOOLS (NON-SESSION)")
        print("=" * 80)

        tools = await self.client.get_tools(
            server_name="stock"
        )

        print(f"\n当前工具数量: {len(tools)}\n")

        tool_names = []

        for idx, tool in enumerate(tools, 1):

            tool_names.append(tool.name)

            print(f"[{idx}] {tool.name}")

            if tool.description:
                print(tool.description[:120])

            print("-" * 80)

        return tool_names

    # ============================================================
    # Call Tool Through Temporary Session
    # ============================================================

    async def call_tool(
        self,
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

        async with self.client.session("stock") as session:

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
# MAIN
# ============================================================

async def main():

    runtime = MCPNonSessionDiscoveryDemo()

    # --------------------------------------------------------
    # 1. 初始工具列表
    # --------------------------------------------------------

    before_tools = await runtime.load_tools()

    # --------------------------------------------------------
    # 2. 检查目标工具是否存在
    # --------------------------------------------------------

    target_tool = "equity_calendar_dividend"

    print("\n")
    print("=" * 80)
    print("BEFORE ACTIVATION")
    print("=" * 80)

    if target_tool in before_tools:

        print(f"✓ 已存在工具: {target_tool}")

    else:

        print(f"✗ 工具不存在: {target_tool}")

    # --------------------------------------------------------
    # 3. 激活工具
    # --------------------------------------------------------

    await runtime.call_tool(
        "activate_tools",
        {
            "tool_names": [
                target_tool
            ]
        }
    )

    # --------------------------------------------------------
    # 4. 再次获取 tools
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RELOAD TOOLS AFTER ACTIVATION")
    print("=" * 80)

    after_tools = await runtime.load_tools()

    # --------------------------------------------------------
    # 5. 验证工具是否出现
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("AFTER ACTIVATION")
    print("=" * 80)

    if target_tool in after_tools:

        print(f"✓ 激活后已发现工具: {target_tool}")

    else:

        print(f"✗ 激活后仍未发现工具: {target_tool}")

    # --------------------------------------------------------
    # 6. diff
    # --------------------------------------------------------

    new_tools = sorted(
        set(after_tools) - set(before_tools)
    )

    print("\n")
    print("=" * 80)
    print("NEWLY DISCOVERED TOOLS")
    print("=" * 80)

    if not new_tools:

        print("\n没有新增工具")

    else:

        print(f"\n新增工具数量: {len(new_tools)}\n")

        for tool in new_tools:

            print("✓", tool)


if __name__ == "__main__":

    asyncio.run(main())