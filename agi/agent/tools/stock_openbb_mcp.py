import asyncio
import json
import re
from collections import defaultdict

from langchain_mcp_adapters.client import MultiServerMCPClient


CATEGORY_PATTERNS = [
    r"\b(equity|news|economy|crypto|etf|currency|fixedincome|commodity|index)\b"
]

TOOL_PATTERNS = [
    r"`([a-zA-Z0-9_]+)`",
    r"use\s+([a-zA-Z0-9_]+)",
    r"call\s+([a-zA-Z0-9_]+)",
    r"([a-z]+_[a-z0-9_]+)",
]


class MCPDiscoveryRuntime:

    def __init__(self):

        self.tools = {}

        # parsed resources from list_resources
        self.resources = []

        # installed skill contents
        self.installed_skills = {}

        # capability graph
        self.skill_graph = defaultdict(dict)

        self.capability_graph = defaultdict(set)

    # ============================================================
    # Pretty Print
    # ============================================================

    async def print_json(self, title, data):

        print(f"\n{'=' * 80}")
        print(title)
        print(f"{'=' * 80}")

        try:
            print(
                json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False
                )
            )
        except Exception:
            print(data)

    # ============================================================
    # Extract categories
    # ============================================================

    def extract_categories(self, text):

        found = set()

        for pattern in CATEGORY_PATTERNS:

            matches = re.findall(
                pattern,
                text,
                re.IGNORECASE
            )

            for m in matches:
                found.add(m.lower())

        return sorted(found)

    # ============================================================
    # Extract tools
    # ============================================================

    def extract_tools(self, text):

        found = set()

        for pattern in TOOL_PATTERNS:

            matches = re.findall(
                pattern,
                text,
                re.IGNORECASE
            )

            for m in matches:

                if isinstance(m, tuple):
                    m = m[0]

                m = m.strip().lower()

                if len(m) < 4:
                    continue

                if "/" in m:
                    continue

                if "." in m:
                    continue

                found.add(m)

        return sorted(found)

    # ============================================================
    # Load tools
    # ============================================================

    async def load_tools(self, client):

        print("\n正在获取 tools...\n")

        tools = await client.get_tools()

        self.tools = {
            t.name: t
            for t in tools
        }

        print(f"发现 {len(tools)} 个 tools\n")

        for idx, tool in enumerate(tools, 1):

            print(f"[{idx}] {tool.name}")

            if tool.description:
                print(tool.description[:300])

            print("-" * 80)

    # ============================================================
    # Discover resources
    # ============================================================

    async def discover_resources(self):

        if "list_resources" not in self.tools:

            print("未发现 list_resources tool")
            return

        print("\n")
        print("=" * 80)
        print("DISCOVER RESOURCES")
        print("=" * 80)

        try:

            result = await self.tools["list_resources"].ainvoke({})

            await self.print_json(
                "RAW list_resources RESULT",
                result
            )

            raw_text = ""

            if isinstance(result, list):

                for item in result:

                    if isinstance(item, dict):
                        raw_text += item.get("text", "")

            elif isinstance(result, dict):

                raw_text = result.get("text", "")

            else:

                raw_text = str(result)

            parsed = json.loads(raw_text)

            valid_resources = []

            for r in parsed:

                if "uri" not in r:
                    continue

                valid_resources.append(r)

            self.resources = valid_resources

            print(f"\n发现 {len(self.resources)} 个有效 resources\n")

            for idx, r in enumerate(self.resources, 1):

                print(f"[{idx}]")

                print("URI:", r.get("uri"))
                print("NAME:", r.get("name"))
                print("DESCRIPTION:", r.get("description"))

                print("-" * 80)

        except Exception as e:

            print("discover_resources 失败:")
            print(e)

    # ============================================================
    # Load system prompt
    # ============================================================

    async def load_system_prompt(self):

        if "get_prompt" not in self.tools:
            return

        print("\n")
        print("=" * 80)
        print("SYSTEM PROMPT")
        print("=" * 80)

        try:

            result = await self.tools["get_prompt"].ainvoke(
                {
                    "name": "system_prompt"
                }
            )

            await self.print_json(
                "SYSTEM PROMPT RESULT",
                result
            )

        except Exception as e:

            print("读取失败:", e)

    # ============================================================
    # Install skills
    # ============================================================

    async def install_skills(self):

        if "read_resource" not in self.tools:

            print("未发现 read_resource tool")
            return

        if "install_skill" not in self.tools:

            print("未发现 install_skill tool")
            return

        read_tool = self.tools["read_resource"]

        install_tool = self.tools["install_skill"]

        print("\n")
        print("=" * 80)
        print("INSTALL SKILLS")
        print("=" * 80)

        for resource in self.resources:

            uri = resource.get("uri")

            if not uri:
                continue

            if not uri.endswith("SKILL.md"):
                continue

            skill_name = (
                uri
                .split("//")[-1]
                .split("/")[0]
            )

            print(f"\n读取 Skill: {skill_name}")

            try:

                result = await read_tool.ainvoke(
                    {
                        "uri": uri
                    }
                )

                await self.print_json(
                    f"RESOURCE CONTENT: {skill_name}",
                    result
                )

                content = ""

                if isinstance(result, list):

                    for item in result:

                        if isinstance(item, dict):
                            content += item.get("text", "")

                elif isinstance(result, dict):

                    content = result.get("text", "")

                else:

                    content = str(result)

                if not content.strip():

                    print("Skill 内容为空")
                    continue

                print(f"\n安装 Skill: {skill_name}")

                install_result = await install_tool.ainvoke(
                    {
                        "skill_name": skill_name,
                        "files": {
                            "SKILL.md": content
                        },
                        "target": "bundled"
                    }
                )

                await self.print_json(
                    f"INSTALL RESULT: {skill_name}",
                    install_result
                )

                self.installed_skills[skill_name] = content

                print(f"\nSkill 安装完成: {skill_name}")

            except Exception as e:

                print(f"\nSkill 安装失败: {skill_name}")
                print(e)

    # ============================================================
    # Analyze installed skills
    # ============================================================

    async def analyze_skills(self):

        print("\n")
        print("=" * 80)
        print("SKILL ANALYSIS")
        print("=" * 80)

        for skill_name, content in self.installed_skills.items():

            categories = self.extract_categories(content)

            tools = self.extract_tools(content)

            self.skill_graph[skill_name] = {
                "categories": categories,
                "tools": tools,
            }

            for c in categories:
                self.capability_graph[c].add(skill_name)

            print(f"\nSkill: {skill_name}")

            print("\nCategories:")
            print(categories)

            print("\nPotential Tools:")
            print(tools[:50])

            print("-" * 80)

    # ============================================================
    # Capability Graph
    # ============================================================

    async def print_capability_graph(self):

        print("\n")
        print("=" * 80)
        print("CAPABILITY GRAPH")
        print("=" * 80)

        if not self.capability_graph:

            print("\nCapability graph empty\n")
            return

        for category, skills in self.capability_graph.items():

            print(f"\n[{category}]")

            for s in skills:
                print(f"  - {s}")

    # ============================================================
    # Semantic discovery
    # ============================================================

    async def discover(self, query):

        q = query.lower()

        matched = []

        for category in self.capability_graph:

            if category in q:
                matched.append(category)

        if "stock" in q:
            matched.extend(["equity", "news"])

        if "bitcoin" in q:
            matched.append("crypto")

        if "macro" in q:
            matched.append("economy")

        return sorted(set(matched))


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

    runtime = MCPDiscoveryRuntime()

    # ------------------------------------------------------------
    # 1. load tools
    # ------------------------------------------------------------

    await runtime.load_tools(client)

    # ------------------------------------------------------------
    # 2. discover resources
    # ------------------------------------------------------------

    await runtime.discover_resources()

    # ------------------------------------------------------------
    # 3. system prompt
    # ------------------------------------------------------------

    await runtime.load_system_prompt()

    # ------------------------------------------------------------
    # 4. install skills
    # ------------------------------------------------------------

    await runtime.install_skills()

    # ------------------------------------------------------------
    # 5. analyze skills
    # ------------------------------------------------------------

    await runtime.analyze_skills()

    # ------------------------------------------------------------
    # 6. capability graph
    # ------------------------------------------------------------

    await runtime.print_capability_graph()

    # ------------------------------------------------------------
    # 7. semantic discovery demo
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SEMANTIC DISCOVERY DEMO")
    print("=" * 80)

    queries = [
        "analyze apple stock",
        "bitcoin trend",
        "macro economy",
    ]

    for q in queries:

        result = await runtime.discover(q)

        print(f"\nQuery: {q}")
        print("Matched Categories:", result)

    # ------------------------------------------------------------
    # summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nInstalled Skills:")
    print(list(runtime.installed_skills.keys()))

    print("\nCapability Categories:")
    print(list(runtime.capability_graph.keys()))


if __name__ == "__main__":
    asyncio.run(main())