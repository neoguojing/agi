import asyncio
import json
import re
from collections import defaultdict

from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():

    client = MultiServerMCPClient(
        {
            "stock": {
                "transport": "http",
                "url": "http://localhost:8001/mcp",
            }
        }
    )

    tools = await client.get_tools()
    print(tools)
'''
        [1] available_categories
List available tool categories and subcategories with tool counts.
--------------------------------------------------------------------------------
[2] available_tools
List tools in a specific category and subcategory.
--------------------------------------------------------------------------------
[3] activate_tools
Activate one or more tools for this session.
--------------------------------------------------------------------------------
[4] deactivate_tools
Deactivate one or more tools for this session.
--------------------------------------------------------------------------------
[5] activate_category
Activate all tools in a category (or subcategory) for this session.
--------------------------------------------------------------------------------
[6] install_skill
Install a skill (SKILL.md + supporting files) into a SkillsDirectoryProvider.

Creates the skill directory if needed, writes all files,
and registers the new skill with the target provider so it becomes
immediately available via list_resources / read_resource.
--------------------------------------------------------------------------------
[7] list_prompts
List all available prompts.

Returns JSON with prompt metadata including name, description,
and optional arguments.
--------------------------------------------------------------------------------
[8] get_prompt
Get a prompt by name with optional arguments.

Returns the rendered prompt as JSON with a messages array.
Arguments should be provided as a dict mapping argument names
to values.
--------------------------------------------------------------------------------
[9] list_resources
List all available resources and resource templates.

Returns JSON with resource metadata. Static resources have a
'uri' field, while templates have a 'uri_template' field with
placeholders like {name}.
--------------------------------------------------------------------------------
[10] read_resource
Read a resource by its URI.

For static resources, provide the exact URI. For templated
resources, provide the URI with template parameters filled in.

Returns the resource content as a string. Binary content is
base64-encoded.
'''

if __name__ == "__main__":
    asyncio.run(main())