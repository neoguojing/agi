"""Middleware for managing scheduled tasks (cron jobs) via LangGraph SDK.

This middleware provides tools to create, delete, update, and search for recurring runs.
It is only active when CLOUD_MODE=True.
"""
import logging
import json
from typing import Any, Callable, Awaitable, List, Optional, Union

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime, BaseTool
from langchain_core.tools import StructuredTool
from langgraph_sdk import get_client

from agi.config import CLOUD_MODE, LANGGRAPH_MAIN_URL

logger = logging.getLogger(__name__)

CRON_SYSTEM_PROMPT = """## Cron Job Management
You have access to tools for managing recurring tasks (cron jobs) in LangGraph.

### Capabilities:
- **create_cron**: Schedule a new recurring run. Requires a standard cron expression (e.g., "0 9 * * *") and an input string (the content of the message). Thread ID and Assistant ID are managed automatically.
- **delete_cron**: Remove a cron job using its unique ID.
- **update_cron**: Modify the schedule or inputs of an existing cron job by its ID. Provide a new input string if updating the content.
- **search_crons**: Find existing cron jobs using filters like thread_id, assistant_id, or enabled status.

### Best Practices:
1.  **Search First**: Before creating a new cron job, use `search_crons` to check if it already exists to avoid duplicates.
2.  **Cron Format**: Ensure the schedule follows standard 5-field cron format (minute hour day-of-month month day-of-week).
3.  **Error Handling**: If a tool returns an error, explain the issue clearly and suggest corrections (e.g., correcting the cron expression or checking the ID)."""

# =====================================================================
# TOOL DESCRIPTIONS
# =====================================================================

CREATE_CRON_TOOL_DESCRIPTION = """Use this tool to schedule a new recurring cron job in LangGraph.
Useful for automating periodic tasks like daily reports, status checks, or scheduled cleanup.

## 1. Schedule Requirements
- `schedule`: Standard 5-field cron expression (e.g., "0 9 * * *").
  - Field order: [minute] [hour] [day_of_month] [month] [day_of_week].
  - Examples:
    - `"*/15 * * * *"`: Run every 15 minutes.
    - `"0 9 * * 1-5"`: Run at 9:00 AM, Monday through Friday.

## 2. Input Content
- `input`: The raw text content you want the agent to process on each run (e.g., "hello!").
  - **Internal Processing**: This string will be automatically wrapped into a message history format: `{"messages": [{"role": "user", "content": "<your_input>"}]}`.
  - Constraint: Provide clear, concise instructions or data points for the agent to act upon.
"""

DELETE_CRON_TOOL_DESCRIPTION = """Delete an existing cron job from LangGraph using its unique identifier.

## Parameters
- `cron_id`: The unique ID of the cron job created during a previous call to `create_cron`.
"""

UPDATE_CRON_TOOL_DESCRIPTION = """Update an existing cron job's schedule or input content in LangGraph.

## Parameters
- `cron_id`: The unique ID of the cron job you wish to modify.
- `schedule` (optional): A new standard 5-field cron expression (e.g., "0 9 * * *").
- `input` (optional): New raw text content for the message. Similar to `create_cron`, this will be automatically wrapped into a user message in the LangGraph state.
"""

SEARCH_CRONS_TOOL_DESCRIPTION = """Search and list existing cron jobs within your LangGraph environment.
Use this first before creating new ones to prevent duplicate scheduling.

## Filters
- `thread_id` (optional): Filter by a specific thread identifier.
- `assistant_id` (optional): Filter by a specific assistant identifier.
- `enabled` (optional): Filter by status (True/False).

## Pagination and Limits
- `limit`: Maximum number of results to return (default: 10).
- `offset`: Starting index for the result list (default: 0).
"""

class CronMiddleware(AgentMiddleware[None, ContextT, ResponseT]):
    """
    Middleware to provide LangGraph cron job management tools.
    Only enabled in CLOUD_MODE.
    """

    def __init__(self, client: Optional[Any] = None):
        super().__init__()
        self.client = client
        # Only active if CLOUD_MODE is True
        if not CLOUD_MODE:
            self.tools = []
            logger.info("CronMiddleware is disabled (CLOUD_MODE=False)")
            return

        # Use provided client or create a new one using the main URL
        if self.client is None:
            try:
                # Note: CronClient typically requires an HttpClient or a base URL.
                # Since we are in a middleware context, we instantiate it here.
                self.client = get_client(url=LANGGRAPH_MAIN_URL)
            except Exception as e:
                logger.error(f"Failed to initialize CronClient: {e}")
                self.tools = []
                return

        # Define the tools available for crons
        self.tools = [
            self._create_cron_tool(),
            self._delete_cron_tool(),
            self._update_cron_tool(),
            self._search_crons_tool(),
        ]
        logger.info("CronMiddleware enabled with LangGraph CronClient")

    def _get_client_error(self) -> str:
        return "Error: CronClient is not initialized. Check CLOUD_MODE and configuration."

    def _create_cron_tool(self) -> BaseTool:
        async def async_create(
            runtime: ToolRuntime[ContextT, Any],
            schedule: str,
            input: str,
        ) -> str:
            """Asynchronous wrapper for create_cron. Thread ID and Assistant ID are injected automatically."""
            if not self.client:
                return self._get_client_error()
            try:
                # Extract thread_id from runtime context
                thread_id = getattr(runtime.context, "thread_id", None)
                if not thread_id:
                    return "Error: Could not retrieve thread_id from runtime context."

                # Automatically format the raw string into the required LangGraph structure
                input_dict = {"messages": [{"role": "user", "content": input}]}

                cron = await self.client.crons.create_for_thread(
                    thread_id=thread_id,
                    assistant_id="main",
                    schedule=schedule,
                    input=input_dict,
                )
                return f"Successfully created cron job with ID: {cron.id}. Schedule: {schedule}"
            except Exception as e:
                logger.error(f"Error creating cron job: {e}")
                return f"Error creating cron job: {str(e)}"

        return StructuredTool.from_function(
            name="create_cron",
            description=CREATE_CRON_TOOL_DESCRIPTION,
            coroutine=async_create,
        )

    def _delete_cron_tool(self) -> BaseTool:
        async def async_delete(runtime: ToolRuntime[ContextT, Any], cron_id: str) -> str:
            if not self.client:
                return self._get_client_error()
            try:
                await self.client.crons.delete(cron_id)
                return f"Successfully deleted cron job {cron_id}"
            except Exception as e:
                logger.error(f"Error deleting cron job {cron_id}: {e}")
                return f"Error deleting cron job {cron_id}: {str(e)}"

        return StructuredTool.from_function(
            name="delete_cron",
            description=DELETE_CRON_TOOL_DESCRIPTION,
            coroutine=async_delete,
        )

    def _update_cron_tool(self) -> BaseTool:
        async def async_update(
            runtime: ToolRuntime[ContextT, Any],
            cron_id: str,
            schedule: Optional[str] = None,
            input: Optional[str] = None,
        ) -> str:
            if not self.client:
                return self._get_client_error()
            try:
                update_params = {}
                if schedule is not None:
                    update_params["schedule"] = schedule
                if input is not None:
                    # Automatically format the raw string into the required LangGraph structure
                    update_params["input"] = {"messages": [{"role": "user", "content": input}]}

                if not update_params:
                    return f"No update parameters provided for cron job {cron_id}"

                await self.client.crons.update(cron_id, **update_params)
                return f"Successfully updated cron job {cron_id} with updates: {list(update_params.keys())}"
            except Exception as e:
                logger.error(f"Error updating cron job {cron_id}: {e}")
                return f"Error updating cron job {cron_id}: {str(e)}"

        return StructuredTool.from_function(
            name="update_cron",
            description=UPDATE_CRON_TOOL_DESCRIPTION,
            coroutine=async_update,
        )

    def _search_crons_tool(self) -> BaseTool:
        async def async_search(
            runtime: ToolRuntime[ContextT, Any],
            thread_id: Optional[str] = None,
            assistant_id: Optional[str] = None,
            enabled: Optional[bool] = None,
            limit: int = 10,
            offset: int = 0,
        ) -> str:
            if not self.client:
                return self._get_client_error()
            try:
                results = await self.client.crons.search(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    enabled=enabled,
                    limit=limit,
                    offset=offset
                )
                if not results:
                    return f"No cron jobs found matching filters: thread_id={thread_id}, assistant_id={assistant_id}, enabled={enabled}"

                res_summary = []
                for r in results:
                    # Ensure we are accessing the correct attributes based on what CronClient returns
                    res_summary.append(f"- ID: {getattr(r, 'id', 'N/A')}, Schedule: {getattr(r, 'schedule', 'N/A')}")

                return f"Found {len(results)} cron job(s):\n" + "\n".join(res_summary)
            except Exception as e:
                logger.error(f"Error searching for crons: {e}")
                return f"Error searching for crons: {str(e)}"

        return StructuredTool.from_function(
            name="search_crons",
            description=SEARCH_CRONS_TOOL_DESCRIPTION,
            coroutine=async_search,
        )

    def wrap_tool_call(self, request, handler):
        # No manual injection needed in wrap_tool_call anymore as the tools themselves
        # extract thread_id from runtime which is provided by LangChain/LangGraph.
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # If cron tools are provided in the current toolset AND CLOUD_MODE is True, inject the prompt
        has_cron_tools = any(tool.name.startswith("cron") for tool in request.tools)
        if has_cron_tools and CLOUD_MODE:
            from langchain.agents.middleware.types import append_to_system_message
            new_system_message = append_to_system_message(request.system_message, CRON_SYSTEM_PROMPT)
            request = request.override(system_message=new_system_message)

        return await handler(request)
