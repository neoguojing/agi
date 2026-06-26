"""Middleware for managing scheduled tasks (cron jobs) via LangGraph SDK.

This middleware provides tools to create, delete, update, and search for recurring runs.
It is only active when CLOUD_MODE=True.
"""
import logging
from typing import Any, Callable, Awaitable, List, Optional, Union

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph_sdk import CronClient

from agi.config import CLOUD_MODE, LANGGRAPH_MAIN_URL

logger = logging.getLogger(__name__)

CRON_SYSTEM_PROMPT = """## Cron Job Management
You have access to tools for managing recurring tasks (cron jobs) in LangGraph.

### Capabilities:
- **create_cron**: Schedule a new recurring run. Requires `thread_id`, `assistant_id`, a standard cron expression (e.g., "0 9 * * *"), and an optional input dictionary.
- **delete_cron**: Remove a cron job using its unique ID.
- **update_cron**: Modify the schedule or inputs of an existing cron job by its ID.
- **search_crons**: Find existing cron jobs using a query string (e.g., searching for IDs or schedules).

### Best Practices:
1.  **Search First**: Before creating a new cron job, use `search_crons` to check if it already exists to avoid duplicates.
2.  **Cron Format**: Ensure the schedule follows standard 5-field cron format (minute hour day-of-month month day-of-week).
3.  **Error Handling**: If a tool returns an error, explain the issue clearly and suggest corrections (e.g., correcting the cron expression or checking the ID)."""

class CronMiddleware(AgentMiddleware[None, ContextT, ResponseT]):
    """
    Middleware to provide LangGraph cron job management tools.
    Only enabled in CLOUD_MODE.
    """

    def __init__(self, client: Optional[CronClient] = None):
        super().__init__()
        # Only active if CLOUD_MODE is True
        if not CLOUD_MODE:
            self.tools = []
            logger.info("CronMiddleware is disabled (CLOUD_MODE=False)")
            return

        # Use provided client or create a new one using the main URL
        if client is None:
            try:
                # Note: CronClient typically requires an HttpClient or a base URL.
                # Since we are in a middleware context, we instantiate it here.
                self.client = CronClient(url=LANGGRAPH_MAIN_URL)
            except Exception as e:
                logger.error(f"Failed to initialize CronClient: {e}")
                self.tools = []
                return
        else:
            self.client = client

        # Define the tools available for crons
        self.tools = [
            self._create_cron_tool(),
            self._delete_cron_tool(),
            self._update_cron_tool(),
            self._search_crons_tool(),
        ]
        logger.info("CronMiddleware enabled with LangGraph CronClient")

    def _create_cron_tool(self) -> BaseTool:
        description = """Create a recurring cron job in LangGraph.
Useful for automating periodic tasks like daily reports, status checks, or scheduled cleanup.
Parameters:
- thread_id: The ID of the thread where the cron should be associated.
- assistant_id: The ID of the assistant to run.
- schedule: Standard cron expression (e.g., "0 9 * * *").
- input_data: Dictionary of inputs for the recurring run."""

        async def async_create(
            thread_id: str,
            assistant_id: str,
            schedule: str,
            input_data: dict,
        ) -> str:
            """Asynchronous wrapper for create_cron."""
            try:
                # schedule follows standard cron format (e.g., "0 9 * * *")
                cron = await self.client.crons.create_for_thread(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    schedule=schedule,
                    input=input_data,
                )
                return f"Successfully created cron job with ID: {cron.id}. Schedule: {schedule}"
            except Exception as e:
                logger.error(f"Error creating cron job: {e}")
                return f"Error creating cron job: {str(e)}"

        return StructuredTool.from_function(
            name="create_cron",
            description=description,
            coroutine=async_create,
        )

    def _delete_cron_tool(self) -> BaseTool:
        description = "Delete a cron job in LangGraph by its ID."

        async def async_delete(cron_id: str) -> str:
            try:
                await self.client.crons.delete(cron_id)
                return f"Successfully deleted cron job {cron_id}"
            except Exception as e:
                logger.error(f"Error deleting cron job {cron_id}: {e}")
                return f"Error deleting cron job {cron_id}: {str(e)}"

        return StructuredTool.from_function(
            name="delete_cron",
            description=description,
            coroutine=async_delete,
        )

    def _update_cron_tool(self) -> BaseTool:
        description = """Update an existing cron job's schedule or input data.
Parameters:
- cron_id: The ID of the cron job to update.
- schedule (optional): New cron expression.
- input_data (optional): New dictionary of inputs."""

        async def async_update(
            cron_id: str,
            schedule: Optional[str] = None,
            input_data: Optional[dict] = None,
        ) -> str:
            try:
                update_params = {}
                if schedule is not None:
                    update_params["schedule"] = schedule
                if input_data is not None:
                    update_params["input"] = input_data

                if not update_params:
                    return f"No update parameters provided for cron job {cron_id}"

                await self.client.crons.update(cron_id, **update_params)
                return f"Successfully updated cron job {cron_id} with updates: {list(update_params.keys())}"
            except Exception as e:
                logger.error(f"Error updating cron job {cron_id}: {e}")
                return f"Error updating cron job {cron_id}: {str(e)}"

        return StructuredTool.from_function(
            name="update_cron",
            description=description,
            coroutine=async_update,
        )

    def _search_crons_tool(self) -> BaseTool:
        description = "Search for existing cron jobs in LangGraph using a query string."

        async def async_search(query: str) -> str:
            try:
                results = await self.client.crons.search(query)
                if not results:
                    return f"No cron jobs found matching query: '{query}'"

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
            description=description,
            coroutine=async_search,
        )

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """Async model interceptor to inject Cron management instructions if tools are available."""
        # If cron tools are provided in the current toolset, inject the prompt
        has_cron_tools = any(tool.name.startswith("cron") for tool in request.tools)
        if has_cron_tools:
            from langchain.agents.middleware.types import append_to_system_message
            new_system_message = append_to_system_message(request.system_message, CRON_SYSTEM_PROMPT)
            request = request.override(system_message=new_system_message)

        return await handler(request)
