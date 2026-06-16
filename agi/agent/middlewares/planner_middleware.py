from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Annotated, Any, cast

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, tool
from langgraph.runtime import Runtime
from langgraph.types import Command

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)

from langchain.tools import ToolRuntime


# ============================================================
# Domain Models
# ============================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Task(BaseModel):
    id: str = Field(description="Unique task id")

    title: str = Field(description="Task title")

    description: str = Field(description="Task description")

    role: str = Field(
        description="executor role such as coder/reviewer/tester/researcher"
    )

    depends_on: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    status: TaskStatus = TaskStatus.PENDING


class WorkflowPlan(BaseModel):
    goal: str

    tasks: list[Task]


# ============================================================
# State
# ============================================================

class PlanningState(AgentState[ResponseT]):
    workflow: Annotated[
        NotRequired[WorkflowPlan],
        OmitFromInput,
    ]


# ============================================================
# Tool Input
# ============================================================

class WritePlanInput(BaseModel):
    plan: WorkflowPlan


# ============================================================
# Planner Prompt
# ============================================================

PLANNER_SYSTEM_PROMPT = """
You are an expert workflow planner.

Your responsibility:

1. Analyze the user goal.
2. Break it into executable tasks.
3. Generate a DAG using depends_on.
4. Assign proper role to each task.
5. Ensure every task is atomic.
6. Validation should be an independent task.
7. Avoid cyclic dependencies.
8. Do NOT execute tasks.
9. Do NOT solve tasks.
10. Only generate workflow plan.

Task Rules:

- Use snake_case task ids.
- Task ids must be unique.
- Keep tasks small and executable.
- Prefer parallel tasks when possible.
- Use depends_on to express dependencies.

You MUST call write_plan.
"""


# ============================================================
# DAG Validator
# ============================================================

class PlanValidator:

    @classmethod
    def validate(cls, plan: WorkflowPlan):

        cls._check_duplicate_ids(plan)
        cls._check_dependencies(plan)
        cls._check_cycle(plan)

    @staticmethod
    def _check_duplicate_ids(plan: WorkflowPlan):

        ids = [t.id for t in plan.tasks]

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task ids detected")

    @staticmethod
    def _check_dependencies(plan: WorkflowPlan):

        task_ids = {t.id for t in plan.tasks}

        for task in plan.tasks:

            for dep in task.depends_on:

                if dep not in task_ids:
                    raise ValueError(
                        f"task={task.id} depends on unknown task={dep}"
                    )

    @staticmethod
    def _check_cycle(plan: WorkflowPlan):

        graph = {
            task.id: task.depends_on
            for task in plan.tasks
        }

        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str):

            if node in stack:
                raise ValueError(
                    f"cycle detected at task={node}"
                )

            if node in visited:
                return

            stack.add(node)

            for dep in graph[node]:
                dfs(dep)

            stack.remove(node)

            visited.add(node)

        for node in graph:
            dfs(node)


# ============================================================
# Tool
# ============================================================

@tool(description="Create workflow DAG plan")
def write_plan(
    plan: WorkflowPlan,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:

    return Command(
        update={
            "workflow": plan,
            "messages": [
                ToolMessage(
                    content="Workflow plan created",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _write_plan(
    runtime: ToolRuntime[
        ContextT,
        PlanningState[ResponseT],
    ],
    plan: WorkflowPlan,
) -> Command[Any]:

    return Command(
        update={
            "workflow": plan,
            "messages": [
                ToolMessage(
                    content="Workflow plan created",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


async def _awrite_plan(
    runtime: ToolRuntime[
        ContextT,
        PlanningState[ResponseT],
    ],
    plan: WorkflowPlan,
) -> Command[Any]:

    return _write_plan(runtime, plan)


# ============================================================
# Middleware
# ============================================================

class PlannerMiddleware(
    AgentMiddleware[
        PlanningState[ResponseT],
        ContextT,
        ResponseT,
    ]
):

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
    ):
        super().__init__()

        self.system_prompt = system_prompt

        self.tools = [
            StructuredTool.from_function(
                name="write_plan",
                description="Create workflow DAG plan",
                func=_write_plan,
                coroutine=_awrite_plan,
                args_schema=WritePlanInput,
                infer_schema=False,
            )
        ]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            ModelResponse[ResponseT],
        ],
    ):

        if request.system_message:

            content = [
                *request.system_message.content_blocks,
                {
                    "type": "text",
                    "text": f"\n\n{self.system_prompt}",
                },
            ]

        else:

            content = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                }
            ]

        return handler(
            request.override(
                system_message=SystemMessage(
                    content=cast(
                        "list[str | dict[str, str]]",
                        content,
                    )
                )
            )
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]],
            Awaitable[ModelResponse[ResponseT]],
        ],
    ):

        if request.system_message:

            content = [
                *request.system_message.content_blocks,
                {
                    "type": "text",
                    "text": f"\n\n{self.system_prompt}",
                },
            ]

        else:

            content = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                }
            ]

        return await handler(
            request.override(
                system_message=SystemMessage(
                    content=cast(
                        "list[str | dict[str, str]]",
                        content,
                    )
                )
            )
        )

    @override
    def after_model(
        self,
        state: PlanningState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:

        workflow = state.get("workflow")

        if workflow is None:
            return None

        try:

            PlanValidator.validate(workflow)

        except Exception as e:

            return {
                "messages": [
                    ToolMessage(
                        content=str(e),
                        tool_call_id="planner_validation",
                        status="error",
                    )
                ]
            }

        return None

    @override
    async def aafter_model(
        self,
        state: PlanningState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:

        return self.after_model(
            state,
            runtime,
        )