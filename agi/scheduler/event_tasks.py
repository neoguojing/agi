from agi.scheduler import hub
from agi.scheduler.task_hub import TaskContext
from pydantic import BaseModel, Field


class ContextSummarySchema(BaseModel):
    verbose: bool = Field(default=True, description="是否打印微观全量明细表")
    limit: int = Field(default=5000, ge=1, le=10000, description="单次最大扫描的任务实例数")


@hub.event(
    event_name="context.summay", 
    runtime=None, 
    max_retries=3,
    timeout=60
)
async def handle_context_summay(ctx: TaskContext, payload: ContextSummarySchema):
    # 与 Cron 任务拥有完全一致的函数签名！
    print(f"[{ctx.trace_id}] 收到用户注册事件: {payload.user_id}")
    db = ctx.runtime.db_conn