from pydantic import BaseModel, Field
from typing import Any
import logging
from datetime import datetime
# 确保从你的核心内核包中导入了 TaskContext
from agi.scheduler.task_hub import TaskContext,hub
from datetime import timedelta
logger = logging.getLogger(__name__)

class GCCleanerSchema(BaseModel):
    retention_days: int = Field(default=7, description="历史数据保留天数")

@hub.cron(
    task_type="sys_garbage_collection",
    runtime=None,
    cron_expr="*/1 * * * *",  # 每天凌晨 2 点准时拆弹
    params={"retention_days": 7}
)
async def system_gc_job(ctx: TaskContext, payload: GCCleanerSchema):
    """
    系统自我进化任务：定期定点清洗老旧的历史流水或死信队列
    """
    history_ns = ["agi", "kernel", "task_history"]
    all_history = await ctx.store.asearch(history_ns, limit=10000)
    
    now = datetime.now()
    cutoff_time = now - timedelta(days=payload.retention_days)
    
    for item in all_history:
        # 解析其记录创建时间，超过阈值则物理删除
        created_at_str = item.value.get("finished_at")
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)
            if created_at < cutoff_time:
                await ctx.store.adelete(namespace=history_ns, key=item.key)
                
    logger.info("[%s] 🌳 调度系统完成自我净化，已剔除 %d 天前的全部历史死数据。", ctx.trace_id, payload.retention_days)

# ==============================================================================
# 1. 声明监控任务的强类型参数契约 (保持不变)
# ==============================================================================
class SystemMonitorSchema(BaseModel):
    verbose: bool = Field(default=True, description="是否打印微观全量明细表")
    limit: int = Field(default=5000, ge=1, le=10000, description="单次最大扫描的任务实例数")


# ==============================================================================
# 2. 适配为标准双参异步纯函数，全面接入 TaskContext 上下文
# ==============================================================================
@hub.cron(
    task_type="sys_monitor", 
    runtime=None,                            # 实际生产中请传入继承自 BaseRuntime 的实例
    cron_expr="*/1 * * * *",           
    target_id="global",     
    params={"verbose": True, "limit": 5000}, 
    timeout=45.0                    
)
async def sys_monitor_job(ctx: TaskContext, payload: SystemMonitorSchema):
    """
    内部自驱监控任务：每分钟由时钟线自驱触发，打印全局大盘，并展示全量任务明细表。
    """
    # 🌟 核心对齐：通过 ctx.store 调用持久化底座的异步搜索接口
    ns = ["agi", "kernel", "tasks"]
    all_instances = await ctx.store.asearch(ns, limit=payload.limit)
    
    # 🌟 状态对齐：适配新架构的 ACTIVE, PROCESSING, COMPLETED, FAILED 生产状态
    stats = {
        "TOTAL": 0, "ACTIVE": 0, "PROCESSING": 0, 
        "COMPLETED": 0, "FAILED": 0, "UNKNOWN": 0
    }
    
    detail_lines = []
    
    for item in all_instances:
        stats["TOTAL"] += 1
        task_id = item.key
        data = item.value
        
        # 解析并归流新版架构的状态
        status = data.get("status", "UNKNOWN")
        if status in stats:
            stats[status] += 1
        else:
            stats["UNKNOWN"] += 1
            
        # 解析新版 Hub 的标准元数据 Plan 结构
        task_type = data.get("task_type", "-")
        plan = data.get("plan", {})
        tgt_id = plan.get("target_id", "-")
        cron = plan.get("cron_expr", "Default")
        
        # 解析最后执行时间
        last_result = data.get("last_result", {})
        last_time = last_result.get("finished_at", "-")
        if last_time != "-":
            last_time = last_time.split(".")[0].replace("T", " ")

        # 格式化单行 (左对齐补齐空格)
        line = f"| {task_id:<32} | {task_type:<12} | {tgt_id:<15} | {status:<10} | {cron:<10} | {last_time:<20} |"
        detail_lines.append(line)

    # ---------------------------------------------------------
    # 1. 打印宏观状态大盘
    # ---------------------------------------------------------
    header_time = datetime.now().strftime('%H:%M:%S')
    board = (
        f"\n"
        f"================ 📊 统一任务中心大盘 ({header_time}) ================\n"
        f"  追踪当前监控 Trace: {ctx.trace_id}\n" # 💡 额外增益：现在可以在日志里打印本次监控的 TraceID 了
        f"  总任务数: {stats['TOTAL']} | ⏳ 活跃/等待: {stats['ACTIVE']} | 🔥 执行中: {stats['PROCESSING']}\n"
        f"  ✅ 成功: {stats['COMPLETED']} | ❌ 失败: {stats['FAILED']} | ❓ 未知/挂起: {stats['UNKNOWN']}\n"
        f"----------------------------------------------------------------"
    )
    print(board)

    # ---------------------------------------------------------
    # 2. 打印微观全量明细表
    # ---------------------------------------------------------
    if payload.verbose and detail_lines:
        table_header = f"| {'Task ID (Store Key)':<32} | {'Task Type':<12} | {'Target ID':<15} | {'Status':<10} | {'Cron':<10} | {'Last Executed At':<20} |"
        separator = "-" * len(table_header)
        
        print(f"\n🔍 任务全量明细 (共 {len(detail_lines)} 条):")
        print(separator)
        print(table_header)
        print(separator)
        
        for line in sorted(detail_lines):
            print(line)
            
        print(separator)
        print("\n")