from agi.scheduler.base import BaseTaskUnit, BaseTaskRuntime, TaskExecutionResult
from typing import Any
from datetime import datetime

class SystemTaskRuntime(BaseTaskRuntime):
    """内聚了大模型交互所需要的全部重型环境依赖，让 Store 干净地回归存储本质"""
    def __init__(self):
        super().__init__()


class SystemMonitorTask(BaseTaskUnit):
    """
    内部自驱监控任务：每分钟打印全局大盘，并展示全量任务明细表。
    """
    task_type = "sys_monitor"
    default_cron = "* * * * *"
    runtime_schema = BaseTaskRuntime

    def should_trigger(self, store_client: Any) -> bool:
        return True

    async def execute(self, store_client: Any) -> TaskExecutionResult:
        ns = ("sys", "scheduler", "tasks")
        all_instances = store_client.search(ns, limit=5000)
        
        stats = {
            "TOTAL": 0, "SCHEDULED": 0, "PROCESSING": 0, 
            "COMPLETED": 0, "FAILED": 0, "SUSPENDED": 0
        }
        
        # 用于存储明细行的列表
        detail_lines = []
        
        for item in all_instances:
            stats["TOTAL"] += 1
            task_id = item.key
            data = item.value
            
            # 解析状态
            status = data.get("status", "UNKNOWN")
            if status in stats:
                stats[status] += 1
                
            # 解析核心数据
            task_type = data.get("task_type", "-")
            plan = data.get("plan", {})
            target_id = plan.get("target_id", "-")
            cron = plan.get("cron_expr", "Default")
            
            # 解析最后执行时间
            last_result = data.get("last_result", {})
            last_time = last_result.get("finished_at", "-")
            if last_time != "-":
                # 截断微秒以保证表格整洁
                last_time = last_time.split(".")[0].replace("T", " ")

            # 格式化单行 (左对齐补齐空格)
            line = f"| {task_id:<32} | {task_type:<12} | {target_id:<15} | {status:<10} | {cron:<10} | {last_time:<20} |"
            detail_lines.append(line)

        # ---------------------------------------------------------
        # 1. 打印宏观大盘
        # ---------------------------------------------------------
        header_time = datetime.now().strftime('%H:%M:%S')
        board = (
            f"\n"
            f"================ 📊 调度器状态大盘 ({header_time}) ================\n"
            f"  总任务数: {stats['TOTAL']} | ⏳ 等待: {stats['SCHEDULED']} | 🔥 执行中: {stats['PROCESSING']}\n"
            f"  ✅ 成功: {stats['COMPLETED']} | ❌ 失败: {stats['FAILED']} | ⏸️ 挂起: {stats['SUSPENDED']}\n"
            f"----------------------------------------------------------------"
        )
        print(board)

        # ---------------------------------------------------------
        # 2. 打印微观全量明细表
        # ---------------------------------------------------------
        if detail_lines:
            table_header = f"| {'Task ID (Store Key)':<32} | {'Task Type':<12} | {'Target ID':<15} | {'Status':<10} | {'Cron':<10} | {'Last Executed At':<20} |"
            separator = "-" * len(table_header)
            
            print(f"\n🔍 任务全量明细 (共 {len(detail_lines)} 条):")
            print(separator)
            print(table_header)
            print(separator)
            
            # 按照 Task ID 排序后打印，方便你寻找重复项
            for line in sorted(detail_lines):
                print(line)
                
            print(separator)
            print("\n")
            
        return TaskExecutionResult(is_success=True, output_data=stats)