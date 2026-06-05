import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Type

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone

# 引入基类和存储契约
from agi.scheduler.base import BaseTaskUnit, SchedulerStorageContract, TaskExecutionResult

logger = logging.getLogger("SchedulerKernel")

def _pure_code_task_proxy(task_id: str, target_id: str, cron_expr: str, 
                          merged_params: dict, task_class: Type[BaseTaskUnit],
                          runtime_handle: Any, store_client: Any):
    """
    时钟代理执行器：统一使用基类名字空间管理状态变更。
    """
    ns = SchedulerStorageContract.TASK_NAMESPACE

    # 1. 悲观锁校验
    current_meta = store_client.get(namespace=ns, key=task_id)
    payload = dict(current_meta.value) if current_meta else {}
    if payload.get("status") == "PROCESSING":
        return

    # 2. 动态构建上下文
    instance = task_class(runtime=runtime_handle)
    instance.task_id = task_id
    instance.target_id = target_id
    instance.cron_expr = cron_expr
    instance.params = merged_params

    try:
        # 3. 准入流控拦截
        if not instance.should_trigger(store_client):
            return

        # 4. 强抢状态锁
        payload.update({"status": "PROCESSING", "updated_at": datetime.now().isoformat()})
        store_client.put(namespace=ns, key=task_id, value=payload)

        # 5. 驱动异步事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result: TaskExecutionResult = loop.run_until_complete(instance.execute(store_client))
        finally:
            loop.close()

        # 6. 归档结果
        next_status = "COMPLETED" if result.is_success else "FAILED"
        log_data = {
            "is_success": result.is_success,
            "error_log": result.error_log,
            "output_snapshot": str(result.output_data),
            "finished_at": result.finished_at.isoformat()
        }

    except Exception as e:
        next_status = "FAILED"
        log_data = {"is_success": False, "error_log": f"Fatal Proxy Panic: {str(e)}", "finished_at": datetime.now().isoformat()}
    
    # 7. 回写持久化存储
    payload.update({"status": next_status, "last_result": log_data})
    store_client.put(namespace=ns, key=task_id, value=payload)


class ConfigurationMergedScheduler:
    """
    基于物理持久化 Store、静态组件插槽与强类型运行时依赖注入的旁路调度内核
    """
    def __init__(self, store_client: Any, tz_str: str = "Asia/Shanghai"):
        self.store_client = store_client
        self.tz = timezone(tz_str)
        self.task_slots: Dict[str, Type[BaseTaskUnit]] = {}
        self.runtime_slots: Dict[str, Any] = {}
        
        self._scheduler = BackgroundScheduler(
            executors={'default': ThreadPoolExecutor(max_workers=30)},
            job_defaults={'coalesce': True, 'max_instances': 1}, 
            timezone=self.tz
        )

    def start(self):
        if not self._scheduler.running:
            self._scheduler.start()
            self._cold_start_recovery()

    def _cold_start_recovery(self):
        """自驱加载：基于统一存储契约检索任务"""
        ns = SchedulerStorageContract.TASK_NAMESPACE
        all_instances = self.store_client.search(namespace_prefix=ns, limit=2000)
        
        for item in all_instances:
            task_id = item.key
            data = item.value
            task_type = data.get("task_type")
            
            if data.get("status") == "SUSPENDED" or task_type not in self.task_slots:
                continue

            if task_type not in self.runtime_slots:
                logger.error(
                    "❌ 核心冷启动拦截: 任务实例 [%s] 的类型 '%s' 缺乏运行时依赖注入，拒绝挂载时钟线。",
                    task_id, task_type
                )
                continue
                
            plan_data = data.get("plan", {})
            self._mount_to_clock_line(task_id, task_type, plan_data)

    def _mount_to_clock_line(self, task_id: str, task_type: str, db_plan: dict):
        task_class = self.task_slots[task_type]
        runtime_handle = self.runtime_slots[task_type] 

        target_id = db_plan.get("target_id", "")
        cron_expr = db_plan.get("cron_expr") or task_class.default_cron
        
        # 运行时参数防御合并
        final_params = {**task_class.default_params, **db_plan.get("params", {})}

        cron_parts = cron_expr.split()
        self._scheduler.add_job(
            func=_pure_code_task_proxy,
            trigger='cron',
            minute=cron_parts[0], hour=cron_parts[1], day=cron_parts[2], month=cron_parts[3], day_of_week=cron_parts[4],
            id=task_id,
            args=[task_id, target_id, cron_expr, final_params, task_class, runtime_handle, self.store_client],
            replace_existing=True
        )

    def shutdown(self):
        if self._scheduler.running:
            self._scheduler.shutdown()