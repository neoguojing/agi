import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Type
from langgraph.store.base import BaseStore
from agi.scheduler.memory_task.memory_task import *
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone

# 引入基类和存储契约
from agi.scheduler.base import BaseTaskUnit, SchedulerStorageContract, TaskExecutionResult

logger = logging.getLogger("SchedulerKernel")

def _pure_code_task_proxy(task_id: str, task_type: str, target_id: str, cron_expr: str, 
                          merged_params: dict, task_class: Type[BaseTaskUnit],
                          runtime_handle: BaseTaskRuntime, store_client: Any):
    """
    时钟代理执行器：负责悲观锁校验与全异步事件循环拉起。
    """
    ns = SchedulerStorageContract.TASK_NAMESPACE

    # 1. 悲观锁校验
    current_meta = store_client.get(namespace=ns, key=task_id)
    payload = dict(current_meta.value) if current_meta else {}
    if payload.get("status") == "PROCESSING":
        logger.warning("🔒 任务 [%s] 正处于 PROCESSING 状态，放弃本次并发触发。", task_id)
        return

    # 2. 实例化任务单元并注入合规的 runtime
    instance = task_class(
        runtime=runtime_handle,
        task_id=task_id,
        target_id=target_id,
        params=merged_params
    )

    try:
        # 3. 准入校验
        if not instance.should_trigger(store_client):
            return

        # 4. 抢占状态锁
        payload.update({"status": "PROCESSING", "updated_at": datetime.now().isoformat()})
        store_client.put(namespace=ns, key=task_id, value=payload)

        # 5. 驱动异步循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result: TaskExecutionResult = loop.run_until_complete(instance.execute(store_client))
        finally:
            loop.close()

        next_status = "COMPLETED" if result.is_success else "FAILED"
        log_data = {
            "is_success": result.is_success,
            "error_log": result.error_log,
            "output_snapshot": str(result.output_data),
            "finished_at": result.finished_at.isoformat()
        }

    except Exception as e:
        next_status = "FAILED"
        log_data = {
            "is_success": False, 
            "error_log": f"Fatal Proxy Panic: {str(e)}", 
            "finished_at": datetime.now().isoformat()
        }
    
    # 6. 回写持久化存储
    payload.update({"status": next_status, "last_result": log_data, "updated_at": datetime.now().isoformat()})
    store_client.put(namespace=ns, key=task_id, value=payload)


# ==============================================================================
# 5. 旁路调度内核（支持启动后安全、动态注册）
# ==============================================================================
class ConfigurationMergedScheduler:
    def __init__(self, store_client: BaseStore, tz_str: str = "Asia/Shanghai"):
        self.store_client = store_client
        self.tz = timezone(tz_str)
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        self._scheduler = BackgroundScheduler(
            executors={'default': ThreadPoolExecutor(max_workers=30)},
            job_defaults={'coalesce': True, 'max_instances': 1}, 
            timezone=self.tz
        )

    def register_task_type(self, task_cls: Type[BaseTaskUnit], runtime_handle: BaseTaskRuntime):
        """
        🚀 动态注册接口（支持启动前/启动后随时调用）
        🔒 强约束点：严密校验 runtime_handle 是否合规，拒绝任意不合规的外部依赖对象。
        """
        task_type = task_cls.task_type
        if not task_type:
            raise ValueError(f"❌ 注册失败: 类 {task_cls.__name__} 未定义静态 task_type")
            
        # 【关键校验】检查注入的运行时依赖，是否是该任务声明的 BaseTaskRuntime 子类
        if not isinstance(runtime_handle, task_cls.runtime_schema):
            raise TypeError(
                f"❌ [依赖注入拦截] 任务类型 [{task_type}] 要求的运行时契约为 "
                f"'{task_cls.runtime_schema.__name__}'，但你传入了不匹配的依赖对象 "
                f"'{type(runtime_handle).__name__}'！"
            )
            
        # 写入注册表
        self.registry[task_type] = {
            "class": task_cls,
            "runtime": runtime_handle
        }
        logger.info("🔌 依赖校验通过！任务类型 [%s] 成功挂载至内核槽位。", task_type)

        # 联动恢复：如果调度器已经在运行，动态追溯属于该新类型的存量 ACTIVE 任务
        if self._scheduler.running:
            logger.info("⚡ 检测到引擎正在运行，开始自动激活持久化层中 [%s] 的历史存量任务...", task_type)
            ns = SchedulerStorageContract.TASK_NAMESPACE
            all_instances = self.store_client.search(ns, limit=2000)
            
            for item in all_instances:
                data = item.value
                if data.get("task_type") == task_type and data.get("status") == "ACTIVE":
                    self._mount_to_clock_line(task_id=item.key, task_type=task_type, db_plan=data.get("plan", {}))
                    logger.info("🚀 存量任务 [%s] 已成功被动态追溯并挂载至时钟线！", item.key)

    def start(self):
        """启动调度内核并自驱加载历史任务"""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("⏰ 后台调度引擎已激活，执行冷启动数据恢复...")
            
            ns = SchedulerStorageContract.TASK_NAMESPACE
            all_instances = self.store_client.search(ns, limit=2000)
            for item in all_instances:
                data = item.value
                task_type = data.get("task_type")
                if data.get("status") == "ACTIVE" and task_type in self.registry:
                    self._mount_to_clock_line(task_id=item.key, task_type=task_type, db_plan=data.get("plan", {}))

    def _mount_to_clock_line(self, task_id: str, task_type: str, db_plan: dict):
        reg = self.registry[task_type]
        cron_expr = db_plan.get("cron_expr") or reg["class"].default_cron
        cron_parts = cron_expr.split()
        
        self._scheduler.add_job(
            func=_pure_code_task_proxy,
            trigger='cron',
            minute=cron_parts[0], hour=cron_parts[1], day=cron_parts[2], month=cron_parts[3], day_of_week=cron_parts[4],
            id=task_id,
            args=[task_id, task_type, db_plan.get("target_id"), cron_expr, db_plan.get("params", {}), reg["class"], reg["runtime"], self.store_client],
            replace_existing=True
        )

    def add_job(self, task_type: str, target_id: str, cron_expr: Optional[str] = None, params: Optional[dict] = None) -> str:
        """动态任务派发（必须先通过 register_task_type 注册，才能成功派发）"""
        if task_type not in self.registry:
            raise ValueError(f"❌ 调度器未注册此任务类型: '{task_type}'，请先注册该类型及其 Runtime 依赖。")

        task_cls = self.registry[task_type]["class"]
        task_id = SchedulerStorageContract.generate_task_key(task_type, target_id)
        final_cron = cron_expr or task_cls.default_cron
        final_params = {**task_cls.default_params, **(params or {})}

        # 1. 业务参数与类型的强一致性验证
        for key, val in (params or {}).items():
            if key not in task_cls.default_params:
                raise KeyError(f"❌ 参数越界: '{key}' 不是任务 [{task_type}] 允许的业务参数项")
            expected_type = type(task_cls.default_params[key])
            if not isinstance(val, expected_type):
                raise TypeError(f"❌ 参数类型不一致: 项 '{key}' 期望类型为 {expected_type.__name__}。")

        # 2. 序列化防呆检查
        try:
            json.dumps(final_params)
        except TypeError as e:
            raise ValueError(f"❌ params 中包含无法落库的复杂对象! 原因: {str(e)}")

        # 3. 持久化落库
        ns = SchedulerStorageContract.TASK_NAMESPACE
        payload = {
            "task_type": task_type,
            "status": "ACTIVE",
            "plan": {"target_id": target_id, "cron_expr": final_cron, "params": final_params},
            "updated_at": datetime.now().isoformat()
        }
        self.store_client.put(namespace=ns, key=task_id, value=payload)

        # 4. 如果调度引擎在线，直接送入生产环境时钟线
        if self._scheduler.running:
            self._mount_to_clock_line(task_id, task_type, payload["plan"])
            logger.info("🚀 任务 [%s] 已实时挂载至后台时钟线！", task_id)
            
        return task_id

    def shutdown(self):
        if self._scheduler.running:
            self._scheduler.shutdown()