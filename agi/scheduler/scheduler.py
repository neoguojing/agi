import asyncio
import traceback
import json
import logging
from datetime import datetime
from typing import Any, Dict, Type, Optional
from langgraph.store.base import BaseStore
from agi.scheduler.memory_task.memory_task import *
from pytz import timezone

# 🌟 核心变更 1：切换为 APScheduler 的 异步IOScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
# 引入基类和存储契约
from agi.scheduler.base import BaseTaskUnit, SchedulerStorageContract, TaskExecutionResult

logger = logging.getLogger("SchedulerKernel")


# 🌟 核心变更 2：代理执行器全面改为 async def，彻底抛弃 run_until_complete
async def _pure_code_task_proxy(task_id: str, task_type: str, target_id: str, cron_expr: str,
                                merged_params: dict, task_class: Type[BaseTaskUnit],
                                runtime_handle: Any, store_client: BaseStore):
    """
    时钟代理执行器：由 AsyncIOScheduler 原生驱动的异步协程任务。
    """
    if not task_id:
        logger.error(f"❌ [Execution Aborted] 触发的任务 ID 为空! Type: {task_type}, Target: {target_id}")
        return

    ns = SchedulerStorageContract.TASK_NAMESPACE
    logger.info("⏰ [Clock Trigger] 尝试触发任务 [%s] (Type: %s, Target: %s)", task_id, task_type, target_id)

    try:
        # 1. 悲观锁校验（切换为 LangGraph Store 异步方法 aget）
        current_meta = await store_client.aget(namespace=ns, key=task_id)
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

        # 3. 准入校验（既然全链路异步化，准入校验也应支持 await）
        should_trigger = await instance.should_trigger(store_client)
  
        if not should_trigger:
            logger.info("⏭️ 任务 [%s] 准入校验未通过 (should_trigger=False)，跳过执行。", task_id)
            return

        # 4. 抢占状态锁（切换为 异步方法 aput）
        payload.update({"status": "PROCESSING", "updated_at": datetime.now().isoformat()})
        await store_client.aput(namespace=ns, key=task_id, value=payload)
        logger.info("⚡ 任务 [%s] 成功抢占状态锁，开始进入执行阶段...", task_id)

        # 5. 驱动异步任务（原生 await，不再创建新 loop）
        result: TaskExecutionResult = await instance.execute(store_client)

        next_status = "COMPLETED" if result.is_success else "FAILED"
        log_data = {
            "is_success": result.is_success,
            "error_log": result.error_log,
            "output_snapshot": str(result.output_data),
            "finished_at": result.finished_at.isoformat()
        }

        if result.is_success:
            logger.info("✅ 任务 [%s] 执行成功。结果快照: %s", task_id, log_data["output_snapshot"])
        else:
            logger.error("❌ 任务 [%s] 执行失败。错误详情: %s", task_id, result.error_log)

    except Exception as e:
        next_status = "FAILED"
        log_data = {
            "is_success": False,
            "error_log": f"Fatal Proxy Panic: {str(e)}",
            "finished_at": datetime.now().isoformat()
        }
        logger.exception("💥 任务 [%s] 发生致命崩溃: %s\n %s", task_id, e, traceback.format_exc())

    # 6. 回写持久化存储（切换为 异步方法 aput）
    try:
        payload.update({"status": next_status, "last_result": log_data, "updated_at": datetime.now().isoformat()})
        await store_client.aput(namespace=ns, key=task_id, value=payload)
        logger.info("💾 任务 [%s] 状态已同步至存储, 最终状态: %s", task_id, next_status)
    except Exception as store_err:
        logger.error("❌ 任务 [%s] 最终状态回写持久化失败: %s", task_id, str(store_err))


# ==============================================================================
# 5. 旁路调度内核（全异步自驱版本）
# ==============================================================================
class ConfigurationMergedScheduler:
    def __init__(self, store_client: BaseStore, tz_str: str = "Asia/Shanghai"):
        self.store_client = store_client
        self.tz = timezone(tz_str)
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        # 🌟 核心变更 3：使用 AsyncIOScheduler，无需再配置 ThreadPoolExecutor
        self._scheduler = AsyncIOScheduler(timezone=self.tz)

    async def register_task_type(self, task_cls: Type[BaseTaskUnit], runtime_handle: Any):
        """🚀 动态注册接口（支持启动前/启动后随时调用）"""
        task_type = task_cls.task_type
        if not task_type:
            raise ValueError(f"❌ 注册失败: 类 {task_cls.__name__} 未定义静态 task_type")
            
        if not isinstance(runtime_handle, task_cls.runtime_schema):
            raise TypeError(
                f"❌ [依赖注入拦截] 任务类型 [{task_type}] 要求的运行时契约为 "
                f"'{task_cls.runtime_schema.__name__}'，但你传入了不匹配的依赖对象 "
                f"'{type(runtime_handle).__name__}'！"
            )
            
        self.registry[task_type] = {
            "class": task_cls,
            "runtime": runtime_handle
        }
        logger.info("🔌 依赖校验通过！任务类型 [%s] 成功挂载至内核槽位。", task_type)

        # 🌟 核心变更 4：动态追溯使用异步 asearch
        if self._scheduler.running:
            logger.info("⚡ 检测到引擎正在运行，开始自动激活持久化层中 [%s] 的历史存量任务...", task_type)
            ns = SchedulerStorageContract.TASK_NAMESPACE
            all_instances = await self.store_client.asearch(ns, limit=2000)
            
            for item in all_instances:
                data = item.value
                if data.get("task_type") == task_type and data.get("status") == "ACTIVE":
                    self._mount_to_clock_line(task_id=item.key, task_type=task_type, db_plan=data.get("plan", {}))
                    logger.info("🚀 存量任务 [%s] 已成功被动态追溯并挂载至时钟线！", item.key)

    async def start(self):
        """启动调度内核并自驱加载历史任务"""
        if not self._scheduler.running:
            # 启动调度器监听
            self._scheduler.start()
            logger.info("⏰ 后台异步调度引擎已激活，执行冷启动数据恢复...")
            
            # 🌟 核心变更 5：冷启动数据扫描改用 asearch
            ns = SchedulerStorageContract.TASK_NAMESPACE
            all_instances = await self.store_client.asearch(ns, limit=2000)
            for item in all_instances:
                data = item.value
                task_type = data.get("task_type")
                if data.get("status") == "ACTIVE" and task_type in self.registry:
                    self._mount_to_clock_line(task_id=item.key, task_type=task_type, db_plan=data.get("plan", {}))

    def _mount_to_clock_line(self, task_id: str, task_type: str, db_plan: dict):
        """
        将任务向调度器注册（纯内存操作，不涉及 I/O，因此保持同步方法定义即可）
        """
        if not task_id:
            logger.error(f"🚨 [CRITICAL] 尝试挂载空 ID 任务! 拦截成功。")
            return
        reg = self.registry[task_type]
        cron_expr = db_plan.get("cron_expr") or reg["class"].default_cron
        cron_parts = cron_expr.split()

        logger.info("📡 正在将任务 [%s] (%s) 挂载至时钟线, Cron: [%s]", task_id, task_type, cron_expr)
        
        # ⚠️ 注意：这里添加的是 async 代理函数
        self._scheduler.add_job(
            func=_pure_code_task_proxy,
            trigger='cron',
            minute=cron_parts[0], hour=cron_parts[1], day=cron_parts[2], month=cron_parts[3], day_of_week=cron_parts[4],
            id=task_id,
            args=[task_id, task_type, db_plan.get("target_id"), cron_expr, db_plan.get("params", {}), reg["class"], reg["runtime"], self.store_client],
            replace_existing=True
        )

    async def add_job(self, task_type: str, target_id: str, cron_expr: Optional[str] = None, params: Optional[dict] = None) -> str:
        """动态任务派发"""
        if task_type not in self.registry:
            raise ValueError(f"❌ 调度器未注册此任务类型: '{task_type}'")

        task_cls = self.registry[task_type]["class"]
        task_id = SchedulerStorageContract.generate_task_key(task_type, target_id)
        final_cron = cron_expr or task_cls.default_cron
        final_params = {**task_cls.default_params, **(params or {})}

        # 业务参数强验证
        for key, val in (params or {}).items():
            if key not in task_cls.default_params:
                raise KeyError(f"❌ 参数越界: '{key}' 不是任务 [{task_type}] 允许的业务参数项")
            expected_type = type(task_cls.default_params[key])
            if not isinstance(val, expected_type):
                raise TypeError(f"❌ 参数类型不一致: 项 '{key}' 期望类型为 {expected_type.__name__}。")

        try:
            json.dumps(final_params)
        except TypeError as e:
            raise ValueError(f"❌ params 中包含无法落库的复杂对象! 原因: {str(e)}")

        # 🌟 核心变更 6：落库改用 aput
        ns = SchedulerStorageContract.TASK_NAMESPACE
        payload = {
            "task_type": task_type,
            "status": "ACTIVE",
            "plan": {"target_id": target_id, "cron_expr": final_cron, "params": final_params},
            "updated_at": datetime.now().isoformat()
        }
        await self.store_client.aput(namespace=ns, key=task_id, value=payload)

        if self._scheduler.running:
            self._mount_to_clock_line(task_id, task_type, payload["plan"])
            logger.info("🚀 任务 [%s] 已实时挂载至后台时钟线！", task_id)
            
        return task_id

    async def remove_job_by_id(self, task_id: str) -> bool:
        """[底层接口] 根据绝对的 task_id 删除并卸载任务"""
        ns = SchedulerStorageContract.TASK_NAMESPACE
        
        # 🌟 核心变更 7：删除改用 adelete
        await self.store_client.adelete(ns, key=task_id)
        
        try:
            if self._scheduler.get_job(task_id):
                self._scheduler.remove_job(task_id)
                logger.info("🗑️ [DELETE] 任务 %s 已从数据库抹除，并安全摘除内存时钟线。", task_id)
                return True
            else:
                logger.info("🗑️ [DELETE] 任务 %s 已从数据库抹除 (当前内存中未挂载)。", task_id)
                return True
        except Exception as e:
            logger.error("❌ [DELETE FAILED] 卸载任务 %s 时发生异常: %s", task_id, str(e))
            return False

    async def remove_job(self, task_type: str, target_id: str) -> bool:
        """[业务接口] 根据业务语义优雅删除任务"""
        task_id = SchedulerStorageContract.generate_task_key(task_type, target_id)
        return await self.remove_job_by_id(task_id)
    
    async def clear_all_jobs(self) -> int:
        """[核弹级接口] 清空系统中所有的定时任务记录与执行线"""
        ns = SchedulerStorageContract.TASK_NAMESPACE
        
        # 🌟 核心变更 8：批量检索与物理删除改用 asearch + adelete
        all_instances = await self.store_client.asearch(ns, limit=10000)
        deleted_count = 0
        
        for item in all_instances:
            try:
                await self.store_client.adelete(namespace=ns, key=item.key)
                deleted_count += 1
            except Exception as e:
                logger.error("❌ 清理任务 %s 的持久化数据时失败: %s", item.key, str(e))
                
        if self._scheduler.running:
            self._scheduler.remove_all_jobs()
            
        logger.warning("☢️ [CLEAR ALL] 调度器已被重置，共清理了 %d 个持久化任务。", deleted_count)
        return deleted_count
    
    async def shutdown(self, wait: bool = True):
        """🛑 纯异步优雅下线接口"""
        if self._scheduler.running:
            # AsyncIOScheduler 的 shutdown 会等待当前事件循环里激活的协程完毕
            self._scheduler.shutdown(wait=wait)
            logger.warning("🛑 异步调度内核已发出停机指令...")
            
            # 如果不等待强制关闭，顺手清理正在 PROCESSING 状态的幽灵状态锁
            if not wait:
                try:
                    ns = SchedulerStorageContract.TASK_NAMESPACE
                    all_instances = await self.store_client.asearch(ns, limit=2000)
                    for item in all_instances:
                        if item.value.get("status") == "PROCESSING":
                            payload = dict(item.value)
                            payload.update({
                                "status": "FAILED", 
                                "last_result": {"is_success": False, "error_log": "System forced shutdown."},
                                "updated_at": datetime.now().isoformat()
                            })
                            await self.store_client.aput(namespace=ns, key=item.key, value=payload)
                except Exception as e:
                    logger.error("❌ 优雅停机释放锁时发生异常: %s", str(e))
                    
            logger.info("✨ 异步内核安全退出。")