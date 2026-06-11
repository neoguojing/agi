import asyncio
import logging
import contextvars
import uuid
from datetime import datetime
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Type, List, get_type_hints
from pydantic import BaseModel, ValidationError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from agi.config import LANGGRAPH_MAIN_URL,CLOUD_MODE,DEFAULT_DB_URI
from pytz import timezone

from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import  ConnectionPool,AsyncConnectionPool

logger = logging.getLogger("AgiTaskHub")
TASK_NAMESPACE = ["agi", "kernel", "tasks"]
DLQ_NAMESPACE = ["agi", "kernel", "dlq"]

current_trace_id = contextvars.ContextVar("trace_id", default="no-trace")

# ==========================================================================
# 🧱 基础设施基类与上下文抽象
# ==========================================================================

class BaseRuntime(ABC):
    """
    运行期依赖注入基类。
    业务方可继承此类，用于挂载全局大模型客户端、向量库、数据库连接等。
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class TaskContext:
    """
    统一的任务执行上下文（隔离底层基础设施与业务数据）
    """
    trace_id: str
    store: Any                     # 分布式状态锁及存储客户端
    runtime: BaseRuntime           # 全局运行时环境
    target_id: str = "global"      # 任务面向的目标实体 ID (Cron 任务特有，Event 默认为 global)
    task_type: Optional[str] = None # 任务或事件的名称


# ==========================================================================
# 🚀 核心任务调度中心
# ==========================================================================

class AgiTaskHub:
    def __init__(self, max_event_workers: int = 3, queue_size: int = 500, tz_str: str = "Asia/Shanghai"):
        pool = AsyncConnectionPool(conninfo=DEFAULT_DB_URI, open=True)
        store = AsyncPostgresStore(conn=pool)
        self.store_client = store
        self.tz = timezone(tz_str)
        
        # ⏰ 定时任务基础设施
        self._scheduler = AsyncIOScheduler(timezone=self.tz)
        self.cron_registry: Dict[str, Dict[str, Any]] = {}
        
        # 📡 事件驱动基础设施
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.event_registry: Dict[str, Dict[str, Any]] = {}
        self.max_event_workers = max_event_workers
        self._workers: List[asyncio.Task] = []
        
        # 🛑 优雅停机控制开关
        self._stopping = asyncio.Event()

    def _extract_pydantic_model(self, func: Callable) -> Type[BaseModel]:
        """反射提取函数入参中继承自 BaseModel 的强类型参数模型"""
        type_hints = get_type_hints(func)
        for param_type in type_hints.values():
            if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                return param_type
        raise ValueError(f"❌ 函数 {func.__name__} 必须声明一个继承自 BaseModel 的参数契约！")

    # ==========================================================================
    # ⏰ 声明式定时 Cron 任务注册器
    # ==========================================================================
    def cron(self, task_type: str, runtime: BaseRuntime, cron_expr: str, target_id: str = "global", params: Optional[dict] = None, timeout: float = 300.0):
        """⏰ 声明式定时任务注解"""
        def decorator(func: Callable) -> Callable:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"❌ 函数 {func.__name__} 必须是 async def 异步协程")
                
            param_model = self._extract_pydantic_model(func)

            # 服务启动期前置静态校验
            try:
                validated_obj = param_model.model_validate(params or {})
                safe_params_dict = validated_obj.model_dump()
            except ValidationError as e:
                raise ValueError(f"❌ 静态任务 [{task_type}] 参数不匹配模型 [{param_model.__name__}]: {e.errors()}")

            async def schedule_lifecycle_wrapper():
                trace_id = f"tr-cron-{uuid.uuid4().hex[:12]}"
                token = current_trace_id.set(trace_id)
                task_id = f"cron:{task_type}:{target_id}"

                # 1. 分布式悲观锁检查
                current_meta = await self.store_client.aget(namespace=TASK_NAMESPACE, key=task_id)
                payload = dict(current_meta.value) if current_meta else {}
                if payload.get("status") == "PROCESSING":
                    logger.warning("[%s] 🔒 任务 [%s] 上一周期仍处于执行中，跳过本次重叠触发。", trace_id, task_id)
                    current_trace_id.reset(token)
                    return

                # 2. 抢占状态锁
                payload.update({"status": "PROCESSING", "updated_at": datetime.now().isoformat()})
                await self.store_client.aput(namespace=TASK_NAMESPACE, key=task_id, value=payload)

                status, is_success, err_msg = "FAILED", False, "Unknown Error"

                # 3. 构造标准的 TaskContext 容器
                ctx = TaskContext(
                    trace_id=trace_id,
                    store=self.store_client,
                    runtime=runtime,
                    target_id=target_id,
                    task_type=task_type
                )

                # 4. 核心业务硬超时熔断管控
                try:
                    await asyncio.wait_for(
                        func(ctx=ctx, payload=validated_obj),
                        timeout=timeout
                    )
                    status, is_success, err_msg = "COMPLETED", True, None
                except asyncio.TimeoutError:
                    err_msg = f"Task execution exceeded timeout limit of {timeout}s."
                    logger.critical("[%s] ❌ 定时任务 [%s] 执行超时被强行熔断！", trace_id, task_id)
                except Exception as e:
                    err_msg = str(e)
                    logger.exception("[%s] 💥 定时任务 [%s] 内部发生业务崩溃", trace_id, task_id)
                finally:
                    # 5. 🔓 状态回写：防止幽灵锁
                    try:
                        payload.update({
                            "status": status, 
                            "last_result": {"is_success": is_success, "error_log": err_msg, "finished_at": datetime.now().isoformat()},
                            "updated_at": datetime.now().isoformat()
                        })
                        await self.store_client.aput(namespace=TASK_NAMESPACE, key=task_id, value=payload)
                    except Exception as store_err:
                        logger.critical("[%s] ❌ 任务 [%s] 状态持久化解锁失败！可能导致幽灵锁: %s", trace_id, task_id, store_err)
                    
                    current_trace_id.reset(token)

            self.cron_registry[task_type] = {
                "wrapped_func": schedule_lifecycle_wrapper,
                "cron_expr": cron_expr,
                "target_id": target_id,
                "params": safe_params_dict
            }
            return func
        return decorator

    # ==========================================================================
    # 📡 消息事件订阅注册器
    # ==========================================================================
    def event(self, event_name: str, runtime: BaseRuntime, max_retries: int = 3, timeout: float = 60.0):
        """📡 消息事件订阅注解"""
        def decorator(func: Callable) -> Callable:
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(f"❌ 函数 {func.__name__} 必须是 async def 异步协程")
                
            param_model = self._extract_pydantic_model(func)

            async def event_lifecycle_wrapper(raw_body: dict, trace_id: str):
                token = current_trace_id.set(trace_id)
                
                # 构造事件的标准 TaskContext 容器
                ctx = TaskContext(
                    trace_id=trace_id,
                    store=self.store_client,
                    runtime=runtime,
                    target_id="global",  # 事件通常是广播或管道，没有特定的控制 Target，默认 global
                    task_type=event_name
                )
                
                for attempt in range(1, max_retries + 1):
                    try:
                        validated_payload = param_model.model_validate(raw_body)
                        await asyncio.wait_for(
                            func(ctx=ctx, payload=validated_payload),
                            timeout=timeout
                        )
                        current_trace_id.reset(token)
                        return  # 成功消费，中断重试
                    except asyncio.TimeoutError:
                        logger.error("[%s] ❌ 事件 [%s] 超时(%ss) [%d/%d]", trace_id, event_name, timeout, attempt, max_retries)
                    except ValidationError as val_err:
                        logger.error("[%s] ❌ 事件 [%s] 契约格式损坏，直接丢弃: %s", trace_id, event_name, val_err.errors())
                        current_trace_id.reset(token)
                        return  # 格式损坏，重试无用
                    except Exception as e:
                        logger.exception("[%s] 💥 事件 [%s] 失败 [%d/%d]", trace_id, event_name, attempt, max_retries)
                    
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** (attempt - 1))  # 指数退避
                
                # 重试耗尽，降落死信队列
                logger.critical("[%s] 🚨 [DLQ] 事件 [%s] 耗尽所有重试次数，转入死信流转！", trace_id, event_name)
                await self._save_to_dead_letter(event_name, raw_body, trace_id)
                current_trace_id.reset(token)

            self.event_registry[event_name] = {"wrapped_func": event_lifecycle_wrapper}
            return func
        return decorator

    # ==========================================================================
    # ⚙️ 核心全链路生命周期及驱动管道 (保持原有逻辑不变)
    # ==========================================================================
    async def emit(self, event_name: str, payload: dict):
        trace_id = current_trace_id.get()
        if trace_id == "no-trace":
            trace_id = f"tr-evt-{uuid.uuid4().hex[:12]}"
        await self.event_queue.put({"name": event_name, "body": payload, "trace_id": trace_id})

    async def _event_worker_loop(self, worker_id: int):
        while not (self._stopping.is_set() and self.event_queue.empty()):
            try:
                raw_event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                name, body, trace_id = raw_event["name"], raw_event["body"], raw_event["trace_id"]
                if name in self.event_registry:
                    await self.event_registry[name]["wrapped_func"](body, trace_id)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker-{worker_id} 遭遇未知循环异常: {str(e)}")

    async def start(self):
        """
        一键激活内核：
        1. 启动并清洗时钟线残留 Job
        2. 🧹 自动清理存储中的“僵尸任务”（代码已删但存储还残留的任务）
        3. 自动清洗异常崩溃引入的“幽灵锁”状态
        4. 元数据冷登记与动态挂载
        """
        if not self._scheduler.running:
            # 1. 激活时钟调度引擎并清空内存残留
            self._scheduler.start()
            try:
                self._scheduler.remove_all_jobs()
                logger.info("🗑️ 时钟线历史 Job 清理完毕。")
            except Exception as clean_err:
                logger.error("⚠️ 清理时钟线历史 Job 失败: %s", clean_err)
            
            # ==================================================================
            # 🌟 核心增补：存储层【僵尸任务】全自动清洗回收 (Garbage Collection)
            # ==================================================================
            try:
                # 1. 计算出当前代码中最新合法注册的所有 Task ID 集合
                active_task_ids = {
                    f"cron:{config['target_id']}:{t_type}" 
                    for t_type, config in self.cron_registry.items()
                }
                
                # 2. 扫描标准命名空间下的全量持久化实例
                all_stored_items = await self.store_client.asearch(TASK_NAMESPACE, limit=10000)
                
                for item in all_stored_items:
                    # 如果存储里的 Key 是 cron 任务，但在最新的代码注册表中找不到它
                    if item.key.startswith("cron:") and item.key not in active_task_ids:
                        logger.warning(
                            "🧹 [存储清洗] 检测到僵尸任务 [%s] (代码中已剔除)，正在从存储中物理抹除...", 
                            item.key
                        )
                        # 执行物理删除，确保存储干净
                        await self.store_client.adelete(namespace=TASK_NAMESPACE, key=item.key)
            except Exception as gc_err:
                logger.error("⚠️ 启动期存储垃圾回收发生异常: %s", gc_err)
            # ==================================================================

            # 2. 迭代当前最新的注册表，执行冷登记和幽灵锁修复
            for task_type, config in self.cron_registry.items():
                task_id = f"cron:{config['target_id']}:{task_type}"
                current_meta = await self.store_client.aget(namespace=TASK_NAMESPACE, key=task_id)
                
                if not current_meta:
                    # 场景 A：全新任务冷登记
                    init_payload = {
                        "task_type": task_type, "status": "ACTIVE",
                        "plan": {"target_id": config["target_id"], "cron_expr": config["cron_expr"], "params": config["params"]},
                        "updated_at": datetime.now().isoformat()
                    }
                    await self.store_client.aput(namespace=TASK_NAMESPACE, key=task_id, value=init_payload)
                else:
                    # 场景 B：历史任务防卡死自愈
                    payload = dict(current_meta.value)
                    if payload.get("status") == "PROCESSING":
                        logger.critical("🚨 [启动清洗] 检测到幽灵锁! 任务 [%s] 正在强行恢复...", task_id)
                        payload.update({
                            "status": "ACTIVE",
                            "updated_at": datetime.now().isoformat(),
                            "last_result": {
                                "is_success": False,
                                "error_log": "System crashed. Force unlocked during startup.",
                                "finished_at": datetime.now().isoformat()
                            }
                        })
                        await self.store_client.aput(namespace=TASK_NAMESPACE, key=task_id, value=payload)
                
                # 3. 挂载进时钟线
                cron_parts = config["cron_expr"].split()
                self._scheduler.add_job(
                    func=config["wrapped_func"], trigger='cron',
                    minute=cron_parts[0], hour=cron_parts[1], day=cron_parts[2], month=cron_parts[3], day_of_week=cron_parts[4],
                    id=task_id, replace_existing=True
                )

        if not self._workers:
            self._workers = [asyncio.create_task(self._event_worker_loop(i)) for i in range(self.max_event_workers)]

    async def shutdown(self):
        self._stopping.set()
        if self._workers:
            await asyncio.gather(*self._workers)
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)

    async def _save_to_dead_letter(self, event_name: str, body: dict, trace_id: str):
        dlq_key = f"{event_name}:{trace_id}"
        await self.store_client.aput(
            namespace=DLQ_NAMESPACE, key=dlq_key,
            value={"body": body, "failed_at": datetime.now().isoformat(), "trace_id": trace_id}
        )