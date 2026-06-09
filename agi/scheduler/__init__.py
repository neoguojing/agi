from agi.scheduler.scheduler import ConfigurationMergedScheduler
from agi.scheduler.memory_task.memory_task import *
from agi.scheduler.system_monitor import SystemMonitorTask,SystemTaskRuntime
from langgraph.store.postgres import PostgresStore
from psycopg_pool import  ConnectionPool
from agi.config import DEFAULT_DB_URI
from agi.agent.models import ModelProvider
import logging
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class SingletonMeta(type):
    """
    线程安全的单例元类。
    利用元类拦截类的实例化过程，确保 __init__ 在全局只执行一次。
    """
    _instances = {}
    _lock = threading.Lock()  # 引入线程锁，确保多线程并发时也是绝对单例

    def __call__(cls, *args, **kwargs):
        # 双重检查锁定（Double-Checked Locking）提高性能
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    # super().__call__ 会自动调用类的 __new__ 和 __init__
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
def run_once(func):
    """
    装饰器：确保被装饰的实例方法在当前生命周期内仅能执行一次。
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 在实例上初始化一个私有集合，用于存储已执行的方法名
        if not hasattr(self, '_executed_methods'):
            # 使用 object.__setattr__ 确保在 __init__ 执行前就能安全注入属性
            object.__setattr__(self, '_executed_methods', set())
        
        # 检查当前方法是否已经执行过
        if func.__name__ in self._executed_methods:
            error_msg = f"❌ 违反单次执行策略：方法 '{self.__class__.__name__}.{func.__name__}' 只能被执行一次！"
            logger.warning(error_msg)
            return
            
        # 记录并执行
        self._executed_methods.add(func.__name__)
        return func(self, *args, **kwargs)
    return wrapper

# 内部私有变量，对外部模块隐藏
_GLOBAL_GRAPH: Optional[Any] = None

def set_global_graph(graph: Any) -> None:
    """
    设置本模块的长生命周期全局 Graph 实例。
    通常在系统启动（App Startup）或主图创建完毕后，由主入口调用一次。
    """
    global _GLOBAL_GRAPH
    if graph is None:
        raise ValueError("全局 Graph 不能设置为 None")
        
    # 如果你想防止重复初始化，可以解开下方注释：
    if _GLOBAL_GRAPH is not None:
        import logging
        logging.warning("全局 Graph 已被初始化，正在覆盖旧实例。")
        
    _GLOBAL_GRAPH = graph


def get_global_graph() -> Any:
    """
    获取本模块的全局 Graph 实例。
    如果后台服务在未初始化图时就启动，会抛出明确的系统级错误。
    """
    global _GLOBAL_GRAPH
    if _GLOBAL_GRAPH is None:
        raise RuntimeError(
            "模块未初始化：请在启动后台 Worker 之前，"
            "调用 `set_global_graph(graph)` 注入主图实例。"
        )
    return _GLOBAL_GRAPH


class SchedulerOrchestrator(metaclass=SingletonMeta):
    """
    统一生命周期管理类：负责调度内核初始化、外部类异步热加载、
    依赖强一致性注入拦截以及业务任务的统一下发。
    """
    def __init__(self):
        store = PostgresStore(conn=ConnectionPool(conninfo=DEFAULT_DB_URI))
        self.llm = ModelProvider.get_falback_model()
        self.graph = get_global_graph()
        # 初始化底层的通用旁路调度内核
        self.scheduler = ConfigurationMergedScheduler(store_client=store)
        logger.info("🏗️ 调度编排器初始化成功，物理存储客户端已就位。")
        self.start_engine()

    @run_once
    def start_engine(self):
        """
        步骤 1: 启动调度时钟引擎（此时引擎空转，无任何任务类）
        """
        logger.info("🛫 正在激活后台时钟线...")
        self.scheduler.start()

    @run_once
    def load_and_register_tasks(self, user_id: str,thread_id:str):
        """
        步骤 2: 动态加载并注册外部任务，同时进行强类型注入检测
        """
        logger.info("🔌 开始跨文件热挂载业务组件...")
        
        # 准备所需的真实运行时依赖实例
        llm_runtime = MemoryTaskRuntime(
                llm=self.llm,
                graph=self.graph,
                thread_id=thread_id,
                user_id=user_id
            )
        
        try:
            self.scheduler.register_task_type(
                task_cls=SystemMonitorTask, 
                runtime_handle=SystemTaskRuntime()
            )
            # 动态注册从 tasks/episodic_task.py 引用过来的任务类
            self.scheduler.register_task_type(
                task_cls=ProfileMemoryTask, 
                runtime_handle=llm_runtime
            )
            self.scheduler.register_task_type(
                task_cls=EpisodicMemoryTask, 
                runtime_handle=llm_runtime
            )
            self.scheduler.register_task_type(
                task_cls=SemanticMemoryTask, 
                runtime_handle=llm_runtime
            )
        except TypeError as e:
            logger.error("🛑 编排器捕获到非法依赖注入: %s", e)
            raise e
    @run_once
    def dispatch_user_mission(self, user_id: str):
        """
        步骤 3: 随时下发生产业务任务
        """
        logger.info("🎯 收到前端/API 业务请求，开始下派具体任务实例...")
        self.scheduler.clear_all_jobs()

        for task_type in ["sys_monitor", "profile", "episodic", "semantic"]:
            self.scheduler.add_job(task_type=task_type, target_id=user_id)

    @run_once
    def stop_engine(self):
        """
        步骤 4: 优雅关闭
        """
        self.scheduler.shutdown()
        

# if __name__ == "__main__":
    
#     store = PostgresStore(conn=ConnectionPool(conninfo=DEFAULT_DB_URI))
#     engine = ConfigurationMergedScheduler(store_client=store)
#     engine.start()