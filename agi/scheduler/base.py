import abc
from datetime import datetime
from typing import Any, Dict, Optional,Tuple

# ==============================================================================
# 核心数据契约
# ==============================================================================
class ExecutionPlan:
    """定义任务实例的执行载荷，业务层只需提供 target_id，其余可选覆盖"""
    def __init__(self, target_id: str, cron_expr: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.target_id: str = target_id
        self.cron_expr: Optional[str] = cron_expr
        self.params: Dict[str, Any] = params or {}


class TaskExecutionResult:
    """标准任务异步业务执行完毕后的输出契约"""
    def __init__(self, is_success: bool, output_data: Any = None, error_log: Optional[str] = None):
        self.is_success: bool = is_success
        self.output_data: Any = output_data
        self.error_log: Optional[str] = error_log
        self.finished_at: datetime = datetime.now()


class BaseTaskUnit(abc.ABC):
    """
    带默认配置与自包含实例化工厂的通用任务基类。
    """
    task_type: str = ""
    default_cron: str = "0 0 * * *"
    default_params: Dict[str, Any] = {}

    def __init__(self):
        self.task_id: str = ""
        self.target_id: str = ""
        self.cron_expr: str = ""
        self.params: Dict[str, Any] = {}

    @classmethod
    def create_instance(
        cls, 
        store_client: Any, 
        task_id: str, 
        target_id: str, 
        cron_expr: Optional[str] = None, 
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        🚀 封装核心：统一的任务分发工厂。
        直接双写进你的 BaseStore，业务方无需关心底层的存储结构和命名空间。
        """
        if not cls.task_type:
            raise ValueError(f"类 {cls.__name__} 未定义有效的静态 task_type")

        # 严格验证外部传入的自定义参数，必须属于 default_params 中声明过的键，防止业务端写错
        if params:
            for key in params.keys():
                if key not in cls.default_params:
                    raise KeyError(f"❌ 参数错误: '{key}' 不是任务 {cls.task_type} 允许的合法业务参数项")

        # 组装满足调度内核所需的标准持久化载荷
        plan_payload: Dict[str, Any] = {"target_id": target_id}
        
        # 只有在明确需要覆盖默认时钟/参数时，才往数据库里写，保持存储极简
        if cron_expr:
            plan_payload["cron_expr"] = cron_expr
        if params:
            plan_payload["params"] = params

        payload = {
            "task_type": cls.task_type,
            "status": "SCHEDULED",  # 统一由工厂赋予初始状态
            "plan": plan_payload,
            "created_at": datetime.now().isoformat()
        }

        # 统一封装物理存储路径约束
        ns: Tuple[str, ...] = ("sys", "scheduler", "tasks")
        store_client.put(namespace=ns, key=task_id, value=payload)

    @abc.abstractmethod
    def should_trigger(self, store_client: Any) -> bool:
        pass

    @abc.abstractmethod
    async def execute(self, store_client: Any) -> Any:
        pass