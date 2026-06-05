import abc
import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type

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


# ==============================================================================
# 强类型运行时基类 (基础设施依赖全内聚于此)
# ==============================================================================
class BaseTaskRuntime:
    """
    所有任务运行时的基类。
    不同的任务集群可以派生自己的强类型 Runtime，并在内部存放各自需要的重型对象（LLM、RPC Client等）。
    """
    pass


# ==============================================================================
# 任务单元基类
# ==============================================================================
class BaseTaskUnit(abc.ABC):
    """
    带默认配置、自包含实例化工厂与强类型依赖注入的通用任务基类。
    """
    task_type: str = ""
    default_cron: str = "0 0 * * *"
    default_params: Dict[str, Any] = {}
    
    # 💡 核心约束：声明此任务类型期望得到的强类型运行时容器类
    runtime_schema: Type[BaseTaskRuntime] = BaseTaskRuntime

    def __init__(self, runtime: Any):
        # ⚡️ 架构升级：强制要求在实例化时注入专属的运行时环境
        self.runtime: Any = runtime
        
        self.task_id: str = ""
        self.target_id: str = ""
        self.cron_expr: str = ""
        self.params: Dict[str, Any] = {}

    @classmethod
    def create_instance(
        cls, 
        store_client: Any, 
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

        internal_task_id = f"job_{cls.task_type}_{target_id}"
        # 严格验证外部传入的自定义参数，必须属于 default_params 中声明过的键，防止业务端写错
        if params:
            for key in params.keys():
                if key not in cls.default_params:
                    raise KeyError(f"❌ 参数错误: '{key}' 不是任务 {cls.task_type} 允许的合法业务参数项")

        # 组装满足调度内核所需的标准持久化载荷
        plan_payload: Dict[str, Any] = {"target_id": target_id}
        
        if cron_expr:
            plan_payload["cron_expr"] = cron_expr
            
        if params:
            # 🔒 架构升级：入库前防呆检查，拒绝不可序列化的复杂对象（如 LLM Client、锁、实体类）
            try:
                json.dumps(params)
            except TypeError as e:
                raise ValueError(
                    f"❌ 任务下发失败: params 字典中包含了无法被持久化存储的复杂 Python 对象！\n"
                    f"报错详情: {str(e)}。\n"
                    f"请将对象转换为纯 Dict 序列化数据，或将重型对象挂载至 Runtime 环境中。"
                )
            plan_payload["params"] = params

        payload = {
            "task_type": cls.task_type,
            "status": "SCHEDULED",  # 统一由工厂赋予初始状态
            "plan": plan_payload,
            "created_at": datetime.now().isoformat()
        }

        # 统一封装物理存储路径约束
        ns: Tuple[str, ...] = ("sys", "scheduler", "tasks")
        store_client.put(namespace=ns, key=internal_task_id, value=payload)

    @abc.abstractmethod
    def should_trigger(self, store_client: Any) -> bool:
        """准入控制流：返回 False 时优雅熔断本次触发"""
        pass

    @abc.abstractmethod
    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """全异步核心业务逻辑执行入口"""
        pass