import abc
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type

logger = logging.getLogger("SchedulerKernel")

# ==============================================================================
# 1. 统一命名空间与存储约束 (解决痛点 3)
# ==============================================================================
class SchedulerStorageContract:
    """统一管理调度器在持久化存储中的命名空间与 Key 生成规则"""
    TASK_NAMESPACE: Tuple[str, ...] = ("sys", "scheduler", "tasks")

    @classmethod
    def generate_task_key(cls, task_type: str, target_id: str) -> str:
        """统一分布式唯一 Key 生成器"""
        return f"job_{task_type}_{target_id}"


# ==============================================================================
# 2. 强类型数据契约 (解决痛点 1 & 2)
# ==============================================================================
class ExecutionPlan:
    """
    定义任务实例的执行载荷。
    正式启用作为入参契约，内聚业务目标、时钟表达式与强约束参数。
    """
    def __init__(self, target_id: str, cron_expr: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        if not target_id or not isinstance(target_id, str):
            raise ValueError("❌ 契约错误: target_id 不能为空且必须为字符串")
        
        self.target_id: str = target_id
        self.cron_expr: Optional[str] = cron_expr
        self.params: Dict[str, Any] = params or {}

    def to_dict(self) -> Dict[str, Any]:
        """将契约安全转换为可落库的干净字典"""
        payload = {"target_id": self.target_id}
        if self.cron_expr:
            payload["cron_expr"] = self.cron_expr
        if self.params:
            payload["params"] = self.params
        return payload


class TaskExecutionResult:
    """标准任务异步业务执行完毕后的输出契约"""
    def __init__(self, is_success: bool, output_data: Any = None, error_log: Optional[str] = None):
        self.is_success: bool = is_success
        self.output_data: Any = output_data
        self.error_log: Optional[str] = error_log
        self.finished_at: datetime = datetime.now()


class BaseTaskRuntime:
    """所有任务运行时的基类（基础设施依赖容器）"""
    pass


# ==============================================================================
# 3. 任务单元基类
# ==============================================================================
class BaseTaskUnit(abc.ABC):
    """
    带默认配置、自包含实例化工厂与强类型依赖注入的通用任务基类。
    """
    task_type: str = ""
    default_cron: str = "0 0 * * *"
    
    # 💡 强约束蓝图：子类定义的默认值不仅作为缺省，也作为参数类型强校验的依据
    default_params: Dict[str, Any] = {}
    runtime_schema: Type[BaseTaskRuntime] = BaseTaskRuntime

    def __init__(self, runtime: Any):
        self.runtime: Any = runtime
        self.task_id: str = ""
        self.target_id: str = ""
        self.cron_expr: str = ""
        self.params: Dict[str, Any] = {}

    @classmethod
    def create_instance(cls, store_client: Any, plan: ExecutionPlan) -> None:
        """
        🚀 统一任务分发工厂
        已完全切换至基于 ExecutionPlan 契约驱动，并对内部参数实施静态强约束校验。
        """
        if not cls.task_type:
            raise ValueError(f"类 {cls.__name__} 未定义有效的静态 task_type")

        # 🔒 强约束拦截点 1：校验参数合法性与类型强一致性
        for key, val in plan.params.items():
            if key not in cls.default_params:
                raise KeyError(f"❌ 参数越界: '{key}' 不是任务 [{cls.task_type}] 允许的业务参数项")
            
            expected_type = type(cls.default_params[key])
            if not isinstance(val, expected_type):
                raise TypeError(
                    f"❌ 参数类型错误: 项 '{key}' 期望类型为 {expected_type.__name__}, "
                    f"但实际传入了 {type(val).__name__}。"
                )

        # 🔒 强约束拦截点 2：拒绝不可序列化的复杂 Python 对象落库
        try:
            json.dumps(plan.params)
        except TypeError as e:
            raise ValueError(f"❌ 序列化失败: params 字典中包含无法持久化的复杂对象! 错误原因: {str(e)}")

        # 组合物理存储
        internal_task_id = SchedulerStorageContract.generate_task_key(cls.task_type, plan.target_id)
        
        payload = {
            "task_type": cls.task_type,
            "status": "SCHEDULED",
            "plan": plan.to_dict(),
            "created_at": datetime.now().isoformat()
        }

        # 统一使用名字空间常量契约，消除硬编码
        store_client.put(
            namespace=SchedulerStorageContract.TASK_NAMESPACE, 
            key=internal_task_id, 
            value=payload
        )

    @abc.abstractmethod
    def should_trigger(self, store_client: Any) -> bool:
        """准入控制流"""
        pass

    @abc.abstractmethod
    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """全异步核心业务逻辑执行入口"""
        pass