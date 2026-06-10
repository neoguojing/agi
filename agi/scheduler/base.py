import abc
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type

logger = logging.getLogger("SchedulerKernel")

# ==============================================================================
# 1. 统一存储数据契约
# ==============================================================================
class SchedulerStorageContract:
    TASK_NAMESPACE: Tuple[str, ...] = ("sys", "scheduler", "tasks")

    @classmethod
    def generate_task_key(cls, task_type: str, target_id: str) -> str:
        return f"job_{target_id}_{task_type}"


class TaskExecutionResult:
    """标准任务异步业务执行完毕后的输出标准件"""
    def __init__(self, is_success: bool, output_data: Any = None, error_log: Optional[str] = None):
        self.is_success: bool = is_success
        self.output_data: Any = output_data
        self.error_log: Optional[str] = error_log
        self.finished_at: datetime = datetime.now()


# ==============================================================================
# 2. 强类型依赖注入基类（核心约束项）
# ==============================================================================
class BaseTaskRuntime(abc.ABC):
    """
    所有任务运行时的基础设施依赖容器基类。
    具体的业务依赖（如 LLM 客户端、DB 桥接器）必须继承此类。
    """
    pass


# ==============================================================================
# 3. 任务单元基类（声明依赖契约）
# ==============================================================================
class BaseTaskUnit(abc.ABC):
    """
    自包含规则定义的通用任务基类。
    通过 runtime_schema 强制约束当前任务必须注入哪种 Runtime 容器。
    """
    task_type: str = ""
    default_cron: str = "0 0 * * *"
    default_params: Dict[str, Any] = {}
    
    # 💡 核心约束蓝图：子类通过覆盖此属性，指定自己需要的依赖类型
    runtime_schema: Type[BaseTaskRuntime] = BaseTaskRuntime

    def __init__(self, runtime: BaseTaskRuntime, task_id: str, target_id: str, params: dict):
        self.runtime: BaseTaskRuntime = runtime  # 已经通过强类型校验的合规依赖
        self.task_id: str = task_id
        self.target_id: str = target_id
        self.params: Dict[str, Any] = params

        if not self.task_id:
            logger.warning(f"⚠️  [Invalid State] 实例化了 ID 为空的任务单元! ObjAddr: {id(self)}")


    @abc.abstractmethod
    async def should_trigger(self, store_client: Any) -> bool:
        """准入控制流"""
        pass

    @abc.abstractmethod
    async def execute(self, store_client: Any) -> TaskExecutionResult:
        """全异步核心业务逻辑执行入口"""
        pass