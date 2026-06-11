import logging
from agi.scheduler.task_hub import AgiTaskHub
from typing import Optional,Any,Dict
import threading
logger = logging.getLogger(__name__)

class ExternalStateBridge:
    """
    线程安全的外部状态桥接器。
    专门应对“参数需要在另一个线程运行之后动态获取”的场景。
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}

    def update_dynamic_deps(self, key: str, value: Any):
        """供另一个线程异步/同步计算完毕后，安全地刷新依赖参数"""
        with self._lock:
            self._data[key] = value
            logger.info("⚡ [Bridge] 另一个线程已成功向上游注入动态运行时依赖: %s=%s", key,value)

    def get_value(self, key: str) -> Any:
        """线程安全地读取当前最新值"""
        with self._lock:
            return self._data.get(key)
        
runtime_state_bridge = ExternalStateBridge()

hub = AgiTaskHub()
