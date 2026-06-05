from agi.scheduler.scheduler import ConfigurationMergedScheduler
from agi.scheduler.memory_task.memory_task import *
from langgraph.store.postgres import PostgresStore
from psycopg_pool import  ConnectionPool
from agi.config import DEFAULT_DB_URI


def run_background_worker(graph):
    # 初始化内核
    store = PostgresStore(conn=ConnectionPool(conninfo=DEFAULT_DB_URI))
    engine = ConfigurationMergedScheduler(store_client=store)
    
    engine.task_slots = {
        "profile": ProfileMemoryTask,
        "semantic": SemanticMemoryTask
    }

    # 3. ⚡️【核心重构体现】：为不同的任务，精准配发专属的强类型运行时依赖容器
    # 即使多个任务都在跑，它们的运行时物理隔离，互不干扰，Store 里也完全没有这些脏对象
    engine.runtime_slots = {
        # 画像任务：分发便宜、低延迟的运行时容器
        "profile": MemoryTaskRuntime(llm=None,graph=graph),
        "episodic": MemoryTaskRuntime(llm=None,graph=graph),
        # 语义图谱任务：分发高精度、带有嵌入向量支持的重型运行时容器
        "semantic": MemoryTaskRuntime(llm=None,graph=graph)
    }

    # 🛠️ 手动登记当前节点支持的记忆抽取原子能力
    engine.task_slots = {
        "profile": ProfileMemoryTask,
        "episodic": EpisodicMemoryTask,
        "semantic": SemanticMemoryTask
    }
    
    # 启动！自驱去底层 sys/scheduler/tasks 捞取所有历史 SCHEDULED 实例
    # 如果捞出来的是 episodic 实例，则自动合并 10 分钟 Cron，拉起时钟线并安全执行循环。
    engine.start()

if __name__ == "__main__":
    
    store = PostgresStore(conn=ConnectionPool(conninfo=DEFAULT_DB_URI))
    engine = ConfigurationMergedScheduler(store_client=store)
    engine.start()