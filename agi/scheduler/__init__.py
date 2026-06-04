from agi.scheduler.scheduler import ConfigurationMergedScheduler
from agi.scheduler.tasks import *
from langgraph.store.postgres import PostgresStore
from psycopg_pool import  ConnectionPool
from agi.config import DEFAULT_DB_URI


if __name__ == "__main__":
    
    store = PostgresStore(conn=ConnectionPool(conninfo=DEFAULT_DB_URI))
    engine = ConfigurationMergedScheduler(store_client=store)
    engine.start()