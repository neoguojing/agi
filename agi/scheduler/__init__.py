from agi.scheduler.scheduler import ConfigurationMergedScheduler
from agi.scheduler.tasks import *



if __name__ == "__main__":
    
    engine = ConfigurationMergedScheduler(store_client=None)
    engine.start()