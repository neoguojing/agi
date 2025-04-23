import time
from agi.config import log

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        log.info(f"执行时间: {self.interval:.4f} 秒")