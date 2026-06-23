import logging
from time import perf_counter
from datetime import datetime

class ElapsedFormatter(logging.Formatter):
    def __init__(self, fmt=None):
        super().__init__(fmt)
        self.start = perf_counter()

    def formatTime(self, record, datefmt=None):
        elapsed = int(perf_counter() - self.start)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
class CombinedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None):
        super().__init__(fmt)
        self.start = perf_counter()

    def formatTime(self, record, datefmt=None):
        # Current time
        now = datetime.now().strftime("%H:%M:%S")

        # Elapsed time
        elapsed = int(perf_counter() - self.start)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

        return f"{now} ({elapsed_str})"