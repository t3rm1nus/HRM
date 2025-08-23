# telemetry.py
import time
from collections import defaultdict

class Telemetry:
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.timings = []

    def incr(self, metric: str):
        pass

    def timing(self, metric: str, start_time: float):
        pass

    def gauge(self, metric: str, value: float):
        pass
    
    def snapshot(self):
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timings": self.timings[-10:],  # últimos 10
        }

telemetry = Telemetry()
