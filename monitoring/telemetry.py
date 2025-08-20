# telemetry.py
import time
from collections import defaultdict

class Telemetry:
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = {}
        self.timings = []

    def incr(self, name: str, val: int = 1):
        self.counters[name] += val

    def gauge(self, name: str, val: float):
        self.gauges[name] = val

    def timing(self, name: str, start_ts: float):
        elapsed = time.time() - start_ts
        self.timings.append((name, elapsed))

    def snapshot(self):
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timings": self.timings[-10:],  # últimos 10
        }

telemetry = Telemetry()
