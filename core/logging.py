# logging.py
import logging
import sys
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.args and isinstance(record.args, dict):
            log_record.update(record.args)  # datos extra
        return json.dumps(log_record)

def setup_logger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Evitar múltiples handlers duplicados
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    return root
