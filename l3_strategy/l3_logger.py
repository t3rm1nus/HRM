# l3_logger.py
# -*- coding: utf-8 -*-
"""
Logger y métrica centralizada para L3 (HRM)
-------------------------------------------
- Persistencia de logs en JSON Lines y en consola
- Decorador para medir tiempo de ejecución
- Registro de métricas de desempeño estratégico
- Compatible con otros módulos (sentiment, regime, volatility, portfolio)
"""

import os
import json
import time
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional

# =========================
# Configuración
# =========================

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "l3_metrics.jsonl")

# =========================
# Logger real de Python para L3
# =========================
# Usamos un logger hijo de "HRM.L3"
_logger = logging.getLogger("HRM.L3")
if not _logger.hasHandlers():
    _logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    _logger.addHandler(ch)

_logger.info("HRM.L3 iniciado")

# =========================
# Utilidades
# =========================

def _persist_jsonl(record: Dict[str, Any], file_path: str = LOG_FILE):
    """Añade un dict como JSON line en el archivo especificado"""
    record["_timestamp"] = datetime.utcnow().isoformat()
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        _logger.warning(f"No se pudo persistir JSONL: {e}")


def log_metric(
    model: str,
    metric_name: str,
    value: float,
    extra: Optional[Dict[str, Any]] = None
):
    record = {"model": model, "metric": metric_name, "value": value}
    if extra:
        record.update(extra)
    _logger.info(f"[{model}] {metric_name}={value}")
    _persist_jsonl(record)


def log_output(model: str, output: Dict[str, Any]):
    record = {"model": model, "output": output}
    _logger.info(f"[{model}] Output registrado.")
    _persist_jsonl(record)


def timeit(model: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                _logger.info(f"[{model}] Tiempo ejecución: {elapsed:.2f}s")
                _persist_jsonl({
                    "model": model,
                    "metric": "execution_time_sec",
                    "value": elapsed
                })
        return wrapper
    return decorator


# =========================
# Ejemplo de uso
# =========================

if __name__ == "__main__":

    @timeit("volatility")
    def fake_volatility_forecast():
        time.sleep(1.2)
        return {"BTC-USD": 0.35, "ETH-USD": 0.42}

    output = fake_volatility_forecast()
    log_output("volatility", output)
    log_metric("volatility", "mse", 0.023, extra={"asset": "BTC-USD"})
    log_metric("sentiment", "accuracy", 0.81)
