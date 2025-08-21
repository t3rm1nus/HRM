"""
Capa de métricas centralizada.
Incluye:
- Latencia de órdenes
- Rechazos / fallas / parciales
- Exposición de portafolio
"""

import threading

_lock = threading.Lock()
_METRICS = {
    "orders_success": 0,
    "orders_rejected": 0,
    "orders_failed": 0,
    "orders_partial": 0,
    "latency_ms": [],
    "portfolio": {},
}


def increment(key: str, value: int = 1):
    with _lock:
        if key in _METRICS:
            _METRICS[key] += value
        else:
            _METRICS[key] = value


def observe_latency(latency_ms: float):
    with _lock:
        _METRICS["latency_ms"].append(latency_ms)


async def update_portfolio(symbol: str):
    from l1_operational import portfolio
    try:
        bal, quote = await portfolio.get_available_balance(symbol)
        with _lock:
            _METRICS["portfolio"][symbol] = {"base": bal, "quote": quote}
    except Exception:
        pass


def snapshot() -> dict:
    with _lock:
        return {k: (v if not isinstance(v, list) else list(v)) for k, v in _METRICS.items()}
