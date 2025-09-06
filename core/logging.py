# core/logging.py
import logging
import sys
from loguru import logger
from typing import Optional, Dict, Any
import json
import csv
import sqlite3
from datetime import datetime
from pathlib import Path

# -------------------------
# InterceptHandler
# -------------------------
class InterceptHandler(logging.Handler):
    """Intercepta logging estándar y lo envía a loguru"""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

# -------------------------
# Logger setup
# -------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

JSON_LOG_FILE = LOG_DIR / "events.json"
CSV_LOG_FILE = LOG_DIR / "events.csv"
SQLITE_DB_FILE = LOG_DIR / "logs.db"

def setup_logger(level: int = logging.INFO):
    """Configura logger centralizado"""
    # --- loguru ---
    logger.remove()  # quitar handlers por defecto
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}", level=level)
    
    # --- logging estándar ---
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

    # --- SQLite init ---
    conn = sqlite3.connect(SQLITE_DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            ts TEXT,
            level TEXT,
            module TEXT,
            message TEXT,
            cycle_id TEXT,
            symbol TEXT,
            extra TEXT
        )
    """)
    conn.commit()
    conn.close()

    return logger

# -------------------------
# Global logger
# -------------------------
logger = setup_logger()

# -------------------------
# Helpers para eventos
# -------------------------
def log_event(
    level: str,
    msg: str,
    module: str = __name__,
    cycle_id: Optional[str] = None,
    symbol: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
):
    """
    Registra un evento en consola, JSON, CSV y SQLite.
    """
    # --- loguru console ---
    logger.log(level.upper(), msg)

    # --- JSON append ---
    json_entry = {
        "ts": datetime.utcnow().isoformat(),
        "level": level.upper(),
        "module": module,
        "message": msg,
        "cycle_id": cycle_id,
        "symbol": symbol,
        "extra": extra
    }
    with open(JSON_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_entry) + "\n")

    # --- CSV append ---
    write_csv = not CSV_LOG_FILE.exists()
    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=json_entry.keys())
        if write_csv:
            writer.writeheader()
        writer.writerow(json_entry)

    # --- SQLite insert ---
    conn = sqlite3.connect(SQLITE_DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (ts, level, module, message, cycle_id, symbol, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        json_entry["ts"],
        json_entry["level"],
        json_entry["module"],
        json_entry["message"],
        json_entry["cycle_id"],
        json_entry["symbol"],
        json.dumps(json_entry["extra"]) if json_entry["extra"] else None
    ))
    conn.commit()
    conn.close()

# -------------------------
# Shortcuts para niveles
# -------------------------
def debug(msg, **kwargs):
    log_event("DEBUG", msg, **kwargs)

def info(msg, **kwargs):
    log_event("INFO", msg, **kwargs)

def warning(msg, **kwargs):
    log_event("WARNING", msg, **kwargs)

def error(msg, **kwargs):
    log_event("ERROR", msg, **kwargs)

def critical(msg, **kwargs):
    log_event("CRITICAL", msg, **kwargs)

# -------------------------
# Cycle Data Logging
# -------------------------
async def log_cycle_data(state: Dict[str, Any], cycle_id: int, cycle_start: datetime):
    """Log cycle portfolio and metrics"""
    # Portfolio value
    portfolio = state.get('portfolio', {})
    total_value = portfolio.get('total', 0)
    logger.debug(f"💰 Valor total del portfolio (ciclo {cycle_id}): {total_value:.2f} USDT")
    
    # Cycle stats
    cycle_stats = state.get('cycle_stats', {})
    signals_count = cycle_stats.get('signals_count', 0)
    orders_count = cycle_stats.get('orders_count', 0)
    rejected_count = cycle_stats.get('rejected_orders', 0)

    # Safe cycle_time calc: support pandas.Timestamp and datetime, tz-aware and naive
    def _to_naive_dt(dt):
        try:
            # pandas Timestamp has tz_localize / tz_convert
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
            if dt.tzinfo is not None:
                # convert to UTC naive
                return dt.astimezone().replace(tzinfo=None)
            return dt
        except Exception:
            return None

    cycle_time = cycle_stats.get('cycle_time')
    if cycle_time is None:
        start_dt = _to_naive_dt(cycle_start)
        now_dt = _to_naive_dt(datetime.utcnow())
        try:
            if start_dt is not None:
                cycle_time = (now_dt - start_dt).total_seconds()
            else:
                cycle_time = 0.0
        except Exception:
            cycle_time = 0.0
    
    # Strategy info
    strategy_keys = state.get("estrategia", {}).keys()
    
    # Log cycle summary
    log_event(
        "INFO",
        f"📊 Ciclo {cycle_id} completado en {cycle_time:.2f}s con {signals_count} señales y {orders_count} órdenes ({rejected_count} rechazadas) | " +
        f"Estrategia keys: {list(strategy_keys)}",
        module="core.logging",
        cycle_id=str(cycle_id),
        extra={
            'total_value': total_value,
            'signals_count': signals_count,
            'orders_count': orders_count,
            'rejected_count': rejected_count,
            'cycle_time': cycle_time,
            'strategy_keys': list(strategy_keys)
        }
    )

# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    info("Logger inicializado", cycle_id="test_cycle", symbol="BTCUSDT", extra={"test": True})
    debug("Debug message")
    warning("Warning message")
    error("Error message")
