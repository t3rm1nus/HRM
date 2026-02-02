# core/logging.py
import logging
import sys
from loguru import logger
from typing import Optional, Dict, Any
import json
import csv
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

def setup_logger(level: int = logging.INFO):
    """Configura logger centralizado con inicialización robusta de archivos de logging"""
    # --- loguru ---
    logger.remove()  # quitar handlers por defecto
    # Formato con color: CRITICAL en violeta y bold para TODO el mensaje
    logger.add(
        sys.stderr,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "<level>{level}</level> | {name} | "
            "{message}"
        ),
        level=level,
        colorize=True,
        filter=lambda record: record["level"].name != "CRITICAL"
    )

    # Handler separado para CRITICAL con todo el mensaje en violeta
    logger.add(
        sys.stderr,
        format="\x1b[95m\x1b[1m{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}\x1b[0m",
        level="CRITICAL",
        colorize=False
    )

    # --- logging estándar ---
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

    # --- Crear directorios de log si no existen ---
    LOG_DIR.mkdir(exist_ok=True)

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
    extra: Optional[Dict[str, Any]] = None,
    exc_info: Optional[bool] = None
):
    """
    Registra un evento en consola, JSON y CSV.
    """
    try:
        # --- loguru console ---
        if exc_info and sys.exc_info()[0] is not None:
            logger.opt(depth=6, exception=True).log(level.upper(), msg)
        else:
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
            # Ensure json_entry is a dict before calling keys()
            if isinstance(json_entry, dict):
                try:
                    fieldnames = list(json_entry.keys())
                except AttributeError:
                    fieldnames = []
            else:
                fieldnames = []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_csv:
                writer.writeheader()
            # Ensure json_entry is still a dict before writing
            if isinstance(json_entry, dict):
                writer.writerow(json_entry)

    except Exception as log_error:
        # Fallback logging to avoid recursive errors
        print(f"[LOGGING ERROR] {log_error} - Original message: {msg}")
        # Don't use logger here to avoid recursion

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

# ------------------------- Estandarizado de trading actions log
# -------------------------
def log_trading_action(symbol: str, strategy: str, regime: str = None,
                      action: str = None, confidence: float = None,
                      reason: str = None, **kwargs):
    """
    Log trading action in standardized format:

    🎯 [BTC] TREND-FOLLOWING:
       Regime: BULL
       Action: BUY (conf=0.78)
       Reason: Price above MA50+MA200 + strong strength score

    Args:
        symbol: Trading symbol (e.g., 'BTC', 'ETH')
        strategy: Strategy name (e.g., 'TREND-FOLLOWING', 'MEAN-REVERSION')
        regime: Market regime (e.g., 'BULL', 'BEAR', 'RANGE')
        action: Trading action (e.g., 'BUY', 'SELL', 'HOLD')
        confidence: Confidence score (0.0 to 1.0)
        reason: Detailed reason for the action
        **kwargs: Additional logging parameters
    """
    try:
        # Extract short symbol name (remove USDT if present)
        short_symbol = symbol.replace("USDT", "") if isinstance(symbol, str) else str(symbol)

        # Build the formatted message
        message_parts = [f"🎯 [{short_symbol}] {strategy.upper()}:"]
        if regime:
            message_parts.append(f"   Regime: {regime.upper()}")
        if action:
            conf_str = f" (conf={confidence:.2f})" if confidence is not None else ""
            message_parts.append(f"   Action: {action.upper()}{conf_str}")
        if reason:
            message_parts.append(f"   Reason: {reason}")

        formatted_message = "\n".join(message_parts)

        # Log using the centralized logger
        log_event("INFO", formatted_message, symbol=symbol, extra={
            'trading_action': {
                'symbol': symbol,
                'strategy': strategy,
                'regime': regime,
                'action': action,
                'confidence': confidence,
                'reason': reason
            },
            **kwargs
        })

    except Exception as e:
        # Fallback to regular logging if something goes wrong
        log_event("ERROR", f"Failed to log trading action: {e}")

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
    estrategia = state.get("estrategia", {})
    if isinstance(estrategia, dict):
        strategy_keys = estrategia.keys()
    else:
        strategy_keys = []
    
    # Log cycle summary separating intent (BUY/SELL) from HOLD states
    # HOLD signals are tactical/strategic states, not intent signals
    log_event(
        "INFO",
        f"📊 Ciclo completado | intent=0 | actionable=0 | orders={orders_count} | " +
        f"Estrategia keys: {list(strategy_keys)}",
        module="core.logging",
        cycle_id=str(cycle_id),
        extra={
            'total_value': total_value,
            'intent_signals': 0,  # HOLD signals don't count as intent
            'actionable_signals': 0,  # Only BUY/SELL signals are actionable
            'tactical_holds': signals_count,  # Track HOLD signals separately
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
