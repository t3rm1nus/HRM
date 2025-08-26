# main.py
import asyncio
import sys
import time
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Coroutine, Optional

import pandas as pd

# ---- Core / Monitoring / Storage ----
from core.logging import setup_logger
from monitoring.telemetry import telemetry
from monitoring.dashboard import render_dashboard
from storage import guardar_estado_csv, guardar_estado_sqlite

# ---- Pipeline levels ----
from l4_meta import procesar_l4
from l3_strategy import procesar_l3
from l1_operational import procesar_l1

# ---- L2 Tactic (async) ----
from l2_tactic.config import L2Config
from l2_tactic.main_processor import L2MainProcessor

# ---- Bus adapter (requiere un bus) ----
from l1_operational.bus_adapter import BusAdapterAsync

from l2_tactic.technical.patterns import detect_all as detect_patterns
from l2_tactic.technical.multi_timeframe import resample_and_consensus
from l2_tactic.technical.support_resistance import swing_pivots


# ------------------------------------------------------------
# Configuración de logging
# ------------------------------------------------------------
logger = setup_logger(level=logging.DEBUG)

# ------------------------------------------------------------
# Bus mínimo en memoria para trabajar con BusAdapterAsync
# ------------------------------------------------------------
class LocalMessageBus:
    """
    Bus pub/sub muy simple y compatible con BusAdapterAsync.
    - subscribe(topic, handler)
    - publish(topic, msg) (async)
    - publish_sync(topic, msg) (sync helper)
    """
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], Any]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], Any]) -> None:
        self._subs[topic].append(handler)

    async def publish(self, topic: str, message: Any) -> None:
        for h in list(self._subs.get(topic, [])):
            try:
                if asyncio.iscoroutinefunction(h):
                    await h(message)
                else:
                    # ejecutar en thread para no bloquear el loop
                    await asyncio.to_thread(h, message)
            except Exception:
                logger.exception(f"[BUS] Error enviando mensaje a handler de '{topic}'")

    # Algunos adaptadores llaman publish en modo sync
    def publish_sync(self, topic: str, message: Any) -> None:
        for h in list(self._subs.get(topic, [])):
            try:
                if asyncio.iscoroutinefunction(h):
                    asyncio.create_task(h(message))
                else:
                    h(message)
            except Exception:
                logger.exception(f"[BUS] Error enviando mensaje sync a handler de '{topic}'")


# Variables globales necesarias para funciones auxiliares
BUS: Optional[BusAdapterAsync] = None
CONFIG_L2: L2Config = L2Config()


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def validate_portfolio(portfolio: Dict[str, float], valid_symbols: List[str], stage: str) -> None:
    invalid = {symbol for symbol in portfolio if symbol not in valid_symbols}
    if invalid:
        logger.error(f"Invalid symbols in portfolio at {stage}: {', '.join(invalid)}")
        raise ValueError(f"Invalid symbols in portfolio at {stage}: {', '.join(invalid)}")


def _calc_portfolio_value(state: dict) -> float:
    """
    Calcula el valor del portafolio usando el último 'close' disponible por símbolo.
    Espera state['mercado'][symbol] como DataFrame con columna 'close'.
    """
    total = 0.0
    for a, qty in state["portfolio"].items():
        df = state["mercado"].get(a)
        if df is None or df.empty or "close" not in df.columns:
            price = 0.0
        else:
            price = float(df["close"].iloc[-1])
        total += float(qty) * price
    return total


# ------------------------------------------------------------
# TICK principal (una iteración del pipeline)
# ------------------------------------------------------------
async def run_tick(state: dict) -> dict:
    global BUS, CONFIG_L2

    start = time.time()
    state["ciclo_id"] = int(state.get("ciclo_id", 0)) + 1
    state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info(f"[TICK] Iniciando ciclo {state['ciclo_id']}")
    logger.debug(f"[MAIN] Portfolio antes de L4: {state['portfolio']}")

    # ---------------- L4 ----------------
    logger.info("[L4] Ejecutando capa Meta...")
    state = procesar_l4(state)
    logger.debug(f"[L4] Portfolio después de L4: {state['portfolio']}")
    validate_portfolio(state["portfolio"], list(state["mercado"].keys()), "after L4")

    # ---------------- L3 ----------------
    logger.info("[L3] Ejecutando capa Strategy...")
    state = procesar_l3(state)
    logger.debug(f"[L3] Portfolio después de L3: {state['portfolio']}")
    logger.debug(f"[L3] Órdenes generadas: {state.get('ordenes', [])}")
    validate_portfolio(state["portfolio"], list(state["mercado"].keys()), "after L3")

    # ---------------- L2 (ASYNC) ----------------
    logger.info("[L2] Ejecutando capa Tactic...")
    if BUS is None or PROCESSOR is None:
        raise RuntimeError("BUS o PROCESSOR no inicializados")
    state = await PROCESSOR.process(state)

    # Validación previa a L1
    logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
    valid_symbols = set(state["mercado"].keys())
    portfolio_symbols = set(state["portfolio"].keys())
    if not portfolio_symbols.issubset(valid_symbols):
        invalid_symbols = portfolio_symbols - valid_symbols
        raise ValueError(f"Invalid symbols in portfolio before L1: {', '.join(invalid_symbols)}")

    # ---------------- L1 ----------------
    logger.info("[L1] Ejecutando capa Operational...")
    if asyncio.iscoroutinefunction(procesar_l1):
        state = await procesar_l1(state)
    else:
        state = await asyncio.to_thread(procesar_l1, state)
    logger.debug(f"[L1] Portfolio después de L1: {state['portfolio']}")
    logger.debug(f"[L1] Órdenes procesadas: {state.get('ordenes', [])}")

    # ---------------- Métricas ----------------
    telemetry.incr("ciclos_total")
    telemetry.timing("tick_time", start)
    valor_portfolio = _calc_portfolio_value(state)
    telemetry.gauge("valor_portfolio", valor_portfolio)
    logger.debug(f"[METRICS] Valor del portafolio: {valor_portfolio}")

    logger.info(
        f"[TICK] Ciclo {state['ciclo_id']} completado",
        extra={
            "ciclo_id": state["ciclo_id"],
            "valor_portfolio": valor_portfolio,
            "ordenes": len(state.get("ordenes", [])),
            "senales": len(state.get("senales", {}).get("signals", [])),
            "riesgo": state.get("riesgo"),
            "deriva": state.get("deriva"),
        },
    )

    # ---------------- Dashboard ----------------
    logger.debug("[DASHBOARD] Renderizando estado actual")
    try:
        render_dashboard(state)
    except Exception:
        logger.exception("[DASHBOARD] Error en render_dashboard")

    # ---------------- Persistencia ----------------
    logger.debug("[STORAGE] Guardando estado en CSV...")
    try:
        guardar_estado_csv(state)
    except Exception:
        logger.exception("[STORAGE] Error guardando CSV")
    logger.debug("[STORAGE] Guardando estado en SQLite...")
    try:
        guardar_estado_sqlite(state)
    except Exception:
        logger.exception("[STORAGE] Error guardando SQLite")

    return state


# ------------------------------------------------------------
# Seed de datos de ejemplo (si no hay feed real aún)
# ------------------------------------------------------------
def build_mock_state() -> dict:
    ts = pd.date_range(start="2025-08-23 21:03:00", periods=200, freq="1min", tz="UTC")
    btc = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [50000.0 + i * 10 for i in range(len(ts))],
            "high": [50500.0 + i * 10 for i in range(len(ts))],
            "low":  [49500.0 + i * 10 for i in range(len(ts))],
            "close":[50200.0 + i * 10 for i in range(len(ts))],
            "volume": [100.0] * len(ts),
        }
    ).set_index("timestamp")
    eth = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [2000.0 + i * 5 for i in range(len(ts))],
            "high": [2020.0 + i * 5 for i in range(len(ts))],
            "low":  [1980.0 + i * 5 for i in range(len(ts))],
            "close":[2010.0 + i * 5 for i in range(len(ts))],
            "volume": [50.0] * len(ts),
        }
    ).set_index("timestamp")
    usdt = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0] * len(ts),
            "high": [1.0] * len(ts),
            "low":  [1.0] * len(ts),
            "close":[1.0] * len(ts),
            "volume": [1000.0] * len(ts),
        }
    ).set_index("timestamp")

    state = {
        "portfolio": {"BTC/USDT": 0.7, "ETH/USDT": 0.2, "USDT": 100.0},
        "mercado": {"BTC/USDT": btc, "ETH/USDT": eth, "USDT": usdt},
        "regimen_context": {},
        "ciclo_id": 0,
    }

        # --- Cálculo de features adicionales ---
    features_by_symbol = {}
    for sym, df in state["mercado"].items():
        pat = detect_patterns(df)
        mtf = resample_and_consensus(df)
        sr  = swing_pivots(df)
        features_by_symbol[sym] = {
            "patterns": pat,
            "multi_tf": mtf,
            "s_r": sr,
        }
    state["features_by_symbol"] = features_by_symbol

    return state


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
async def main() -> None:
    
    global BUS, CONFIG_L2, PROCESSOR

    # Bus principal y adaptador
    core_bus = LocalMessageBus()
    BUS = BusAdapterAsync(core_bus)
    await BUS.start()

    # Config L2
    CONFIG_L2 = L2Config()

    # Inicializar Processor
    PROCESSOR = L2MainProcessor(CONFIG_L2, BUS)
    

    # Windows: usar Proactor si está disponible (coincide con el log "Using proactor: IocpProactor")
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.debug("Using proactor: IocpProactor")
        except Exception:
            logger.debug("Falling back to default event loop policy")

    # Bus principal y adaptador
    core_bus = LocalMessageBus()
    BUS = BusAdapterAsync(core_bus)  # el adaptador ya se suscribe a topics internos
    await BUS.start()

    # Config L2 (thresholds, sizing/risk, etc.)
    CONFIG_L2 = L2Config()

    # Estado inicial (mock si no hay datos reales aún)
    state = build_mock_state()
    logger.info(f"[MAIN] Estado inicial de portfolio: {state['portfolio']}")
    logger.debug(f"[MAIN] Símbolos válidos: {list(state['mercado'].keys())}")
    logger.info("[MAIN] Iniciando loop principal con ciclo cada 10s...")

    # Bucle continuo
    try:
        while True:
            try:
                state = await run_tick(state)
            except Exception:
                logger.exception("[MAIN] Error en tick; continuo tras breve espera")

            # Espera entre ciclos
            await asyncio.sleep(10)
    except asyncio.CancelledError:
        logger.info("[MAIN] Cancelado, cerrando...")
    except KeyboardInterrupt:
        logger.info("[MAIN] Interrumpido por usuario, cerrando...")


if __name__ == "__main__":
    asyncio.run(main())
