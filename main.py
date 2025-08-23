import asyncio
import time
from core.scheduler import run_every
from core.utils import dump_state
from comms.schemas import State
from core.logging import setup_logger
from monitoring.telemetry import telemetry
from monitoring.dashboard import render_dashboard
from storage import guardar_estado_csv, guardar_estado_sqlite

from l2_tactic.procesar_l2 import procesar_l2
from l2_tactic.config import L2Config
from l2_tactic.signal_generator import SignalGenerator

from l4_meta import procesar_l4
from l3_strategy import procesar_l3
from l1_operational import procesar_l1

import pandas as pd

logger = setup_logger()

# --- Configuración global ---
config_l2 = L2Config()


async def tick(state_holder: dict):
    try:
        start = time.time()
        state = state_holder["state"]
        state["ciclo_id"] += 1
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(f"[TICK] Iniciando ciclo {state['ciclo_id']}")

        # --- pipeline ---
        logger.info("[L4] Ejecutando capa Meta...")
        state = procesar_l4(state)

        logger.info("[L3] Ejecutando capa Strategy...")
        state = procesar_l3(state)

        logger.info("[L2] Ejecutando capa Tactic...")
        state = procesar_l2(state, config_l2)

        logger.info("[L1] Ejecutando capa Operational...")
        if asyncio.iscoroutinefunction(procesar_l1):
            state = await procesar_l1(state)
        else:
            state = await asyncio.to_thread(procesar_l1, state)

        # --- métricas ---
        telemetry.incr("ciclos_total")
        telemetry.timing("tick_time", start)
        valor_portfolio = sum(
            qty * state["mercado"][a]["close"].iloc[-1] for a, qty in state["portfolio"].items()
        )
        telemetry.gauge("valor_portfolio", valor_portfolio)

        logger.info(
            f"[TICK] Ciclo {state['ciclo_id']} completado",
            extra={
                "ciclo_id": state["ciclo_id"],
                "valor_portfolio": valor_portfolio,
                "ordenes": len(state.get("ordenes", [])),
                "senales": len(state.get("senales", {}).get("signals", [])),
                "riesgo": state.get("riesgo"),
                "deriva": state.get("deriva"),
            }
        )

        # --- dashboard ---
        logger.debug("[DASHBOARD] Renderizando estado actual")
        render_dashboard(state)

        # --- persistencia ---
        logger.debug("[STORAGE] Guardando estado en CSV...")
        guardar_estado_csv(state)

        logger.debug("[STORAGE] Guardando estado en SQLite...")
        guardar_estado_sqlite(state)

        state_holder["state"] = state

    except Exception as e:
        logger.error(f"[TICK] Error en ciclo {state['ciclo_id']}: {e}", exc_info=True)
        raise

async def tick_wrapper(holder: dict):
    """Wrapper sin argumentos extras para scheduler"""
    await tick(holder)


async def main():
    # Estado inicial (adaptado para multiasset con símbolos consistentes)
    state: State = {
        "version": "v1",
        "ciclo_id": 0,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "portfolio": {"BTC/USDT": 0.7, "ETH/USDT": 0.2, "USDT": 100.0},
        "mercado": {
            "BTC/USDT": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
                "open": [50000.0 + i * 10 for i in range(100)],
                "high": [50500.0 + i * 10 for i in range(100)],
                "low": [49500.0 + i * 10 for i in range(100)],
                "close": [50200.0 + i * 10 for i in range(100)],
                "volume": [100.0] * 100
            }),
            "ETH/USDT": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
                "open": [3000.0 + i * 5 for i in range(100)],
                "high": [3050.0 + i * 5 for i in range(100)],
                "low": [2950.0 + i * 5 for i in range(100)],
                "close": [3020.0 + i * 5 for i in range(100)],
                "volume": [200.0] * 100
            }),
            "USDT": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
                "open": [1.0] * 100,
                "high": [1.0] * 100,
                "low": [1.0] * 100,
                "close": [1.0] * 100,
                "volume": [10000.0] * 100
            }),
        },
        "exposicion": {"BTC/USDT": 1.0, "ETH/USDT": 0.5, "USDT": 50.0},
        "estrategia": "estrategia_agresiva",
        "universo": ["BTC/USDT", "ETH/USDT", "USDT"],
        "senales": {},
        "ordenes": [],
        "riesgo": {"aprobado": True, "motivo": None},
        "deriva": False,
    }

    holder = {"state": state}
    stop = asyncio.Event()

    logger.info("[MAIN] Iniciando loop principal con ciclo cada 10s...")
    try:
        task = asyncio.create_task(run_every(10.0, lambda: tick_wrapper(holder), stop))
        await task
    except asyncio.CancelledError:
        logger.info("[MAIN] Loop cancelado correctamente")
    except Exception as e:
        logger.error(f"[MAIN] Error en el loop principal: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())