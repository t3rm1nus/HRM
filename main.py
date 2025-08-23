# main.py
import asyncio
import time
import logging
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

logger = setup_logger(level=logging.DEBUG)  # Nivel DEBUG para más detalles

# --- Configuración global ---
config_l2 = L2Config()

def validate_portfolio(portfolio, valid_symbols):
    invalid = {symbol for symbol in portfolio if symbol not in valid_symbols}
    if invalid:
        logger.error(f"Invalid symbols in portfolio before L1: {invalid}")
        raise ValueError(f"Invalid symbols in portfolio before L1: {invalid}")

async def tick(state_holder: dict):
    try:
        start = time.time()
        state = state_holder["state"]
        state["ciclo_id"] += 1
        state["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(f"[TICK] Iniciando ciclo {state['ciclo_id']}")
        logger.debug(f"[TICK] Portfolio al inicio del ciclo: {state['portfolio']}")

        # --- pipeline ---
        logger.info("[L4] Ejecutando capa Meta...")
        try:
            state = procesar_l4(state)
            logger.debug(f"[L4] Portfolio después de procesar_l4: {state['portfolio']}")
        except Exception as e:
            logger.error(f"[L4] Error en procesar_l4: {e}", exc_info=True)
            raise

        logger.info("[L3] Ejecutando capa Strategy...")
        try:
            state = procesar_l3(state)
            logger.debug(f"[L3] Portfolio después de procesar_l3: {state['portfolio']}")
            logger.debug(f"[L3] Órdenes generadas: {state.get('ordenes', [])}")
        except Exception as e:
            logger.error(f"[L3] Error en procesar_l3: {e}", exc_info=True)
            raise

        logger.info("[L2] Ejecutando capa Tactic...")
        try:
            state = procesar_l2(state, config_l2)
            logger.debug(f"[L2] Portfolio después de procesar_l2: {state['portfolio']}")
            logger.debug(f"[L2] Señales generadas: {state.get('senales', {}).get('signals', [])}")
        except Exception as e:
            logger.error(f"[L2] Error en procesar_l2: {e}", exc_info=True)
            raise

        # Validar state["portfolio"] antes de procesar_l1
        logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
        valid_symbols = set(state["mercado"].keys())
        portfolio_symbols = set(state["portfolio"].keys())
        if not portfolio_symbols.issubset(valid_symbols):
            invalid_symbols = portfolio_symbols - valid_symbols
            logger.error(f"Invalid symbols in portfolio before L1: {invalid_symbols}")
            raise ValueError(f"Invalid symbols in portfolio before L1: {invalid_symbols}")

        logger.info("[L1] Ejecutando capa Operational...")
        try:
            if asyncio.iscoroutinefunction(procesar_l1):
                state = await procesar_l1(state)
            else:
                state = await asyncio.to_thread(procesar_l1, state)
            logger.debug(f"[L1] Portfolio después de procesar_l1: {state['portfolio']}")
            logger.debug(f"[L1] Órdenes procesadas: {state.get('ordenes', [])}")
        except Exception as e:
            logger.error(f"[L1] Error en procesar_l1: {e}", exc_info=True)
            raise

        # --- métricas ---
        try:
            telemetry.incr("ciclos_total")
            telemetry.timing("tick_time", start)
            valid_symbols = set(state["mercado"].keys())
            portfolio_symbols = set(state["portfolio"].keys())
            if not portfolio_symbols.issubset(valid_symbols):
                invalid_symbols = portfolio_symbols - valid_symbols
                raise ValueError(f"Invalid symbols in portfolio: {invalid_symbols}")
            
            valor_portfolio = sum(
                qty * state["mercado"][a]["close"].iloc[-1] for a, qty in state["portfolio"].items()
            )
            telemetry.gauge("valor_portfolio", valor_portfolio)
            logger.debug(f"[METRICS] Valor del portafolio: {valor_portfolio}")
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}", exc_info=True)
            raise

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
        try:
            logger.debug("[DASHBOARD] Renderizando estado actual")
            render_dashboard(state)
        except Exception as e:
            logger.error(f"[DASHBOARD] Error en render_dashboard: {e}", exc_info=True)
            raise

        # --- persistencia ---
        try:
            logger.debug("[STORAGE] Guardando estado en CSV...")
            guardar_estado_csv(state)
            logger.debug("[STORAGE] Guardando estado en SQLite...")
            guardar_estado_sqlite(state)
        except Exception as e:
            logger.error(f"[STORAGE] Error en almacenamiento: {e}", exc_info=True)
            raise

        state_holder["state"] = state

    except Exception as e:
        logger.error(f"[TICK] Error en ciclo {state['ciclo_id']}: {e}", exc_info=True)
        raise

async def tick_wrapper(holder: dict):
    """Wrapper sin argumentos extras para scheduler"""
    await tick(holder)

async def main():
    # Mock configuration and data
    valid_symbols = ["BTC/USDT", "ETH/USDT", "USDT"]
    portfolio = {"BTC/USDT": 0.7, "ETH/USDT": 0.2, "USDT": 100.0}
    market_data = {
        "BTC/USDT": pd.DataFrame({
            "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
            "open": [50000.0 + i * 10 for i in range(100)],
            "high": [50500.0 + i * 10 for i in range(100)],
            "low": [49500.0 + i * 10 for i in range(100)],
            "close": [50200.0 + i * 10 for i in range(100)],
            "volume": [100.0] * 100
        }).set_index("timestamp"),
        "ETH/USDT": pd.DataFrame({
            "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
            "open": [2000.0 + i * 5 for i in range(100)],
            "high": [2020.0 + i * 5 for i in range(100)],
            "low": [1980.0 + i * 5 for i in range(100)],
            "close": [2010.0 + i * 5 for i in range(100)],
            "volume": [50.0] * 100
        }).set_index("timestamp"),
        "USDT": pd.DataFrame({
            "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
            "open": [1.0] * 100,
            "high": [1.0] * 100,
            "low": [1.0] * 100,
            "close": [1.0] * 100,
            "volume": [1000.0] * 100
        }).set_index("timestamp")
    }
    state = {
        "portfolio": portfolio,
        "mercado": market_data,
        "regimen_context": {},
        "ciclo_id": 0
    }

    logger.info(f"[MAIN] Estado inicial de portfolio: {portfolio}")
    logger.info("[MAIN] Iniciando loop principal con ciclo cada 10s...")

    for cycle in range(1, 2):  # Single cycle for testing
        logger.info(f"[TICK] Iniciando ciclo {cycle}")
        logger.info("[L4] Ejecutando capa Meta...")
        state = procesar_l4(state)
        logger.info("[L3] Ejecutando capa Strategy...")
        state = procesar_l3(state)
        logger.info("[L2] Ejecutando capa Tactic...")
        
        # Validate portfolio
        validate_portfolio(portfolio, valid_symbols)
        
        # Process L2 with config_l2
        state = procesar_l2(state, config_l2)
        
        logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
        await tick({"state": state})  # Ejecutar tick para procesar L1 y posteriores

if __name__ == "__main__":
    asyncio.run(main())