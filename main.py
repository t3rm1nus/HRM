import asyncio 
from core.scheduler import run_every
from core.utils import dump_state
from comms.schemas import State
import time
from core.logging import setup_logger
from monitoring.telemetry import telemetry
from monitoring.dashboard import render_dashboard
from storage import guardar_estado_csv, guardar_estado_sqlite

from l4_meta import procesar_l4
from l3_strategy import procesar_l3
from l2_tactic import procesar_l2
from l1_operational import procesar_l1   # nuevo hook unificado L1

logger = setup_logger()

async def tick(state_holder: dict):
    start = time.time()
    state = state_holder["state"]
    state["ciclo_id"] += 1

    # --- pipeline ---
    state = procesar_l4(state)
    state = procesar_l3(state)
    state = procesar_l2(state)
    state = await asyncio.to_thread(procesar_l1, state)  # L1 ejecuta órdenes y actualiza estado

    # --- métricas ---
    telemetry.incr("ciclos_total")
    telemetry.timing("tick_time", start)
    valor_portfolio = sum(qty * state["mercado"][a] for a, qty in state["portfolio"].items())
    telemetry.gauge("valor_portfolio", valor_portfolio)

    logger.info("Ciclo completado", extra={"ciclo_id": state["ciclo_id"], "valor_portfolio": valor_portfolio})

    # --- dashboard ---
    render_dashboard(state)

    # --- persistencia ---
    guardar_estado_csv(state)
    guardar_estado_sqlite(state)

    state_holder["state"] = state

async def main():
    state: State = {
        "version": "v1",
        "ciclo_id": 0,
        "ts": "1970-01-01T00:00:00Z",
        "portfolio": {"BTC": 0.7, "ETH": 0.2, "USDT": 100.0},  # unidades
        "mercado": {"BTC": 30000, "ETH": 2000, "USDT": 1},     # precios USD
        "exposicion": {"BTC": 1.0, "ETH": 0.5, "USDT": 50.0},  # target en unidades
        "estrategia": None,
        "universo": [],
        "senales": {},
        "ordenes": [],    # ← aquí L2 manda órdenes, L1 las consume
        "riesgo": {"aprobado": True, "motivo": None},
        "deriva": False,
    }
    holder = {"state": state}
    stop = asyncio.Event()
    task = asyncio.create_task(run_every(10.0, lambda: tick(holder), stop))

    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
