# scheduler.py
import asyncio
from datetime import datetime
from typing import Callable, Awaitable

async def run_every(interval_s: float, fn: Callable[[], Awaitable[None]], stop_event: asyncio.Event):
    while not stop_event.is_set():
        t0 = datetime.utcnow()
        try:
            await fn()
        except Exception as e:
            # aquí puedes loggear/telemetría y decidir si sigues o paras
            print(f"[SCHED] Error ciclo: {e}")
        # respeta el intervalo
        elapsed = (datetime.utcnow() - t0).total_seconds()
        timeout_seconds = max(0.0, interval_s - elapsed)
        if timeout_seconds == 0:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            # Timeout esperado: continuar con el siguiente ciclo
            pass
