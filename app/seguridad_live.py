# app/seguridad_live.py - Protección contra arrancar en live por accidente

import asyncio
import os

from core.logging import logger

from app.constantes import SEGUNDOS_CANCELAR_LIVE


def forzar_modo_paper():
    logger.warning(
        "⚠️ BINANCE_MODE=live detectado pero PAPER_MODE no es 'false' explícito. "
        "Forzando paper mode por seguridad. Para operar en live, debes establecer "
        "PAPER_MODE=false de forma explícita y consciente."
    )
    os.environ['PAPER_MODE'] = 'true'
    os.environ['BINANCE_MODE'] = 'paper'


async def esperar_confirmacion_live():
    logger.critical("🚨 MODO LIVE CON DINERO REAL DETECTADO 🚨")
    logger.critical(f"   Capital en riesgo: {os.getenv('INITIAL_CAPITAL', 'DESCONOCIDO')} USDT")
    logger.critical(f"   Tienes {SEGUNDOS_CANCELAR_LIVE} segundos para cancelar (Ctrl+C)...")
    await asyncio.sleep(SEGUNDOS_CANCELAR_LIVE)
    logger.critical("🚨 ARRANCANDO EN MODO LIVE — SIN VUELTA ATRÁS")


async def verificar_seguridad_modo_live():
    """Solo se opera en live si PAPER_MODE=false está explícito."""
    modo_live = os.getenv('BINANCE_MODE', 'paper').lower() == 'live'
    paper_desactivado = os.getenv('PAPER_MODE', 'true').lower() == 'false'

    if modo_live and not paper_desactivado:
        forzar_modo_paper()
    elif modo_live and paper_desactivado:
        await esperar_confirmacion_live()
