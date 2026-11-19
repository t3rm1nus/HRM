# -*- coding: utf-8 -*-
# main.py - Punto de entrada de HRM

import asyncio
import os
import sys
import traceback

# Silenciar avisos de TensorFlow antes de cualquier import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Raíz del proyecto al principio del path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bootstrap import setup_signal_handlers
from core.logging import logger

from app.apagado import apagar_sistema, guardar_portfolio
from app.arranque import iniciar_sistema
from app.ciclo import ejecutar_bucle_trading
from app.seguridad_live import verificar_seguridad_modo_live


async def main():
    """Sistema principal HRM."""
    await verificar_seguridad_modo_live()

    sistema = None
    try:
        setup_signal_handlers()
        sistema = await iniciar_sistema()
        if sistema is None:
            logger.info("👋 HRM System shutdown complete (fail fast)")
            return
        await ejecutar_bucle_trading(sistema)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
        guardar_portfolio(sistema)
    except Exception as error:
        logger.critical(f"🚨 FATAL ERROR: {error}")
        logger.critical(traceback.format_exc())
    finally:
        await apagar_sistema()


if __name__ == "__main__":
    asyncio.run(main())
