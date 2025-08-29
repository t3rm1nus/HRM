#!/usr/bin/env python3
"""
Script para ejecutar el sistema toda la noche con logging persistente.
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging para archivo y consola
log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("overnight")

async def run_overnight():
    """Ejecutar el sistema toda la noche."""
    logger.info("🌙 INICIANDO EJECUCIÓN NOCTURNA")
    logger.info(f"📁 Logfile: {log_file}")
    logger.info("📊 Logging persistente activado")
    logger.info("⏰ Ejecución programada hasta interrupción")
    
    try:
        # Importar y ejecutar main
        from main import main
        await main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución interrumpida por usuario")
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
    finally:
        logger.info("🌅 EJECUCIÓN NOCTURNA FINALIZADA")

if __name__ == "__main__":
    # Ejecutar con manejo de excepciones
    try:
        asyncio.run(run_overnight())
    except KeyboardInterrupt:
        print("\n🛑 Ejecución cancelada por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")