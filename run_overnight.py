# run_overnight.py (PICKLE PATCH ARREGLADO)
"""
Script para ejecutar el sistema toda la noche con logging persistente.
ARREGLADO: Pickle patch sin deprecation warnings
"""

import pickle
import numpy as np
import warnings

# Suprimir warnings específicos de NumPy/Pickle
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*numpy.*_core.*", category=DeprecationWarning)

# PICKLE PATCH ARREGLADO - sin usar APIs deprecadas
_original_pickle_load = pickle.load
_original_pickle_loads = pickle.loads

def debug_pickle_load(file, *args, **kwargs):
    try:
        return _original_pickle_load(file, *args, **kwargs)
    except Exception as e:
        if "persistent" in str(e).lower() or "numpy" in str(e).lower():
            print(f"🔥 ERROR PICKLE DETECTADO EN LOAD:")
            print(f"   Archivo: {getattr(file, 'name', 'unknown')}")
            print(f"   Error: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
        raise

def debug_pickle_loads(data, *args, **kwargs):
    try:
        return _original_pickle_loads(data, *args, **kwargs)
    except Exception as e:
        if "persistent" in str(e).lower() or "numpy" in str(e).lower():
            print(f"🔥 ERROR PICKLE DETECTADO EN LOADS:")
            print(f"   Error: {str(e)[:200]}")
        raise

# Aplicar monkey patch mejorado
pickle.load = debug_pickle_load
pickle.loads = debug_pickle_loads

# Resto del código sin cambios
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
        logger.error(f"⚠ Error crítico: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info("🌅 EJECUCIÓN NOCTURNA FINALIZADA")

if __name__ == "__main__":
    # Ejecutar con manejo de excepciones
    try:
        asyncio.run(run_overnight())
    except KeyboardInterrupt:
        print("\n🛑 Ejecución cancelada por usuario")
    except Exception as e:
        print(f"⚠ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
