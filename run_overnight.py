# run_overnight.py (PICKLE PATCH ARREGLADO)
"""
Script para ejecutar el sistema toda la noche con logging persistente.
ARREGLADO: Pickle patch sin deprecation warnings
"""

import pickle
import numpy as np
import warnings
import asyncio
from datetime import datetime
from pathlib import Path

from core.logging import logger  # ✅ Logger centralizado

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
            logger.error("🔥 ERROR PICKLE DETECTADO EN LOAD:")
            logger.error(f"   Archivo: {getattr(file, 'name', 'unknown')}")
            logger.error(f"   Error: {str(e)[:200]}", exc_info=True)
        raise

def debug_pickle_loads(data, *args, **kwargs):
    try:
        return _original_pickle_loads(data, *args, **kwargs)
    except Exception as e:
        if "persistent" in str(e).lower() or "numpy" in str(e).lower():
            logger.error("🔥 ERROR PICKLE DETECTADO EN LOADS:")
            logger.error(f"   Error: {str(e)[:200]}", exc_info=True)
        raise

# Aplicar monkey patch mejorado
pickle.load = debug_pickle_load
pickle.loads = debug_pickle_loads

async def run_overnight():
    """Ejecutar el sistema toda la noche."""
    logger.info("🌙 INICIANDO EJECUCIÓN NOCTURNA")
    logger.info("📊 Logging centralizado activado")
    logger.info("⏰ Ejecución programada hasta interrupción")
    
    try:
        # Importar y ejecutar main
        from main import main
        await main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución interrumpida por usuario")
    except Exception as e:
        logger.error(f"⚠ Error crítico: {e}", exc_info=True)
    finally:
        logger.info("🌅 EJECUCIÓN NOCTURNA FINALIZADA")

if __name__ == "__main__":
    try:
        asyncio.run(run_overnight())
    except KeyboardInterrupt:
        print("\n🛑 Ejecución cancelada por usuario")
    except Exception as e:
        print(f"⚠ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
