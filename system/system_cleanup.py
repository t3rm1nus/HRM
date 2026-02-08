"""
System Cleanup Module - Resetear singletons y estado del sistema

Este módulo proporciona funciones para limpiar y resetear completamente
el estado del sistema HRM entre ejecuciones.

Cleanup Strategy:
- filesystem_cleanup(): Limpiar archivos de estado/persistent_state
- memory_reset(): Resetear singletons en memoria
- async_context_reset(): Resetear contexto async
"""

import os
import glob
import logging
from typing import Dict, Any, Optional
from core.logging import logger

# ============================================================================
# SIMULATED EXCHANGE CLIENT RESET
# ============================================================================

def cleanup_simulated_exchange_client():
    """
    Resetear SimulatedExchangeClient singleton para nueva ejecución.
    
    🔥 PRIORIDAD 3: Quitar SIM_INIT_ONCE tras cleanup
    """
    try:
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        
        # Resetear flags de singleton - QUITAR SIM_INIT_ONCE
        SimulatedExchangeClient._instance = None
        SimulatedExchangeClient._initialized = False
        
        logger.info("🔄 SimulatedExchangeClient singleton reseteado (SIM_INIT_ONCE removido)")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ SimulatedExchangeClient no disponible para cleanup: {e}")
        return False


# ============================================================================
# STATE COORDINATOR RESET
# ============================================================================

def cleanup_state_coordinator():
    """
    Resetear StateCoordinator singleton para nueva ejecución.
    """
    try:
        from system.state_coordinator import StateCoordinator
        from core.state_manager import _global_state_coordinator
        
        # Resetear referencia global
        import core.state_manager
        core.state_manager._global_state_coordinator = None
        
        # Resetear cualquier estado estático de StateCoordinator
        if hasattr(StateCoordinator, '_instance'):
            StateCoordinator._instance = None
        
        logger.info("🔄 StateCoordinator singleton reseteado")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ StateCoordinator no disponible para cleanup: {e}")
        return False


# ============================================================================
# POSITION MANAGER RESET  
# ============================================================================

def cleanup_position_manager():
    """
    Resetear PositionManager singleton si existe.
    
    🔥 PRIORIDAD 3: Resetear PositionManager tras cleanup
    """
    try:
        from l1_operational.position_manager import PositionManager
        
        # Si PositionManager es singleton, resetear
        if hasattr(PositionManager, '_instance'):
            PositionManager._instance = None
            logger.info("🔄 PositionManager singleton reseteado")
        
        logger.info("🔄 PositionManager cleanup completado")
        return True
    except ImportError:
        logger.info("🔄 PositionManager no disponible para cleanup")
        return True


# ============================================================================
# CORE CONFIG RESET
# ============================================================================

def cleanup_core_config():
    """
    Resetear configuración core para forzar modo paper.
    """
    try:
        import core.config
        core.config._config_instance = None
        
        logger.info("🔄 core.config singleton reseteado")
        return True
    except Exception as e:
        logger.warning(f"⚠️ core.config cleanup falló: {e}")
        return False


# ============================================================================
# FILESYSTEM CLEANUP
# ============================================================================

def filesystem_cleanup() -> Dict[str, Any]:
    """
    Limpiar archivos de estado del sistema.
    
    Returns:
        Dict con información de archivos eliminados
    """
    deleted_files = []
    deleted_dirs = []
    errors = []
    
    # Patrones de archivos a eliminar
    patterns_to_clean = [
        "persistent_state/*.json",
        "persistent_state/*.bak",
        "portfolio_state*.json",
        "*.log",
        "paper_trades/*.json",
    ]
    
    for pattern in patterns_to_clean:
        try:
            files = glob.glob(pattern, recursive=True)
            for file_path in files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    logger.debug(f"🗑️ Eliminado: {file_path}")
        except Exception as e:
            errors.append(f"Error limpiando {pattern}: {e}")
    
    # Limpiar directorios vacíos
    dirs_to_check = ["persistent_state", "paper_trades", "logs"]
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    deleted_dirs.append(dir_path)
            except Exception as e:
                pass  # Directorio no vacío o no removable
    
    result = {
        "deleted_files": len(deleted_files),
        "deleted_dirs": len(deleted_dirs),
        "files_list": deleted_files,
        "errors": errors
    }
    
    logger.info(f"🧹 Filesystem cleanup: {len(deleted_files)} archivos eliminados")
    return result


# ============================================================================
# MEMORY RESET
# ============================================================================

def memory_reset() -> Dict[str, Any]:
    """
    Resetear todos los singletons en memoria.
    
    🔥 PRIORIDAD 3: Resetear SimulatedExchangeClient, PositionManager, StateCoordinator
    
    Returns:
        Dict con estado del reset
    """
    reset_results = {}
    
    # Reset SimulatedExchangeClient
    reset_results["simulated_exchange"] = cleanup_simulated_exchange_client()
    
    # Reset StateCoordinator
    reset_results["state_coordinator"] = cleanup_state_coordinator()
    
    # Reset core config
    reset_results["core_config"] = cleanup_core_config()
    
    # Reset PositionManager
    reset_results["position_manager"] = cleanup_position_manager()
    
    # Limpiar variables globales de config
    try:
        import core.config
        # Forzar PAPER_MODE global
        if hasattr(core.config, 'TEMPORARY_AGGRESSIVE_MODE'):
            core.config.TEMPORARY_AGGRESSIVE_MODE = False
        logger.info("🔄 Variables globales de config reseteadas")
    except Exception as e:
        logger.warning(f"⚠️ Error reseteando variables globales: {e}")
    
    total_reset = sum(1 for v in reset_results.values() if v)
    logger.info(f"🧠 Memory reset: {total_reset}/{len(reset_results)} componentes reseteados")
    
    return reset_results


# ============================================================================
# ASYNC CONTEXT RESET
# ============================================================================

def async_context_reset() -> Dict[str, Any]:
    """
    Resetear contexto async del sistema.
    
    Returns:
        Dict con estado del reset
    """
    reset_results = {}
    
    # Resetear event loops si es posible
    reset_results["event_loop_status"] = "not_applicable"
    
    # Limpiar caches async
    try:
        # Limpiar cache de sentiment si existe
        import sentiment.sentiment_manager as sm
        if hasattr(sm, '_sentiment_cache'):
            sm._sentiment_cache = {}
            logger.info("🔄 Sentiment cache limpio")
    except Exception as e:
        logger.debug(f"No se pudo limpiar sentiment cache: {e}")
    
    # Limpiar caches de L2
    try:
        import l2_tactic.signal_generators as sg
        if hasattr(sg, '_signal_cache'):
            sg._signal_cache = {}
            logger.info("🔄 L2 signal cache limpio")
    except Exception as e:
        logger.debug(f"No se pudo limpiar L2 signal cache: {e}")
    
    logger.info("🔄 Async context reset completado")
    return reset_results


# ============================================================================
# FULL CLEANUP - MODO EXPLÍCITO PAPER
# ============================================================================

def perform_full_cleanup(mode: str = "paper") -> Dict[str, Any]:
    """
    Realizar limpieza completa del sistema.
    
    Args:
        mode: Modo forzado tras cleanup ("paper" por defecto)
    
    Returns:
        Dict con resultados de cleanup
    """
    logger.info(f"🧹 INICIANDO CLEANUP COMPLETO (modo forzado: {mode})")
    
    results = {
        "mode": mode,
        "filesystem": None,
        "memory": None,
        "async_context": None
    }
    
    # 1. Filesystem cleanup
    try:
        results["filesystem"] = filesystem_cleanup()
    except Exception as e:
        logger.error(f"❌ Filesystem cleanup falló: {e}")
        results["filesystem"] = {"error": str(e)}
    
    # 2. Memory reset
    try:
        results["memory"] = memory_reset()
    except Exception as e:
        logger.error(f"❌ Memory reset falló: {e}")
        results["memory"] = {"error": str(e)}
    
    # 3. Async context reset
    try:
        results["async_context"] = async_context_reset()
    except Exception as e:
        logger.error(f"❌ Async context reset falló: {e}")
        results["async_context"] = {"error": str(e)}
    
    # 4. FORZAR MODO PAPER EXPLÍCITAMENTE
    try:
        import core.config
        core.config._config_instance = None
        
        # Forzar PAPER_MODE en todas las configs
        import sys
        module = sys.modules.get('core.config')
        if module:
            # Patchear para que siempre devuelva paper
            original_get = module.EnvironmentConfig.get
            def patched_get(self, key, default=None):
                if key in ["PAPER_MODE", "OPERATION_MODE", "mode"]:
                    if key == "PAPER_MODE":
                        return True
                    elif key == "OPERATION_MODE":
                        return "PAPER"
                    elif key == "mode":
                        return mode
                return original_get(self, key, default)
            module.EnvironmentConfig.get = patched_get
        
        logger.info(f"✅ MODO FORZADO: {mode}")
    except Exception as e:
        logger.warning(f"⚠️ Error forzando modo: {e}")
    
    # Resumen
    total_deleted = results["filesystem"].get("deleted_files", 0) if results["filesystem"] else 0
    memory_reset_count = sum(1 for k, v in results["memory"].items() if v and k != "errors") if results.get("memory") else 0
    
    logger.info(f"✅ CLEANUP COMPLETO:")
    logger.info(f"   📁 Archivos eliminados: {total_deleted}")
    logger.info(f"   🧠 Singletons reseteados: {memory_reset_count}")
    logger.info(f"   🎯 Modo forzado: {mode}")
    
    results["success"] = True
    return results


# ============================================================================
# FORZAR MODO PAPER (UTILIDAD)
# ============================================================================

def force_paper_mode():
    """
    Forzar que todo el sistema opere en modo paper.
    """
    logger.info("🎯 FORZANDO MODO PAPER EN TODO EL SISTEMA")
    
    # 1. Resetear SimulatedExchangeClient
    cleanup_simulated_exchange_client()
    
    # 2. Resetear core config
    cleanup_core_config()
    
    # 3. Forzar variables globales
    try:
        import core.config
        core.config.PAPER_MODE = True
        
        import sys
        module = sys.modules.get('core.config')
        if module:
            # Hacer que get_config siempre devuelva paper
            module._config_instance = None
    except Exception as e:
        logger.warning(f"⚠️ Error en force_paper_mode: {e}")
    
    logger.info("✅ MODO PAPER FORZADO ACTIVADO")
    return True
