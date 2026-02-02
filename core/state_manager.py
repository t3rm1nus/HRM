"""
Gestión de Estado del Sistema HRM

Este módulo maneja el estado global del sistema entre ciclos de trading,
incluyendo validación, persistencia y registro de datos de ciclo.
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime
import json
import os

from core.logging import logger
from core.config import get_config


def initialize_state(symbols: List[str], initial_balance: float) -> Dict:
    """
    Inicializa el estado del sistema con valores por defecto.
    
    ⚠️  ESTE MÉTODO NO DEBE EJECUTARSE DENTRO DEL TRADING LOOP
    ⚠️  Solo debe usarse durante la inicialización del sistema
    
    Args:
        symbols: Lista de símbolos de trading
        initial_balance: Balance inicial en USDT
        
    Returns:
        Dict con el estado inicial del sistema
    """
    # Protección contra ejecución en trading loop
    if hasattr(initialize_state, '_in_loop') and initialize_state._in_loop:
        logger.error("❌ initialize_state() NO puede ejecutarse dentro del trading loop")
        raise RuntimeError("initialize_state() called from within trading loop - this is prohibited")
    
    config = get_config("live")
    
    state = {
        "cycle_id": 0,
        "market_data": {},
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": initial_balance,
            "total_value": initial_balance
        },
        "l3_output": {},
        "l3_decision_cache": None,
        "l3_last_update": 0,
        "l3_previous_regime": None,
        "l3_previous_setup_type": None,
        "signal_execution_logged": False,
        "l3_fallback": False,
        "total_value": initial_balance,
        "btc_balance": 0.0,
        "eth_balance": 0.0,
        "usdt_balance": initial_balance
    }
    
    logger.info(f"✅ Estado inicializado para {len(symbols)} símbolos con balance inicial {initial_balance} USDT")
    return state


def validate_state_structure(state: Dict) -> Dict:
    """
    Valida y repara la estructura del estado del sistema.
    
    ⚠️  ESTE MÉTODO NO DEBE EJECUTARSE DENTRO DEL TRADING LOOP
    ⚠️  Solo debe usarse durante la inicialización del sistema
    
    Args:
        state: Estado del sistema a validar
        
    Returns:
        Dict con el estado validado y reparado
    """
    # Protección contra ejecución en trading loop
    if hasattr(validate_state_structure, '_in_loop') and validate_state_structure._in_loop:
        logger.error("❌ validate_state_structure() NO puede ejecutarse dentro del trading loop")
        raise RuntimeError("validate_state_structure() called from within trading loop - this is prohibited")
    
    if not isinstance(state, dict):
        logger.error("❌ Estado inválido: no es un diccionario")
        return initialize_state(["BTCUSDT", "ETHUSDT"], 3000.0)
    
    # Validar estructura mínima requerida
    required_keys = ["market_data", "portfolio", "l3_output"]
    for key in required_keys:
        if key not in state:
            logger.warning(f"⚠️ Clave faltante en estado: {key}")
            if key == "market_data":
                state[key] = {}
            elif key == "portfolio":
                state[key] = {
                    "btc_balance": 0.0,
                    "eth_balance": 0.0,
                    "usdt_balance": 3000.0,
                    "total_value": 3000.0
                }
            elif key == "l3_output":
                state[key] = {}
    
    # Validar tipos de datos
    if not isinstance(state["market_data"], dict):
        logger.warning("⚠️ market_data no es un diccionario, inicializando vacío")
        state["market_data"] = {}
    
    if not isinstance(state["portfolio"], dict):
        logger.warning("⚠️ portfolio no es un diccionario, inicializando valores por defecto")
        state["portfolio"] = {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    
    # Validar balances
    for key in ["btc_balance", "eth_balance", "usdt_balance", "total_value"]:
        if key not in state["portfolio"]:
            state["portfolio"][key] = 0.0
        elif not isinstance(state["portfolio"][key], (int, float)):
            logger.warning(f"⚠️ Balance {key} no es numérico, estableciendo a 0.0")
            state["portfolio"][key] = 0.0
    
    logger.debug("✅ Estado validado y reparado exitosamente")
    return state


def log_cycle_data(state: Dict, cycle_id: int, start_time: pd.Timestamp) -> None:
    """
    Registra datos del ciclo de trading para auditoría.
    
    Args:
        state: Estado actual del sistema
        cycle_id: ID del ciclo actual
        start_time: Timestamp de inicio del ciclo
    """
    try:
        cycle_data = {
            "cycle_id": cycle_id,
            "timestamp": start_time.isoformat(),
            "duration": (pd.Timestamp.utcnow() - start_time).total_seconds(),
            "portfolio": state.get("portfolio", {}),
            "l3_output": state.get("l3_output", {}),
            "market_symbols": list(state.get("market_data", {}).keys()),
            "total_value": state.get("total_value", 0.0)
        }
        
        # Guardar en archivo de logs
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"cycle_{cycle_id}.json")
        with open(log_file, 'w') as f:
            json.dump(cycle_data, f, indent=2, default=str)
        
        logger.debug(f"📊 Datos del ciclo {cycle_id} registrados en {log_file}")
        
    except Exception as e:
        logger.error(f"❌ Error registrando datos del ciclo {cycle_id}: {e}")


def update_state_from_market_data(state: Dict, market_data: Dict) -> Dict:
    """
    Actualiza el estado con nuevos datos de mercado.
    
    Args:
        state: Estado actual del sistema
        market_data: Nuevos datos de mercado
        
    Returns:
        Dict con el estado actualizado
    """
    if not market_data or not isinstance(market_data, dict):
        logger.warning("⚠️ Datos de mercado inválidos, omitiendo actualización")
        return state
    
    # Validar y filtrar datos de mercado
    valid_data = {}
    for symbol, data in market_data.items():
        if isinstance(data, (dict, pd.DataFrame)):
            valid_data[symbol] = data
    
    if valid_data:
        state["market_data"] = valid_data
        logger.debug(f"✅ Estado actualizado con datos de {len(valid_data)} símbolos")
    
    return state


def get_state_summary(state: Dict) -> Dict:
    """
    Obtiene un resumen del estado actual del sistema.
    
    Args:
        state: Estado del sistema
        
    Returns:
        Dict con resumen del estado
    """
    portfolio = state.get("portfolio", {})
    
    summary = {
        "cycle_id": state.get("cycle_id", 0),
        "total_symbols": len(state.get("market_data", {})),
        "portfolio_value": portfolio.get("total_value", 0.0),
        "btc_balance": portfolio.get("btc_balance", 0.0),
        "eth_balance": portfolio.get("eth_balance", 0.0),
        "usdt_balance": portfolio.get("usdt_balance", 0.0),
        "l3_active": bool(state.get("l3_output", {})),
        "l3_fallback": state.get("l3_fallback", False)
    }
    
    return summary


# Variable global para almacenar la referencia al StateCoordinator
_global_state_coordinator = None

def inject_state_coordinator(coordinator):
    """
    Inyecta la referencia al StateCoordinator global.
    
    DEBE ser llamado desde main.py ANTES del trading loop.
    
    Args:
        coordinator: Instancia de StateCoordinator creada en main.py
    """
    global _global_state_coordinator
    _global_state_coordinator = coordinator
    logger.info("✅ StateCoordinator inyectado correctamente en state_manager")

def get_system_state() -> Dict:
    """
    Obtiene el estado actual del sistema.
    
    Returns:
        Dict con el estado del sistema
        
    Raises:
        RuntimeError: Si StateCoordinator no ha sido inyectado
    """
    global _global_state_coordinator
    
    if _global_state_coordinator is None:
        error_msg = "StateCoordinator not injected. Call inject_state_coordinator() first."
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    return _global_state_coordinator.get_state("current")

def get_state_manager():
    """
    Obtiene el gestor de estado del sistema.
    
    Returns:
        Referencia al StateCoordinator inyectado
        
    Raises:
        RuntimeError: Si StateCoordinator no ha sido inyectado
    """
    global _global_state_coordinator
    
    if _global_state_coordinator is None:
        error_msg = "StateCoordinator not injected. Call inject_state_coordinator() first."
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    return _global_state_coordinator

def transition_system_state(state_type: str, reason: str, metadata: Dict = None) -> None:
    """
    Transiciona el estado del sistema a un nuevo estado.
    
    Args:
        state_type: Tipo de estado al que transicionar (e.g., "BLIND", "NORMAL")
        reason: Razón de la transición
        metadata: Metadatos adicionales para la transición
    """
    global _global_state_coordinator
    
    if _global_state_coordinator is None:
        error_msg = "StateCoordinator not injected. Call inject_state_coordinator() first."
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    try:
        # Obtener el modo actual del sistema
        system_mode = get_system_mode()
        
        # FIX DEFINITIVO - BLIND MODE PROHIBIDO en simulated
        if system_mode == "simulated":
            logger.info("🛡️ FIX DEFINITIVO: Modo simulated detectado - BLIND MODE prohibido")
            # En modo simulated, forzar estado ACTIVE
            updates = {
                "system_state_type": "ACTIVE",
                "system_state_reason": "Simulated mode - local portfolio authoritative",
                "system_state_metadata": {
                    "mode": "simulated",
                    "blind_mode_disabled": True,
                    "local_portfolio_trusted": True
                },
                "system_state_timestamp": datetime.utcnow().isoformat()
            }
        else:
            # En modo real, permitir transición normal
            updates = {
                "system_state_type": state_type,
                "system_state_reason": reason,
                "system_state_metadata": metadata,
                "system_state_timestamp": datetime.utcnow().isoformat()
            }
        
        _global_state_coordinator.update_state(updates)
        logger.info(f"✅ Sistema transicionado a estado: {state_type} - {reason}")
    except Exception as e:
        logger.error(f"❌ Error transicionando estado del sistema: {e}")
        raise


def get_system_mode() -> str:
    """
    Obtiene el modo actual del sistema (live, testnet, backtest, simulated).
    
    Returns:
        str: Modo actual del sistema
    """
    try:
        from core.config import get_config
        config = get_config("live")
        return getattr(config, 'mode', 'unknown')
    except Exception as e:
        logger.warning(f"⚠️ No se pudo obtener el modo del sistema: {e}")
        return 'unknown'


def enforce_fundamental_rule() -> None:
    """
    REGLA FUNDAMENTAL: En modo simulated, el portfolio local ES la fuente de verdad.
    
    Esto desactiva BLIND MODE en simulated y permite:
    - órdenes
    - rebalance
    - rotación
    """
    global _global_state_coordinator
    
    if _global_state_coordinator is None:
        error_msg = "StateCoordinator not injected. Call inject_state_coordinator() first."
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    try:
        # Obtener el modo actual del sistema
        system_mode = get_system_mode()
        
        if system_mode == "simulated":
            logger.info("🛡️ REGLA FUNDAMENTAL: Modo simulated detectado - desactivando BLIND MODE")
            
            # Forzar estado ACTIVE en simulated mode
            updates = {
                "system_state_type": "ACTIVE",
                "system_state_reason": "Simulated mode - local balances trusted",
                "system_state_metadata": {
                    "mode": "simulated",
                    "blind_mode_disabled": True,
                    "local_portfolio_trusted": True
                },
                "system_state_timestamp": datetime.utcnow().isoformat()
            }
            
            _global_state_coordinator.update_state(updates)
            logger.info("✅ REGLA FUNDAMENTAL: Sistema en modo ACTIVE para simulated")
        else:
            logger.debug(f"🔄 Modo {system_mode} - REGLA FUNDAMENTAL no aplicable")
            
    except Exception as e:
        logger.error(f"❌ Error aplicando REGLA FUNDAMENTAL: {e}")
        raise

def can_system_rebalance() -> bool:
    """
    Verifica si el sistema puede realizar reequilibrio.
    
    Esta función actúa como fachada que delega internamente a StateCoordinator
    para validar las condiciones de reequilibrio sin contener lógica de trading.
    
    Returns:
        bool indicando si el sistema puede reequilibrar
    """
    try:
        # Importar StateCoordinator solo cuando se necesita
        from system.state_coordinator import StateCoordinator
        
        # Obtener instancia del coordinador
        coordinator = StateCoordinator()
        
        # Obtener estado actual
        state = coordinator.get_state("current")
        
        # Validar condiciones básicas para reequilibrio
        portfolio = state.get("portfolio", {})
        total_value = portfolio.get("total_value", 0.0)
        
        # No reequilibrar si no hay valor en cartera
        if total_value <= 0:
            return False
        
        # No reequilibrar si está en modo fallback L3
        if state.get("l3_fallback", False):
            return False
        
        # No reequilibrar si no hay datos de mercado
        market_data = state.get("market_data", {})
        if not market_data:
            return False
        
        return True
        
    except ImportError:
        logger.error("❌ StateCoordinator no disponible, bloqueando reequilibrio")
        return False
    except Exception as e:
        logger.error(f"❌ Error validando condiciones de reequilibrio: {e}")
        return False
