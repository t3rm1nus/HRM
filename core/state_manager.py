# core/state_manager.py - Gestión del estado del sistema

import pandas as pd
from core.logging import log_event
from l2_tactic.models import L2State

def initialize_state(symbols):
    """Inicializa el estado del sistema"""
    from l2_tactic.models import L2State
    return {
        'mercado': {symbol: {} for symbol in symbols},
        'estrategia': 'neutral',
        'portfolio': {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0},
        'universo': symbols,
        'exposicion': {symbol: 0.0 for symbol in symbols},
        "signals": [],
        'ordenes': [],
        'riesgo': {},
        'deriva': False,
        'ciclo_id': 0,
        'l2': L2State(),
    }

async def log_cycle_data(state, cycle_id, ciclo_start):
    """
    Registra métricas de cada ciclo de trading (L3 → L2 → L1).
    Usa log_event centralizado y resume señales, órdenes y estrategia.
    """
    from core.logging import log_cycle_data as core_log_cycle_data
    
    # Actualizar estadísticas del ciclo
    from l2_tactic.models import L2State

    now = pd.Timestamp.utcnow()
    cycle_time = (now - ciclo_start).total_seconds()

    # Obtener señales desde state['l2'] soportando L2State o dict
    l2_obj = state.get("l2")
    if isinstance(l2_obj, L2State):
        signals = getattr(l2_obj, "signals", []) or []
    elif isinstance(l2_obj, dict):
        signals = l2_obj.get("signals", []) or []
    else:
        # fallback a state['signals'] si existe
        signals = state.get("signals", []) or []

    orders = state.get("ordenes", []) or []
    filled_count = sum(1 for o in orders if (o or {}).get("status") == "filled")
    rejected_count = sum(1 for o in orders if (o or {}).get("status") == "rejected")

    cycle_stats = {
        'cycle_time': cycle_time,
        'signals_count': len(signals),
        'orders_count': filled_count,
        'rejected_orders': rejected_count
    }
    
    # Actualizar state con stats
    state['cycle_stats'] = cycle_stats
    state['portfolio'] = state.get('portfolio', {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0})
    
    # Usar el logger centralizado
    await core_log_cycle_data(state, cycle_id, ciclo_start)

def validate_state_structure(state):
    """Valida y corrige que el state tenga la estructura mínima requerida"""
    from l2_tactic.models import L2State
    from core.logging import logger
    
    logger.debug(f"[validate_state_structure] Validando state type: {type(state)}")
    
    # Asegurar que state es un dict
    if not isinstance(state, dict):
        logger.warning("⚠️ State no es dict, inicializando...")
        state = {}
    
    # Asegurar que l2 es L2State
    if not isinstance(state.get("l2"), L2State):
        logger.info("ℹ️ Inicializando L2State...")
        signals = []
        if isinstance(state.get("l2"), dict):
            signals = state["l2"].get("signals", [])
            logger.debug(f"Recuperando {len(signals)} señales existentes")
        
        l2_state = L2State()
        l2_state.signals = signals
        state["l2"] = l2_state
    
    # Asegurar otros campos básicos
    state.setdefault("mercado", {})
    state.setdefault("estrategia", "neutral")
    state.setdefault("portfolio", {"BTCUSDT": 0.0, "ETHUSDT": 0.0, "USDT": 3000.0})
    state.setdefault("signals", [])
    state.setdefault("ordenes", [])
    state.setdefault("riesgo", {})
    state.setdefault("deriva", False)
    state.setdefault("ciclo_id", 0)
    
    logger.debug(f"[validate_state_structure] Salida state['l2'] tipo: {type(state.get('l2'))}")
    return state
