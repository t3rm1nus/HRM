"""
Control de Dominancia L3 y Jerarquía de Señales

Este módulo maneja la lógica de jerarquía entre señales L1, L2 y L3,
incluyendo el control de dominancia estratégica y la validación de ejecución.
"""

from typing import Dict, Tuple, Any
from datetime import datetime, timezone

from core.logging import logger
from l3_strategy.l3_dominance import should_l3_block_l2_signals


def should_execute_with_l3_dominance(l2_signal: Dict, l3_info: Dict) -> Tuple[bool, str]:
    """
    Decide si una señal L2 debe ejecutarse basado en la lógica corregida de dominancia L3.
    
    Usa la lógica de fix_l3_dominance.py para manejar correctamente la dominancia L3.
    Solo bloquea señales L2 cuando L3 HOLD tiene confianza > 0.90 (certeza muy alta)
    
    REGLA ESPECIAL: "VENTA TÁCTICA DE SALIDA LIMPIA"
    Incluso si L3 normalmente bloquearía señales L2, permite señales de VENTA cuando:
    - tiene_posición (hay una posición actual en el símbolo)
    - l3_confidence < 0.6 (el sistema muestra duda)
    - l3_regime == "TRENDING" (el mercado está en tendencia)
    
    Args:
        l2_signal: Dict con información de la señal L2 (acción, símbolo, confianza, etc.)
        l3_info: Dict con información del régimen L3 (régimen, señal, confianza, allow_l2)
        
    Returns:
        tuple: (debe_ejecutar: bool, razón: str)
    """
    # Extraer información L3
    l3_signal = l3_info.get('signal', 'hold')
    l3_confidence = l3_info.get('confidence', 0.0)
    regime = l3_info.get('regime', 'unknown')
    l3_allow_l2 = l3_info.get('allow_l2_signals', l3_info.get('allow_l2', True))
    symbol = l2_signal.get('symbol', 'UNKNOWN')
    action = l2_signal.get('action', 'hold')

    # Obtener estado de posición para la decisión de dominancia L3
    try:
        # Verificar si el portfolio_manager está disponible en el scope global
        if 'portfolio_manager' in globals():
            pm = globals()['portfolio_manager']
            position_balance = pm.get_balance(symbol)
            has_position = position_balance > 0.00001  # Umbral mínimo para considerar posición
        else:
            # Fallback: asumir que tiene posición si se generan señales de VENTA
            has_position = True
            logger.warning(f"⚠️ Portfolio manager no accesible, asumiendo tiene_posición=True para {symbol}")
    except Exception as e:
        logger.warning(f"⚠️ Error verificando posición para {symbol}: {e}, asumiendo tiene_posición=True")
        has_position = True

    # ========================================================================================
    # LÓGICA NORMAL DE DOMINANCIA L3 (si la excepción táctica no se aplica)
    # ========================================================================================

    # Usar lógica de dominancia corregida con nueva excepción quirúrgica
    should_block = should_l3_block_l2_signals(
        l3_signal=l3_signal,
        l3_confidence=l3_confidence,
        regime=regime,
        current_allocation={},  # Se obtendría de datos del portfolio
        target_allocation=l3_info.get('asset_allocation', {}),
        l2_signal_action=action,  # Pasar acción L2 para excepción quirúrgica
        has_position=has_position,  # Pasar estado de posición para excepción quirúrgica
        allow_l2_signals=l3_allow_l2
    )

    if should_block:
        reason = f"L3 {l3_signal.upper()} (conf={l3_confidence:.2f}) bloquea L2 en régimen {regime}"
        logger.warning(f"🚫 DOMINANCIA L3: {reason}")
        return False, reason
    else:
        reason = f"Dominancia L3 relajada (conf={l3_confidence:.2f} < 0.90) - L2 permitido en {regime}"
        logger.info(f"🔓 {reason}")
        return True, reason


def validate_signal_execution_hierarchy(l1_signals: list, l2_signals: list, l3_info: Dict) -> Dict:
    """
    Valida la jerarquía de ejecución de señales entre L1, L2 y L3.
    
    Args:
        l1_signals: Lista de señales L1
        l2_signals: Lista de señales L2
        l3_info: Información del régimen L3
        
    Returns:
        Dict con señales validadas y estadísticas
    """
    validated_signals = {
        'l1_signals': [],
        'l2_signals': [],
        'blocked_signals': [],
        'execution_stats': {
            'l1_total': len(l1_signals),
            'l2_total': len(l2_signals),
            'l2_blocked': 0,
            'l2_allowed': 0
        }
    }
    
    # Procesar señales L2 con validación de dominancia L3
    for signal in l2_signals:
        should_execute, reason = should_execute_with_l3_dominance(signal, l3_info)
        
        if should_execute:
            validated_signals['l2_signals'].append(signal)
            validated_signals['execution_stats']['l2_allowed'] += 1
            logger.debug(f"✅ Señal L2 {signal.get('symbol', 'UNKNOWN')} {signal.get('action', 'hold').upper()} permitida: {reason}")
        else:
            validated_signals['blocked_signals'].append({
                'signal': signal,
                'reason': reason,
                'blocked_at': datetime.now(timezone.utc).isoformat()
            })
            validated_signals['execution_stats']['l2_blocked'] += 1
            logger.warning(f"🚫 Señal L2 {signal.get('symbol', 'UNKNOWN')} {signal.get('action', 'hold').upper()} bloqueada: {reason}")
    
    # Señales L1 siempre pasan (son fundamentales)
    validated_signals['l1_signals'] = l1_signals
    
    return validated_signals


def get_signal_priority_info(l2_signal: Dict, l3_info: Dict) -> Dict:
    """
    Obtiene información de prioridad para una señal L2 específica.
    
    Args:
        l2_signal: Señal L2 a analizar
        l3_info: Información del régimen L3
        
    Returns:
        Dict con información de prioridad y validación
    """
    should_execute, reason = should_execute_with_l3_dominance(l2_signal, l3_info)
    
    priority_info = {
        'signal': l2_signal,
        'l3_info': l3_info,
        'should_execute': should_execute,
        'execution_reason': reason,
        'priority_level': 'HIGH' if should_execute else 'BLOCKED',
        'l3_confidence': l3_info.get('confidence', 0.0),
        'l3_signal': l3_info.get('signal', 'hold'),
        'regime': l3_info.get('regime', 'unknown'),
        'allow_l2': l3_info.get('allow_l2', True)
    }
    
    return priority_info


def log_signal_hierarchy_decision(l2_signal: Dict, l3_info: Dict, decision: bool, reason: str):
    """
    Registra la decisión de jerarquía de señales para auditoría.
    
    Args:
        l2_signal: Señal L2 procesada
        l3_info: Información del régimen L3
        decision: Decisión de ejecución
        reason: Razón de la decisión
    """
    symbol = l2_signal.get('symbol', 'UNKNOWN')
    action = l2_signal.get('action', 'hold')
    l3_signal = l3_info.get('signal', 'hold')
    l3_confidence = l3_info.get('confidence', 0.0)
    regime = l3_info.get('regime', 'unknown')
    
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'signal': {
            'symbol': symbol,
            'action': action,
            'confidence': l2_signal.get('confidence', 0.0)
        },
        'l3_context': {
            'signal': l3_signal,
            'confidence': l3_confidence,
            'regime': regime,
            'allow_l2': l3_info.get('allow_l2', True)
        },
        'decision': {
            'execute': decision,
            'reason': reason
        }
    }
    
    if decision:
        logger.info(f"✅ DECISIÓN JERARQUÍA: {symbol} {action.upper()} - {reason}")
    else:
        logger.warning(f"🚫 DECISIÓN JERARQUÍA: {symbol} {action.upper()} - {reason}")
    
    # Guardar en archivo de logs de decisiones
    try:
        import os
        import json
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "signal_hierarchy_decisions.json")
        
        # Leer decisiones existentes
        existing_decisions = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    existing_decisions = json.load(f)
                except json.JSONDecodeError:
                    existing_decisions = []
        
        # Añadir nueva decisión
        existing_decisions.append(log_entry)
        
        # Guardar decisiones actualizadas
        with open(log_file, 'w') as f:
            json.dump(existing_decisions, f, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"❌ Error guardando decisión de jerarquía: {e}")


def get_hierarchy_summary(l2_signals: list, l3_info: Dict) -> Dict:
    """
    Obtiene un resumen de la jerarquía de señales para reporting.
    
    Args:
        l2_signals: Lista de señales L2
        l3_info: Información del régimen L3
        
    Returns:
        Dict con resumen de la jerarquía
    """
    total_signals = len(l2_signals)
    allowed_signals = 0
    blocked_signals = 0
    blocking_reasons = {}
    
    for signal in l2_signals:
        should_execute, reason = should_execute_with_l3_dominance(signal, l3_info)
        
        if should_execute:
            allowed_signals += 1
        else:
            blocked_signals += 1
            if reason in blocking_reasons:
                blocking_reasons[reason] += 1
            else:
                blocking_reasons[reason] = 1
    
    summary = {
        'total_l2_signals': total_signals,
        'allowed_signals': allowed_signals,
        'blocked_signals': blocked_signals,
        'allowance_rate': allowed_signals / total_signals if total_signals > 0 else 0,
        'blocking_reasons': blocking_reasons,
        'l3_context': {
            'signal': l3_info.get('signal', 'hold'),
            'confidence': l3_info.get('confidence', 0.0),
            'regime': l3_info.get('regime', 'unknown'),
            'allow_l2': l3_info.get('allow_l2', True)
        },
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    return summary