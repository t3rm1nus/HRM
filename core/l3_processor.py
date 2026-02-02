"""
Procesamiento L3 Estratégico del Sistema HRM

Este módulo maneja la lógica estratégica de nivel 3, incluyendo:
- Clasificación de régimen de mercado
- Decisión estratégica basada en análisis macro
- Sistema de datos frescos L3
- Gestión de actualizaciones de decisiones estratégicas
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import pandas as pd

from core.logging import logger
from l3_strategy.decision_maker import make_decision
from l3_strategy.regime_classifier import ejecutar_estrategia_por_regimen


def get_l3_decision(market_data: Dict) -> Dict:
    """
    Obtiene una decisión fresca de L3 basada en datos de mercado actuales.
    
    Args:
        market_data: Datos de mercado actuales
        
    Returns:
        Dict con decisión L3 completa
    """
    try:
        # Obtener régimen actual usando el clasificador completo
        regimen_resultado = ejecutar_estrategia_por_regimen(market_data)

        # Crear output L3 con timestamp fresco
        l3_output = {
            'regime': regimen_resultado.get('regime', 'unknown'),
            'signal': regimen_resultado.get('signal', 'hold'),
            'confidence': regimen_resultado.get('confidence', 0.5),
            'market_regime': regimen_resultado.get('regime', 'unknown'),
            'allow_l2_signals': regimen_resultado.get('allow_l2_signal', True),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fresh_update': True,
            'setup_type': regimen_resultado.get('setup_type'),
            'subtype': regimen_resultado.get('subtype', 'unknown')
        }

        # Si hay decisión estratégica disponible, añadirla
        if regimen_resultado and 'regime' in regimen_resultado:
            # FIX 3 - L3 debe aceptar el portfolio local como verdad
            # Obtener el modo del sistema
            try:
                from core.config import get_config
                config = get_config("live")
                system_mode = getattr(config, 'mode', 'unknown')
            except Exception:
                system_mode = 'unknown'
            
            # En modo simulated, usar portfolio local como válido
            # FIX: Pass state instead of balances_synced flag
            state = None
            if system_mode == "simulated":
                state = {
                    'system_state_type': 'NORMAL',
                    'system_state_metadata': {'l3_balance_sync_failed': False}
                }
            
            strategic_decision = make_decision(
                inputs={},
                portfolio_state={},
                market_data=market_data,
                regime_decision=regimen_resultado,
                state=state
            )

            if strategic_decision:
                l3_output.update({
                    'asset_allocation': strategic_decision.get('asset_allocation', {}),
                    'risk_appetite': strategic_decision.get('risk_appetite', 'moderate'),
                    'loss_prevention_filters': strategic_decision.get('loss_prevention_filters', {}),
                    'strategic_guidelines': strategic_decision.get('strategic_guidelines', {}),
                    'exposure_decisions': strategic_decision.get('exposure_decisions', {}),
                    'winning_trade_rules': strategic_decision.get('winning_trade_rules', {})
                })

        # FIX EXTRA - BLIND MODE HANDLING
        # Detectar si estamos en modo blind y establecer fallback flag
        blind_mode = False
        if l3_output.get('regime') == 'unknown' and l3_output.get('confidence', 0) < 0.1:
            blind_mode = True
            logger.warning("👁️ BLIND MODE detectado - L3 sin confianza suficiente")
        
        # Si hay error en el régimen, también considerar como blind mode
        if l3_output.get('regime') == 'error':
            blind_mode = True
            logger.warning("👁️ BLIND MODE detectado - Error en clasificación de régimen")
        
        l3_output['blind_mode'] = blind_mode

        logger.info(f"✅ Decisión L3 generada: régimen={l3_output['regime']}, señal={l3_output['signal']}, confianza={l3_output['confidence']:.2f}, blind_mode={blind_mode}")
        return l3_output

    except Exception as e:
        logger.error(f"❌ Error generando decisión L3: {e}")
        # Fallback a decisión básica con blind_mode=True
        return {
            'regime': 'error',
            'signal': 'hold',
            'confidence': 0.0,
            'market_regime': 'error',
            'allow_l2_signals': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fresh_update': True,
            'setup_type': None,
            'subtype': 'error',
            'blind_mode': True  # FIX EXTRA - Indicar que estamos en blind mode
        }


def get_current_regime(market_data: Dict) -> str:
    """
    Obtiene el régimen de mercado actual usando el clasificador de régimen completo.
    
    Args:
        market_data: Datos de mercado actuales
        
    Returns:
        String con el régimen detectado
    """
    try:
        from l3_strategy.regime_classifier import ejecutar_estrategia_por_regimen

        regimen_resultado = ejecutar_estrategia_por_regimen(market_data)

        if regimen_resultado and isinstance(regimen_resultado, dict) and 'regime' in regimen_resultado:
            regime = regimen_resultado['regime']
            confidence = regimen_resultado.get('confidence', 0.0)
            logger.info(f"🎯 Régimen detectado: {regime.upper()} (confianza: {confidence:.2f})")
            return regime
        else:
            logger.warning("⚠️ Clasificación de régimen retornó resultado inválido")
            return 'unknown'

    except Exception as e:
        logger.warning(f"❌ Error detectando régimen actual: {e}")
        return 'unknown'


def should_force_l3_update(l3_decision: Dict, current_regime: str) -> bool:
    """
    Determina si se debe forzar una actualización de L3 basado en condiciones críticas.
    
    Args:
        l3_decision: Decisión L3 actual
        current_regime: Régimen actual detectado
        
    Returns:
        bool: True si se debe forzar la actualización
    """
    # Obtener timestamp de última actualización L3
    l3_timestamp_str = l3_decision.get('timestamp', '')

    if not l3_timestamp_str:
        logger.warning("⚠️ Decisión L3 sin timestamp, forzando actualización")
        return True

    try:
        # Parsear timestamp L3
        l3_timestamp = datetime.fromisoformat(l3_timestamp_str.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)

        # Calcular edad de datos L3
        age_seconds = (current_time - l3_timestamp).total_seconds()
        age_minutes = age_seconds / 60

        logger.debug(f"📅 Edad de datos L3: {age_minutes:.1f} minutos")

        # CONDICIONES PARA FORZAR ACTUALIZACIÓN:
        # 1. Datos > 10 minutos viejos
        if age_minutes > 10:
            logger.warning(f"🚨 Datos L3 ANTIGUOS ({age_minutes:.1f}m > 10m), forzando actualización")
            return True

        # 2. Régimen TRENDING detectado pero L3 dice ERROR
        if current_regime == 'TRENDING' and l3_decision.get('market_regime') == 'ERROR':
            logger.warning(f"🚨 Régimen TRENDING detectado pero L3 ERROR, forzando actualización")
            return True

        # 3. L3 permite L2 pero señal es muy vieja (> 5 minutos)
        if not l3_decision.get('allow_l2_signals', False) and age_minutes > 5:
            logger.warning(f"🚨 L3 bloqueando L2 con datos antiguos ({age_minutes:.1f}m), forzando actualización")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error verificando timestamp L3: {e}")
        return True  # Si hay error, forzar actualización por seguridad


def should_recalculate_l3(l3_decision_cache: Dict, current_regime: str, cycle_id: int) -> Tuple[bool, str]:
    """
    Determina si se debe recalcular L3 basado en condiciones optimizadas.
    
    Args:
        l3_decision_cache: Decisión L3 en caché
        current_regime: Régimen actual detectado
        cycle_id: ID del ciclo actual
        
    Returns:
        Tuple[bool, str]: (debe_recalcular, razón)
    """
    # Verificar condiciones para recálculo
    l3_regime_changed = False
    l3_cache_expired = False
    l3_critical_event = False
    l3_setup_changed = False

    # 1. Cambio de régimen (PRIORIDAD MÁXIMA)
    previous_regime = l3_decision_cache.get('previous_regime')
    if previous_regime != current_regime:
        l3_regime_changed = True
        logger.warning(f"🚨 CAMBIO DE RÉGIMEN: {previous_regime} → {current_regime} (FORZAR RECALCULO)")

    # 2. Cache expirado (> 30 minutos)
    l3_last_update = l3_decision_cache.get('last_update', 0)
    cache_age_minutes = (time.time() - l3_last_update) / 60 if l3_last_update else float('inf')
    if cache_age_minutes > 30:
        l3_cache_expired = True
        logger.warning(f"📅 CACHE L3 EXPIRADO: {cache_age_minutes:.1f}min > 30min (FORZAR RECALCULO)")

    # 3. Eventos críticos (primer ciclo, error, etc.)
    if not l3_decision_cache or cycle_id == 1 or current_regime == 'error':
        l3_critical_event = True
        logger.warning(f"🚨 EVENTO CRÍTICO: ciclo={cycle_id}, sin_cache={not l3_decision_cache}, error_regime={current_regime == 'error'} (FORZAR RECALCULO)")

    # 4. Cambio de setup type
    previous_setup = l3_decision_cache.get('previous_setup_type')
    current_setup = l3_decision_cache.get('setup_type')
    if previous_setup != current_setup and current_setup is not None:
        l3_setup_changed = True
        logger.warning(f"🚨 CAMBIO DE SETUP: {previous_setup} → {current_setup} (FORZAR RECALCULO)")

    # Decisión de recálculo
    should_recalculate = l3_regime_changed or l3_cache_expired or l3_critical_event or l3_setup_changed

    if should_recalculate:
        reasons = []
        if l3_regime_changed: reasons.append("cambio_regimen")
        if l3_cache_expired: reasons.append("cache_expirado")
        if l3_critical_event: reasons.append("evento_critico")
        if l3_setup_changed: reasons.append("cambio_setup")
        
        reason = f"{' | '.join(reasons)}"
        logger.warning(f"🔄 RECALCULO L3 TRIGGERED: {reason}")
        return True, reason
    else:
        logger.debug(f"⏸️ RECALCULO L3 OMITIDO: No se cumplen condiciones de trigger")
        return False, "no_trigger_conditions"


def get_l3_regime_info(l3_decision: Dict) -> Dict:
    """
    Obtiene información resumida del régimen L3 para uso en otros módulos.
    
    Args:
        l3_decision: Decisión L3 completa
        
    Returns:
        Dict con información resumida del régimen
    """
    return {
        'regime': l3_decision.get('regime', 'unknown'),
        'subtype': l3_decision.get('subtype', l3_decision.get('regime', 'unknown')),
        'confidence': l3_decision.get('confidence', 0.5),
        'signal': l3_decision.get('signal', 'hold'),
        'allow_l2': l3_decision.get('allow_l2_signals', True),
        'allow_setup_trades': l3_decision.get('strategic_control', {}).get('allow_setup_trades', False),
        'setup_type': l3_decision.get('setup_type'),
        'l3_output': l3_decision
    }


def is_l3_fallback_active(l3_regime_info: Dict) -> bool:
    """
    Determina si el fallback L3 está activo (modo HOLD GLOBAL).
    
    Args:
        l3_regime_info: Información del régimen L3
        
    Returns:
        bool: True si el fallback está activo
    """
    fallback_conditions = [
        l3_regime_info.get('market_regime') == 'unknown_no_sync',
        l3_regime_info.get('strategic_hold_active', False),
        not l3_regime_info.get('allow_l2_signals', True)
    ]
    
    fallback_active = any(fallback_conditions)
    
    if fallback_active:
        logger.warning("🛡️ FALLBACK L3 ACTIVO: Modo HOLD GLOBAL - Se congelan posiciones y señales tácticas")
    
    return fallback_active


def get_l3_decision_with_fallback(market_data: Dict, l3_cache: Dict = None) -> Dict:
    """
    Obtiene decisión L3 con manejo de fallback integrado.
    
    Args:
        market_data: Datos de mercado actuales
        l3_cache: Cache de decisiones L3
        
    Returns:
        Dict con decisión L3 final
    """
    if l3_cache is None:
        l3_cache = {}
    
    # Obtener régimen actual
    current_regime = get_current_regime(market_data)
    
    # Verificar si se debe forzar actualización
    force_update = should_force_l3_update(l3_cache, current_regime)
    
    # Verificar condiciones de recálculo
    should_recalculate, reason = should_recalculate_l3(l3_cache, current_regime, 0)
    
    # Decidir si obtener nueva decisión
    if force_update or should_recalculate:
        logger.info("🔄 Obteniendo nueva decisión L3...")
        new_decision = get_l3_decision(market_data)
        
        # Actualizar cache
        l3_cache.update({
            'decision': new_decision,
            'last_update': time.time(),
            'previous_regime': current_regime,
            'previous_setup_type': new_decision.get('setup_type'),
            'timestamp': new_decision.get('timestamp')
        })
        
        return new_decision
    else:
        logger.debug("✅ Usando decisión L3 en caché")
        return l3_cache.get('decision', get_l3_decision(market_data))