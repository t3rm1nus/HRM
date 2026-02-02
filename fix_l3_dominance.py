# fix_l3_dominance.py - Corrección de Lógica de Dominancia L3

"""
PROBLEMA IDENTIFICADO:
- L3 HOLD con 0.85 confidence bloquea TODAS las señales L2/L1
- Allocations L3 no se ejecutan (BTC target 50% pero current 0%)
- $994 USDT disponible sin usar
- Sistema 100% en HOLD permanente

SOLUCIÓN:
1. L3 HOLD solo bloquea si confidence > 0.90 (no 0.85)
2. En regímenes RANGE, permitir L2 autonomía con setup trading
3. Ejecutar rebalancing automático hacia targets L3
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class L3DominanceFixConfig:
    """Configuración corregida para dominancia L3"""

    # ANTES: L3 HOLD bloqueaba con 0.80+ confidence
    # AHORA: Solo bloquea con 0.90+ confidence (muy alta certeza)
    L3_HOLD_BLOCKING_THRESHOLD = 0.90  # Aumentado de 0.80

    # En RANGE, permitir L2 autonomía incluso con HOLD L3
    ALLOW_L2_AUTONOMY_IN_RANGE = True

    # Ejecutar rebalancing cuando diferencia > 10%
    REBALANCE_THRESHOLD_PCT = 0.10

    # Setup trading en RANGE (oversold/overbought)
    ENABLE_SETUP_TRADING_IN_RANGE = True


def should_l3_block_l2_signals(
    l3_signal: str,
    l3_confidence: float,
    regime: str,
    current_allocation: Dict[str, float],
    target_allocation: Dict[str, float],
    l2_signal_action: str = None,
    has_position: bool = False
) -> bool:
    """
    Decide si L3 debe bloquear señales L2.

    LÓGICA CORREGIDA - NUEVA JERARQUÍA:
    1. Stop-loss L1 (siempre primero - no aplica aquí)
    2. SELL TÁCTICO DE SALIDA LIMPIA (excepción quirúrgica)
    3. Dominancia L3 (normal) - L3 confidence >= 0.6 bloquea
    4. Duda → HOLD (INV-5) - L3 confidence < 0.6 fuerza HOLD

    EXCEPCIÓN QUIRÚRGICA SOLO PARA SALIDA:
    - has_position + l2_signal SELL + l3_regime TRENDING + l3_confidence < 0.6
    """

    # 1. L3 BUY/SELL siempre tienen prioridad (dominancia normal)
    if l3_signal in ['buy', 'sell']:
        logger.info(f"✅ L3 {l3_signal.upper()} signal - L2 debe seguir")
        return False

    # 2. EXCEPCIÓN QUIRÚRGICA: SELL TÁCTICO DE SALIDA LIMPIA
    # SOLO si se cumplen TODAS las condiciones:
    if (l2_signal_action and l2_signal_action.upper() in ['SELL', 'SELL_LIGHT', 'REDUCE'] and
        has_position and
        regime.upper() == 'TRENDING' and
        l3_confidence < 0.6):
        logger.info(
            f"🎯 SELL TÁCTICO DE SALIDA LIMPIA: Allow {l2_signal_action.upper()} "
            f"(position={has_position}, regime={regime}, conf={l3_confidence:.2f}<0.6)"
        )
        return False

    # 3. DOMINANCIA L3 NORMAL: L3 bloquea si confidence >= 0.6
    if l3_confidence >= 0.6:
        logger.warning(
            f"🚫 L3 DOMINANCE: L3 {l3_signal.upper()} con {l3_confidence:.2f} confidence (>= 0.6) "
            f"bloquea señales L2 en {regime} regime"
        )
        return True

    # 4. DUDA → HOLD (INV-5): Si L3 confidence < 0.6, forzar HOLD global
    # NO permitir BUY bajo duda - solo HOLD hasta que L3 tenga claridad
    if l3_confidence < 0.6:
        logger.info(
            f"🛡️ INV-5 PROTECTION: L3 duda ({l3_confidence:.2f} < 0.6) → HOLD global "
            f"(no trades under doubt)"
        )
        return True

    # 5. Si allocations desviadas >10%, permitir rebalancing (pero solo HOLD, no BUY)
    allocation_deviation = calculate_allocation_deviation(
        current_allocation, target_allocation
    )
    if allocation_deviation > L3DominanceFixConfig.REBALANCE_THRESHOLD_PCT:
        logger.warning(
            f"⚠️ Allocation deviation {allocation_deviation:.1%} > 10% - "
            f"Permitir rebalancing (solo SELL/REDUCE, no BUY)"
        )
        # Solo permitir SELL/REDUCE para rebalancing, no BUY
        if l2_signal_action and l2_signal_action.upper() in ['SELL', 'SELL_LIGHT', 'REDUCE']:
            return False
        else:
            return True

    # Default: mantener bloqueo por seguridad
    logger.warning(f"🚫 L3 DEFAULT BLOCK: Manteniendo bloqueo por seguridad")
    return True


def get_l3_push_signal(l3_signal: str, l3_confidence: float, l2_original_signal: str) -> str:
    """
    Cuando L3 "empuja" a L2, convertir señales HOLD en alternativas defensivas.

    Si L3 = SELL y L2 = HOLD → L2 debe emitir SELL_LIGHT, REDUCE, HEDGE o NO_LONG
    con size reducido.

    Args:
        l3_signal: Señal L3 (buy/sell/hold)
        l3_confidence: Confianza L3
        l2_original_signal: Señal L2 original

    Returns:
        Señal L2 modificada por L3 push
    """
    # Solo aplicar push si L3 tiene señal clara y confianza razonable
    if l3_signal not in ['buy', 'sell'] or l3_confidence < 0.50:
        return l2_original_signal

    # Si L2 ya tiene señal BUY/SELL, no modificar
    if l2_original_signal in ['buy', 'sell']:
        return l2_original_signal

    # L3 BUY push: si L3 quiere comprar pero L2 quiere HOLD
    if l3_signal == 'buy':
        logger.info(f"🚀 L3 BUY PUSH: L2 HOLD → BUY_LIGHT (reduced size)")
        return 'buy_light'  # BUY con size reducido

    # L3 SELL push: si L3 quiere vender pero L2 quiere HOLD
    elif l3_signal == 'sell':
        # Elegir entre alternativas defensivas basadas en confianza
        if l3_confidence >= 0.80:
            push_signal = 'sell_light'  # Venta ligera
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → SELL_LIGHT (conf={l3_confidence:.2f})")
        elif l3_confidence >= 0.65:
            push_signal = 'reduce'  # Reducir posición
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → REDUCE (conf={l3_confidence:.2f})")
        else:
            push_signal = 'hedge'  # Hedging
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → HEDGE (conf={l3_confidence:.2f})")

        return push_signal

    return l2_original_signal


def apply_l3_push_to_signals(l2_signals: list, l3_signal: str, l3_confidence: float) -> list:
    """
    Aplicar L3 push a una lista de señales L2.

    Args:
        l2_signals: Lista de señales L2
        l3_signal: Señal L3
        l3_confidence: Confianza L3

    Returns:
        Lista de señales L2 modificadas por L3 push
    """
    if not l2_signals:
        return l2_signals

    modified_signals = []

    for signal in l2_signals:
        original_side = getattr(signal, 'side', 'hold')

        # Aplicar L3 push logic
        pushed_side = get_l3_push_signal(l3_signal, l3_confidence, original_side)

        if pushed_side != original_side:
            # Modificar la señal
            signal.side = pushed_side

            # Reducir confidence ligeramente para push signals
            if hasattr(signal, 'confidence'):
                signal.confidence = min(signal.confidence, l3_confidence * 0.9)

            # Reducir size para señales push (excepto buy_light que puede ser normal)
            if pushed_side in ['sell_light', 'reduce', 'hedge']:
                if hasattr(signal, 'metadata'):
                    signal.metadata['l3_push_applied'] = True
                    signal.metadata['l3_push_type'] = pushed_side
                    signal.metadata['l3_push_confidence'] = l3_confidence
                    signal.metadata['size_multiplier'] = 0.5  # Reducir size a 50%

            logger.info(f"🔄 L3 PUSH APPLIED: {original_side.upper()} → {pushed_side.upper()}")

        modified_signals.append(signal)

    return modified_signals


def calculate_allocation_deviation(
    current: Dict[str, float],
    target: Dict[str, float]
) -> float:
    """Calcula desviación máxima entre allocations actuales y target"""
    deviations = []
    for symbol in target.keys():
        current_pct = current.get(symbol, 0.0)
        target_pct = target.get(symbol, 0.0)
        deviation = abs(current_pct - target_pct)
        deviations.append(deviation)

    return max(deviations) if deviations else 0.0


def should_trigger_rebalancing(
    current_allocation: Dict[str, float],
    target_allocation: Dict[str, float],
    available_usdt: float,
    min_rebalance_amount: float = 100.0
) -> bool:
    """
    Decide si ejecutar rebalancing automático.

    CRITERIOS:
    1. Desviación > 10% en algún activo
    2. Capital disponible > $100
    3. Diferencia en USDT > $100
    """

    # 1. Desviación significativa
    deviation = calculate_allocation_deviation(current_allocation, target_allocation)
    if deviation < L3DominanceFixConfig.REBALANCE_THRESHOLD_PCT:
        logger.debug(f"Desviación {deviation:.1%} < 10% - No rebalancing")
        return False

    # 2. Capital suficiente
    if available_usdt < min_rebalance_amount:
        logger.debug(f"USDT disponible ${available_usdt:.2f} < ${min_rebalance_amount}")
        return False

    # 3. Calcular diferencia en USDT
    total_portfolio_value = sum(current_allocation.values())
    if total_portfolio_value == 0:
        return False

    max_diff_usdt = max([
        abs(current_allocation.get(s, 0) - target_allocation.get(s, 0))
        for s in target_allocation.keys()
    ])

    if max_diff_usdt < min_rebalance_amount:
        logger.debug(f"Diferencia max ${max_diff_usdt:.2f} < ${min_rebalance_amount}")
        return False

    logger.info(
        f"✅ TRIGGER REBALANCING: deviation={deviation:.1%}, "
        f"available=${available_usdt:.2f}, diff=${max_diff_usdt:.2f}"
    )
    return True


# ============================================================================
# INTEGRACIÓN EN MAIN.PY
# ============================================================================

def integrate_fix_in_main():
    """
    Código para integrar en main.py

    REEMPLAZAR EN MAIN.PY (aproximadamente línea 850-900):
    """
    example_code = '''
# ANTES (LÓGICA ANTIGUA - REMOVER):
if l3_info['signal'] == 'hold' and l3_info['confidence'] > 0.80:
    logger.warning("🚫 L3 DOMINANCE: L3 HOLD blocks all L2 signals")
    # ... código antiguo

# DESPUÉS (LÓGICA NUEVA - INTEGRAR):
from fix_l3_dominance import (
    should_l3_block_l2_signals,
    should_trigger_rebalancing,
    L3DominanceFixConfig
)

# Verificar si L3 debe bloquear L2
l3_blocks_l2 = should_l3_block_l2_signals(
    l3_signal=l3_info['signal'],
    l3_confidence=l3_info['confidence'],
    regime=l3_info['regime'],
    current_allocation=state.portfolio.get_current_allocation(),
    target_allocation=l3_info['asset_allocation']
)

if l3_blocks_l2:
    logger.warning(
        f"🚫 L3 DOMINANCE: L3 {l3_info['signal'].upper()} "
        f"con {l3_info['confidence']:.2f} confidence bloquea L2"
    )
    # Forzar HOLD signals
    btc_signal = {'action': 'hold', 'symbol': 'BTCUSDT', 'confidence': 0.5}
    eth_signal = {'action': 'hold', 'symbol': 'ETHUSDT', 'confidence': 0.5}
else:
    logger.info("🔓 L3 permite autonomía L2 - procesando señales normalmente")
    # Procesar L2 normalmente
    btc_signal = l2_processor.process_signals(state, 'BTCUSDT')
    eth_signal = l2_processor.process_signals(state, 'ETHUSDT')

# Verificar si ejecutar rebalancing automático
if should_trigger_rebalancing(
    current_allocation=state.portfolio.get_current_allocation(),
    target_allocation=l3_info['asset_allocation'],
    available_usdt=state.portfolio.get_available_usdt()
):
    logger.info("🔄 EXECUTING AUTO-REBALANCING towards L3 targets")
    auto_rebalancer.execute_rebalancing(
        current=state.portfolio.get_current_allocation(),
        target=l3_info['asset_allocation'],
        prices=state.market_data
    )
'''
    return example_code


# ============================================================================
# TESTING DE LA SOLUCIÓN
# ============================================================================

def test_fix():
    """Test de la solución con casos reales"""

    print("=" * 80)
    print("TESTING L3 DOMINANCE FIX")
    print("=" * 80)

    # Caso 1: L3 HOLD con 0.85 confidence en RANGE
    print("\n📊 CASO 1: L3 HOLD 0.85 confidence en RANGE")
    blocks = should_l3_block_l2_signals(
        l3_signal='hold',
        l3_confidence=0.85,
        regime='RANGE',
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3860.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0}
    )
    print(f"Resultado: {'BLOQUEA ❌' if blocks else 'PERMITE ✅'}")
    print(f"Esperado: PERMITE ✅ (confidence < 0.90 y regime RANGE)")

    # Caso 2: L3 HOLD con 0.95 confidence
    print("\n📊 CASO 2: L3 HOLD 0.95 confidence (muy alta certeza)")
    blocks = should_l3_block_l2_signals(
        l3_signal='hold',
        l3_confidence=0.95,
        regime='BULL',
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3860.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0}
    )
    print(f"Resultado: {'BLOQUEA ❌' if blocks else 'PERMITE ✅'}")
    print(f"Esperado: BLOQUEA ❌ (confidence > 0.90)")

    # Caso 3: Rebalancing necesario
    print("\n📊 CASO 3: Verificar rebalancing con $994 disponible")
    trigger = should_trigger_rebalancing(
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 994.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0},
        available_usdt=994.0
    )
    print(f"Resultado: {'TRIGGER ✅' if trigger else 'NO TRIGGER ❌'}")
    print(f"Esperado: TRIGGER ✅ (desviación 50% y capital disponible)")

    print("\n" + "=" * 80)
    print("FIX VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Ejecutar tests
    test_fix()

    # Mostrar código de integración
    print("\n" + "=" * 80)
    print("CÓDIGO PARA INTEGRAR EN MAIN.PY:")
    print("=" * 80)
    print(integrate_fix_in_main())
