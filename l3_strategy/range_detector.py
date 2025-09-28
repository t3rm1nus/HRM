# range_detector.py
import pandas as pd
import numpy as np
from core.logging import logger

def detect_range_market(indicators):
    """
    Detecta si el mercado está en un rango lateral.
    Retorna True si está en range, False si no.
    """
    try:
        # Verificar que tenemos los indicadores necesarios
        required_indicators = ['bollinger_upper', 'bollinger_lower', 'bollinger_middle']
        if not all(ind in indicators for ind in required_indicators):
            logger.warning("⚠️ Indicadores insuficientes para detectar rango de mercado")
            return False

        # Calcular ancho de las bandas de Bollinger
        bb_width = indicators['bollinger_upper'] - indicators['bollinger_lower']
        bb_width_pct = bb_width / indicators['bollinger_middle']

        # Range si las bandas están estrechas (< 2%)
        is_range = bb_width_pct < 0.02

        logger.info(f"📊 Detección de rango: {'Sí' if is_range else 'No'} (Ancho BB: {bb_width_pct:.4f})")
        return is_range

    except Exception as e:
        logger.error(f"❌ Error detectando rango de mercado: {e}")
        return False

def range_trading_signals(price, indicators):
    """
    Genera señales de trading para mercados en rango usando mean-reversion.
    Retorna: "buy", "sell", "hold"
    """
    try:
        # Verificar que tenemos RSI
        if 'rsi' not in indicators:
            logger.warning("⚠️ RSI no disponible para señales de rango")
            return "hold"

        rsi = indicators['rsi']

        # Lógica de mean-reversion para rangos
        if rsi < 40:  # Sobrevendido en rango
            signal = "buy"
        elif rsi > 60:  # Sobrecomprado en rango
            signal = "sell"
        else:
            signal = "hold"

        logger.info(f"📈 Señal de rango: {signal} (RSI: {rsi:.2f})")
        return signal

    except Exception as e:
        logger.error(f"❌ Error generando señales de rango: {e}")
        return "hold"
