# range_detector.py
import pandas as pd
import numpy as np
from core.logging import logger

def classify_regime(indicators):
    """
    Clasifica el régimen de mercado: RANGE, TREND o VOLATILE.

    Parámetros:
        indicators (dict): debe contener indicadores calculados por technical_indicators.py
            - 'bollinger_upper', 'bollinger_lower', 'bollinger_middle'
            - 'close' (precio actual)
            - opcionales: 'sma_20', 'sma_50', 'rsi', etc.

    Retorna:
        (regime: str, confidence: float)
    """
    try:
        # Required indicators that are actually calculated by technical_indicators.py
        required = ['bollinger_upper', 'bollinger_lower', 'bollinger_middle', 'close']
        if not all(ind in indicators for ind in required):
            logger.warning("⚠️ Indicadores insuficientes para clasificar régimen - faltan requeridos")
            return "RANGE", 0.3

        price = indicators['close']
        bb_width = indicators['bollinger_upper'] - indicators['bollinger_lower']
        bb_width_pct = bb_width / indicators['bollinger_middle'] if indicators['bollinger_middle'] != 0 else 0

        # Use RSI as volatility proxy (since ATR isn't calculated)
        rsi = indicators.get('rsi', 50)
        # High RSI deviation from 50 indicates volatility
        rsi_volatility = abs(rsi - 50) / 50

        # Use SMA slope as trend indicator (SMA_20 vs SMA_50 if available)
        slope = 0.0
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            if sma_50 != 0:
                slope = (sma_20 - sma_50) / sma_50
        # Alternative: use momentum_20 as trend indicator
        elif 'momentum_20' in indicators and price != 0:
            slope = indicators['momentum_20'] / price

        # --- Lógica de clasificación actualizada --- (más permisiva para RANGE)
        if bb_width_pct < 0.05 and rsi_volatility < 0.5:  # Rango: moderate BB (5%), moderate RSI volatility
            regime = "RANGE"
            confidence = 0.8
        elif rsi_volatility > 0.7:  # Volatile: very high RSI volatility (extreme readings)
            regime = "VOLATILE"
            confidence = min(1.0, 0.6 + rsi_volatility)
        elif abs(slope) > 0.01:  # Trend: moderate slope in moving averages
            regime = "TREND"
            confidence = min(0.9, 0.5 + abs(slope) * 10)
        else:
            # fallback: rango moderado (más común en mercados crypto)
            regime = "RANGE"
            confidence = 0.6

        logger.info(f"📊 Régimen clasificado: {regime} "
                    f"[BB_width={bb_width_pct:.3f}, RSI_vol={rsi_volatility:.3f}, Slope={slope:.4f}]")
        return regime, confidence

    except Exception as e:
        logger.error(f"❌ Error clasificando régimen de mercado: {e}")
        return "RANGE", 0.3


def detect_range_market(indicators):
    """
    Detecta si el mercado está en un régimen de RANGE.
    Retorna boolean: True si detecta RANGE, False para TREND o VOLATILE.
    """
    try:
        regime, confidence = classify_regime(indicators)
        return regime == "RANGE" and confidence > 0.5
    except Exception as e:
        logger.error(f"❌ Error detectando rango: {e}")
        return False


def range_trading_signals(price, indicators):
    """
    Genera señales de trading mejoradas para mercados en rango usando mean-reversion.
    Combina RSI, BB position y momentum para señales más precisas.
    Retorna: "buy", "sell", "hold"
    """
    try:
        required_indicators = ['rsi', 'bollinger_upper', 'bollinger_lower', 'bollinger_middle']
        if not all(ind in indicators for ind in required_indicators):
            logger.warning("⚠️ Indicadores insuficientes para señales de rango mejoradas")
            return "hold"

        rsi = indicators['rsi']
        bb_upper = indicators['bollinger_upper']
        bb_lower = indicators['bollinger_lower']
        bb_middle = indicators['bollinger_middle']

        # Calcular posición dentro de las bandas de Bollinger (0-1)
        if bb_upper > bb_lower:
            bb_position = (price - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5

        # Calcular momentum corto (5 periodos) si disponible
        momentum = indicators.get('momentum_5', 0)

        # Lógica mean-reversion
        buy_condition = (
            rsi < 35 and
            bb_position < 0.2 and
            momentum < -0.005
        )

        sell_condition = (
            rsi > 65 and
            bb_position > 0.8 and
            momentum > 0.005
        )

        if buy_condition:
            signal = "buy"
            confidence = min(0.8, 0.5 + abs(bb_position - 0.2) * 2)
        elif sell_condition:
            signal = "sell"
            confidence = min(0.8, 0.5 + abs(bb_position - 0.8) * 2)
        else:
            signal = "hold"
            confidence = 0.3

        logger.info(f"📈 Señal de rango mejorada: {signal} (RSI: {rsi:.2f}, BB Pos: {bb_position:.2f}, "
                    f"Momentum: {momentum:.4f}, Confianza: {confidence:.2f})")
        return signal

    except Exception as e:
        logger.error(f"❌ Error generando señales de rango mejoradas: {e}")
        return "hold"
