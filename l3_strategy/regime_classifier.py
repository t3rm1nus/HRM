# regime_classifier.py
import pandas as pd
import numpy as np
from core.logging import logger

def clasificar_regimen(datos_mercado):
    """
    Clasifica el régimen de mercado basado en datos de BTC y tendencias.
    Retorna: 'bull', 'bear', 'neutral'
    """
    try:
        # Extraer datos de BTC
        btc_data = datos_mercado.get("BTCUSDT", {})
        if not isinstance(btc_data, dict) or 'close' not in btc_data:
            logger.warning("⚠️ Datos de BTC insuficientes para clasificación de régimen")
            return "neutral"

        # Si es un diccionario simple, convertir a serie temporal básica
        if isinstance(btc_data['close'], (int, float)):
            # Simular una serie corta para análisis básico
            prices = [btc_data['close']] * 20  # Simular 20 periodos iguales
        else:
            prices = btc_data['close'][-20:] if len(btc_data['close']) >= 20 else btc_data['close']

        if len(prices) < 5:
            return "neutral"

        # Calcular indicadores básicos
        prices = pd.Series(prices)

        # Tendencia: comparación de precio actual vs promedio móvil
        current_price = prices.iloc[-1]
        ma_10 = prices.rolling(10).mean().iloc[-1] if len(prices) >= 10 else prices.mean()
        ma_20 = prices.mean()

        # Volatilidad: desviación estándar
        volatility = prices.std() / prices.mean()

        # Momentum: cambio porcentual reciente
        if len(prices) >= 5:
            momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        else:
            momentum = 0.0

        # Lógica de clasificación más dinámica - no quedarse atascado en neutral
        trend_strength = abs(momentum)

        # Bull market: precio por encima de MA y momentum positivo fuerte
        if current_price > ma_10 * 1.01 and momentum > 0.01:
            regime = "bull"
        # Bear market: precio por debajo de MA y momentum negativo fuerte
        elif current_price < ma_10 * 0.99 and momentum < -0.01:
            regime = "bear"
        # Volatile market: alta volatilidad independientemente de dirección
        elif volatility > 0.03:
            regime = "volatile"
        # Range market: movimiento lateral con baja volatilidad
        elif abs(current_price - ma_10) / ma_10 < 0.005 and volatility < 0.02:
            regime = "range"
        # Default to bull if slight uptrend, bear if slight downtrend
        elif momentum > 0.005:
            regime = "bull"
        elif momentum < -0.005:
            regime = "bear"
        else:
            regime = "neutral"  # Solo como último recurso

        logger.info(f"📈 Régimen detectado: {regime} (Precio: {current_price:.2f}, MA10: {ma_10:.2f}, Momentum: {momentum:.4f})")
        return regime

    except Exception as e:
        logger.error(f"❌ Error clasificando régimen: {e}")
        return "neutral"
