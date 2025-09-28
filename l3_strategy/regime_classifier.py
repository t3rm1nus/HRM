# regime_classifier_improved.py
import pandas as pd
import numpy as np
from core.logging import logger
from .range_detector import detect_range_market, range_trading_signals
from .filters import apply_range_filters, get_range_filters
from core.technical_indicators import calculate_range_indicators

def clasificar_regimen_mejorado(datos_mercado, symbol="BTCUSDT"):
    """
    Clasificación mejorada de régimen de mercado con multi-timeframe analysis.
    Retorna: 'bull', 'bear', 'range', 'volatile', 'neutral'
    """
    try:
        btc_data = datos_mercado.get(symbol, {})
        
        # Validación más robusta de datos
        if btc_data is None or (isinstance(btc_data, (dict, pd.DataFrame)) and 'close' not in btc_data):
            logger.warning("⚠️ Sin datos suficientes para clasificación")
            return "neutral"
            
        # USAR DATOS REALES DE SERIES TEMPORALES - validar DataFrame
        if isinstance(btc_data, pd.DataFrame) and not btc_data.empty:
            # Es un DataFrame real - usar .tail(20) para últimos 20 periodos
            if len(btc_data) < 20:
                logger.warning(f"⚠️ DataFrame insuficiente: {len(btc_data)} filas")
                return "neutral"
            prices = btc_data['close'].tail(20)  # Usar últimos 20 periodos
        elif isinstance(btc_data, dict) and 'close' in btc_data:
            # Formato legacy (dict/list)
            prices = btc_data['close']
            if not isinstance(prices, (list, np.ndarray)) or len(prices) < 50:
                logger.warning(f"⚠️ Datos insuficientes: {len(prices) if isinstance(prices, (list, np.ndarray)) else 'N/A'} periodos")
                return "neutral"
        else:
            logger.warning("⚠️ Formato de datos no válido para clasificación")
            return "neutral"

        prices = pd.Series(prices)
        
        # ANÁLISIS MULTI-TIMEFRAME
        # Corto plazo (10 periodos)
        ma_10 = prices.rolling(10).mean().iloc[-1]
        momentum_10 = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0
        
        # Medio plazo (20 periodos)  
        ma_20 = prices.rolling(20).mean().iloc[-1]
        momentum_20 = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] if len(prices) >= 20 else 0
        
        # Largo plazo (50 periodos)
        ma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else ma_20
        
        # VOLATILIDAD multi-timeframe
        volatility_10 = prices.rolling(10).std().iloc[-1] / prices.rolling(10).mean().iloc[-1]
        volatility_20 = prices.rolling(20).std().iloc[-1] / prices.rolling(20).mean().iloc[-1]
        
        # TENDENCIA CONFIRMADA (multi-timeframe alignment)
        trend_alignment = sum([
            1 if ma_10 > ma_20 else -1,
            1 if ma_20 > ma_50 else -1,
            1 if momentum_10 > 0 else -1,
            1 if momentum_20 > 0 else -1
        ])
        
        current_price = prices.iloc[-1]
        price_vs_ma10 = (current_price - ma_10) / ma_10
        
        # CLASIFICACIÓN MEJORADA PARA CRYPTO (thresholds más sensibles)
        # 1. Mercado VOLÁTIL (crypto-appropriate thresholds)
        if volatility_20 > 0.015 or volatility_10 > 0.02:  # 1.5% y 2% vs 4% y 5%
            regime = "volatile"

        # 2. BULL MARKET confirmado (multi-timeframe, más sensible)
        elif trend_alignment >= 3 and price_vs_ma10 > 0.005:  # 0.5% vs 0.5%
            regime = "bull"

        # 3. BEAR MARKET confirmado (multi-timeframe, más sensible)
        elif trend_alignment <= -3 and price_vs_ma10 < -0.005:  # -0.5% vs -0.5%
            regime = "bear"

        # 4. RANGE MARKET (precio cerca de MA, baja volatilidad)
        elif abs(price_vs_ma10) < 0.008 and volatility_20 < 0.025:
            regime = "range"

        # 5. SISTEMA MENOS CONSERVADOR - momentum-based por defecto (muy sensible)
        elif momentum_20 > 0.001:  # Ultra-sensible (0.1% vs 0.2%)
            regime = "bull"
        elif momentum_20 < -0.001:  # Ultra-sensible (-0.1% vs -0.2%)
            regime = "bear"
        elif momentum_10 > 0.0005:  # Corto plazo también considerado
            regime = "bull"
        elif momentum_10 < -0.0005:  # Corto plazo también considerado
            regime = "bear"

        else:
            regime = "neutral"

        # LOG DETALLADO
        logger.info(f"📊 Régimen {symbol}: {regime.upper()} "
                   f"[Precio: {current_price:.0f}, MA10: {ma_10:.0f} ({price_vs_ma10:+.2%}), "
                   f"Vol: {volatility_20:.2%}, TrendAlign: {trend_alignment}/4]")
        
        return regime

    except Exception as e:
        logger.error(f"❌ Error en clasificación mejorada: {e}")
        return "neutral"

# Función adicional para detectar cambios de régimen
def detectar_cambio_regimen(regimen_actual, nuevo_regimen, historial_regimen):
    """
    Detecta si hay un cambio significativo de régimen
    """
    if len(historial_regimen) < 3:
        return True  # Primeras detecciones

    # Solo considerar cambio si se mantiene por 2 de 3 últimas detecciones
    ultimos_3 = historial_regimen[-3:]
    conteo_nuevo = ultimos_3.count(nuevo_regimen)

    return conteo_nuevo >= 2 and nuevo_regimen != regimen_actual

def ajustar_estrategia_para_range(market_data, symbol="BTCUSDT"):
    """
    Ajusta la estrategia específicamente para mercados en rango.
    Activa mean-reversion trading con filtros conservadores.
    """
    try:
        logger.info("🎯 Activando estrategia específica para mercado RANGE")

        # Obtener datos del símbolo
        symbol_data = market_data.get(symbol, {})
        if not symbol_data:
            logger.warning(f"⚠️ No hay datos para {symbol}")
            return None

        # Calcular indicadores técnicos incluyendo los de rango
        from core.technical_indicators import calculate_technical_indicators
        indicators = calculate_technical_indicators({symbol: symbol_data})

        if symbol not in indicators or indicators[symbol].empty:
            logger.warning(f"⚠️ No se pudieron calcular indicadores para {symbol}")
            return None

        # Obtener indicadores de la última fila
        last_indicators = indicators[symbol].iloc[-1].to_dict()
        current_price = symbol_data['close'][-1] if isinstance(symbol_data['close'], list) else symbol_data['close']

        # Verificar si efectivamente estamos en rango usando la nueva función
        is_range = detect_range_market(last_indicators)

        if not is_range:
            logger.info("📊 Mercado no confirmado como rango - usando estrategia estándar")
            return None

        # Generar señal de trading específica para rango
        signal = range_trading_signals(current_price, last_indicators)

        # Aplicar filtros específicos de rango
        signal_data = {
            'signal': signal,
            'regime': 'range',
            'confidence': 0.7,  # Confianza base para estrategias de rango
            'indicators': last_indicators
        }

        adjusted_signal = apply_range_filters(signal_data, 'range')

        logger.info(f"🔄 Señal de rango generada: {signal} (Precio: {current_price:.2f})")
        return adjusted_signal

    except Exception as e:
        logger.error(f"❌ Error ajustando estrategia para rango: {e}")
        return None

def ajustar_estrategia_para_tendencia(market_data, regime, symbol="BTCUSDT"):
    """
    Ajusta la estrategia para mercados en tendencia (bull/bear).
    Activa trend-following con parámetros agresivos.
    """
    try:
        logger.info(f"📈 Activando estrategia de tendencia para régimen {regime.upper()}")

        # Configuraciones específicas para tendencias
        trend_settings = {
            'bull': {
                'required_confidence': 0.75,
                'profit_target': 0.025,  # 2.5% target
                'stop_loss': 0.012,      # 1.2% stop
                'max_position_time': 12  # Más tiempo en tendencias
            },
            'bear': {
                'required_confidence': 0.75,
                'profit_target': 0.025,
                'stop_loss': 0.012,
                'max_position_time': 12
            }
        }

        settings = trend_settings.get(regime, trend_settings['bull'])

        # Obtener datos del símbolo
        symbol_data = market_data.get(symbol, {})
        if not symbol_data:
            logger.warning(f"⚠️ No hay datos para {symbol}")
            return None

        # Calcular indicadores técnicos
        from core.technical_indicators import calculate_technical_indicators
        indicators = calculate_technical_indicators({symbol: symbol_data})

        if symbol not in indicators or indicators[symbol].empty:
            logger.warning(f"⚠️ No se pudieron calcular indicadores para {symbol}")
            return None

        # Obtener indicadores de la última fila
        last_indicators = indicators[symbol].iloc[-1].to_dict()
        current_price = symbol_data['close'][-1] if isinstance(symbol_data['close'], list) else symbol_data['close']

        # Lógica de señales para tendencias
        if regime == 'bull':
            # En bull: buscar entradas en retrocesos, confirmar con momentum
            signal = "buy" if last_indicators.get('rsi', 50) < 70 else "hold"
        elif regime == 'bear':
            # En bear: buscar entradas en rebotes, confirmar con momentum
            signal = "sell" if last_indicators.get('rsi', 50) > 30 else "hold"
        else:
            signal = "hold"

        # Aplicar configuración de tendencia
        signal_data = {
            'signal': signal,
            'regime': regime,
            'confidence': settings['required_confidence'],
            'profit_target': settings['profit_target'],
            'stop_loss': settings['stop_loss'],
            'max_position_time': settings['max_position_time'],
            'indicators': last_indicators
        }

        logger.info(f"📊 Señal de tendencia generada: {signal} para {regime.upper()} (Precio: {current_price:.2f})")
        return signal_data

    except Exception as e:
        logger.error(f"❌ Error ajustando estrategia para tendencia: {e}")
        return None

def ejecutar_estrategia_por_regimen(market_data, symbol="BTCUSDT"):
    """
    Función principal que integra todo el sistema de regímenes.
    Clasifica el régimen y ejecuta la estrategia apropiada.
    """
    try:
        # Clasificar régimen actual
        regimen_actual = clasificar_regimen_mejorado(market_data, symbol)

        # Ejecutar estrategia específica según régimen
        if regimen_actual == "range":
            estrategia_resultado = ajustar_estrategia_para_range(market_data, symbol)

        elif regimen_actual in ["bull", "bear"]:
            estrategia_resultado = ajustar_estrategia_para_tendencia(market_data, regimen_actual, symbol)

        elif regimen_actual == "volatile":
            # Para mercados volátiles, usar estrategia conservadora
            logger.info("⚡ Mercado volátil detectado - usando estrategia conservadora")
            estrategia_resultado = {
                'signal': 'hold',
                'regime': 'volatile',
                'confidence': 0.5,
                'message': 'Mercado volátil - esperando estabilización'
            }

        else:  # neutral
            logger.info("😐 Mercado neutral - sin señal clara")
            estrategia_resultado = {
                'signal': 'hold',
                'regime': 'neutral',
                'confidence': 0.4,
                'message': 'Mercado neutral - esperando dirección clara'
            }

        # Log del resultado final
        if estrategia_resultado:
            logger.info(f"🎯 Estrategia ejecutada para régimen {regimen_actual.upper()}: "
                       f"Señal={estrategia_resultado.get('signal', 'N/A')}, "
                       f"Confianza={estrategia_resultado.get('confidence', 'N/A')}")

        return estrategia_resultado

    except Exception as e:
        logger.error(f"❌ Error ejecutando estrategia por régimen: {e}")
        return None
