"""
Función principal de procesamiento L2 para integración con main.py
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from .models import TacticalSignal, SignalDirection, SignalSource
from .config import L2Config
from .signal_generator import SignalGenerator

# Configurar logger específico para L2
logger = logging.getLogger("l2_tactic")

def procesar_l2(state: Dict[str, Any], config: L2Config) -> Dict[str, Any]:


    """
    Función principal de L2 que procesa el estado y genera señales tácticas.
    
    Args:
        state: Estado actual del sistema desde L3
        
    Returns:
        state actualizado con señales L2
    """
    logger.info("🎯 INICIANDO procesamiento L2 - Nivel Táctico")
    
    try:      
        
        # 2. Inicializar generador de señales
        logger.info("🚀 Inicializando SignalGenerator...")
        try:
            signal_generator = SignalGenerator(config)
            logger.info("✅ SignalGenerator inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando SignalGenerator: {e}")
            state["senales"] = {"signals": [], "timestamp": datetime.now(), "error": str(e)}
            return state
        
        # 3. Extraer datos de mercado del state (adaptado para multiasset: dict de DataFrames)
        logger.info("📊 Extrayendo datos de mercado del state...")
        market_data = _extract_market_data(state)
        
        if not market_data:
            logger.warning("⚠️ No hay datos de mercado disponibles")
            state["senales"] = {"signals": [], "timestamp": datetime.now(), "warning": "No market data"}
            return state
        
        symbols = list(market_data.keys())
        logger.info(f"📈 Datos de mercado extraídos para símbolos: {symbols}")
        
        # 5. Contexto de régimen desde L3 (si está disponible)
        regime_context = _extract_regime_context(state)
        logger.info(f"🧠 Contexto de régimen: {regime_context}")
        
        # 6. Generar señales tácticas (para todos los símbolos)
        logger.info("🔍 Generando señales tácticas...")
        signals = signal_generator.generate_signals(
            market_data=market_data,
            regime_context=regime_context
        )
        
        logger.info(f"✅ Generadas {len(signals)} señales tácticas para {len(symbols)} símbolos")
        
        # 7. Procesar y formatear señales para el state
        formatted_signals = []
        for i, signal in enumerate(signals):
            signal_dict = {
                "id": f"L2_{i}_{int(signal.timestamp.timestamp())}",
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "price": signal.price,
                "timestamp": signal.timestamp.isoformat(),
                "source": signal.source.value,
                "metadata": signal.metadata,
                "expires_at": signal.expires_at.isoformat() if signal.expires_at else None,
                "effective_strength": signal.effective_strength
            }
            formatted_signals.append(signal_dict)
            
            logger.debug(f"📡 Señal {i+1}: {signal.direction.value} {signal.symbol} "
                        f"strength={signal.strength:.2f} confidence={signal.confidence:.2f} "
                        f"source={signal.source.value}")
        
        # 8. Actualizar state con señales L2
        state["senales"] = {
            "signals": formatted_signals,
            "timestamp": datetime.now().isoformat(),
            "count": len(formatted_signals),
            "symbols": symbols,
            "regime_context": regime_context,
            "generator_info": signal_generator.get_model_info()
        }
        
        logger.info(f"✅ L2 completado - {len(formatted_signals)} señales agregadas al state")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error crítico en procesamiento L2: {e}", exc_info=True)
        state["senales"] = {
            "signals": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        }
        return state


def _extract_market_data(state: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Extrae datos de mercado del state (adaptado para multiasset: dict de DataFrames por símbolo)
    """
    market_data = state.get("mercado")
    
    if isinstance(market_data, dict) and all(isinstance(v, pd.DataFrame) for v in market_data.values()):
        logger.debug(f"✅ Datos multiasset encontrados: {list(market_data.keys())}")
        return market_data
    
    # Fallback si no es dict de DataFrames
    logger.warning("⚠️ Formato de mercado no es dict de DataFrames, intentando convertir...")
    
    if isinstance(market_data, dict):
        try:
            # Intentar convertir cada valor a DataFrame si no lo es
            converted = {}
            for symbol, data in market_data.items():
                if not isinstance(data, pd.DataFrame):
                    if isinstance(data, list):  # e.g., lista de velas
                        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        df = pd.DataFrame(data, columns=columns[:len(data[0])])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        converted[symbol] = df
                    elif isinstance(data, dict):  # e.g., dict simple
                        df = pd.DataFrame([data])  # Wrap in list to create DataFrame
                        converted[symbol] = df
                    else:
                        continue
                else:
                    converted[symbol] = data
            if converted:
                logger.debug(f"✅ Convertido a dict de DataFrames: {list(converted.keys())}")
                return converted
        except Exception as e:
            logger.warning(f"⚠️ Error convirtiendo datos de mercado: {e}")
    
    # Opción alternativa: datos en formato de lista de velas (para compatibilidad)
    if "velas" in state:
        logger.debug("📊 Encontrados datos en state['velas']")
        try:
            velas = state["velas"]
            if isinstance(velas, dict):  # Si ya es dict por símbolo
                return {symbol: pd.DataFrame(v) for symbol, v in velas.items()}
            elif isinstance(velas, list) and len(velas) > 0:
                # Asumir single symbol, default a BTC/USDT
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(velas, columns=columns[:len(velas[0])])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return {"BTC/USDT": df}
        except Exception as e:
            logger.warning(f"⚠️ Error procesando velas: {e}")
    
    # Generar datos mock si nada funciona
    logger.warning("⚠️ No se encontraron datos de mercado, generando datos mock")
    return _generate_mock_market_data()


def _extract_regime_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae contexto de régimen desde el state (información de L3)
    """
    logger.debug("🧠 Extrayendo contexto de régimen...")
    
    regime_context = {}
    
    # Buscar información de estrategia/régimen
    if "estrategia" in state:
        regime_context["strategy"] = state["estrategia"]
        logger.debug(f"📋 Estrategia: {state['estrategia']}")
    
    if "regimen" in state:
        regime_context["regime"] = state["regimen"]
        logger.debug(f"📊 Régimen: {state['regimen']}")
    
    # Información de exposición
    if "exposicion" in state:
        regime_context["exposure"] = state["exposicion"]
        logger.debug(f"💰 Exposición: {state['exposicion']}")
    
    # Información de universo
    if "universo" in state:
        regime_context["universe"] = state["universo"]
        logger.debug(f"🌍 Universo: {state['universo']}")
    
    # Contexto por defecto si no hay información
    if not regime_context:
        regime_context = {
            "regime": "unknown",
            "strategy": "neutral",
            "risk_level": "moderate"
        }
        logger.debug("🔧 Usando contexto de régimen por defecto")
    
    return regime_context


def _generate_mock_market_data() -> Dict[str, pd.DataFrame]:
    """
    Genera datos de mercado mock para testing cuando no hay datos reales (multiasset)
    """
    logger.warning("🎭 Generando datos mock para testing...")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generar 100 velas de 1 minuto para cada símbolo
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
    
    mock_data = {}
    for symbol, base_price in [("BTC/USDT", 50000.0), ("ETH/USDT", 3000.0)]:
        np.random.seed(42)  # Para reproducibilidad
        price_changes = np.random.normal(0, 0.001, 100)  # 0.1% std deviation
        prices = [base_price]
        
        for change in price_changes[:-1]:
            prices.append(prices[-1] * (1 + change))
        
        # Construir OHLCV
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            open_price = prices[i-1] if i > 0 else close
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.0005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.0005)))
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        mock_data[symbol] = df
    
    logger.debug(f"✅ Generados datos mock para: {list(mock_data.keys())}")
    
    return mock_data