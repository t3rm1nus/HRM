"""
L2TacticProcessor - Generador de señales tácticas ARREGLADO
==========================================================
ARREGLADO: Integración correcta con FinRL y manejo de errores mejorado
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from core.logging import logger
from .models import TacticalSignal
from .ai_model_integration import AIModelWrapper
from .finrl_integration import FinRLProcessor
from .technical.multi_timeframe import MultiTimeframeTechnical
from .risk_overlay import RiskOverlay

class L2TacticProcessor:
    """
    Generador de señales tácticas para L2
    ARREGLADO: Usa AIModelWrapper correctamente integrado con FinRL
    """
    
    def __init__(self, config):
        self.config = config
        
        # Inicializar componentes
        self.ai_model = AIModelWrapper(config)
        self.technical = MultiTimeframeTechnical(config)
        self.risk_overlay = RiskOverlay(config)
        
        logger.info("🎯 L2TacticProcessor inicializado correctamente")
    
    async def ai_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales usando el modelo de IA (FinRL)
        ARREGLADO: Manejo correcto de FinRL async/sync
        """
        signals = []
        
        try:
            # Procesar cada símbolo en el universo
            universe = self.config.signals.universe
            logger.debug(f"🤖 Generando señales IA para universo: {universe}")
            
            for symbol in universe:
                if symbol == "USDT":  # Skip stablecoin
                    continue
                    
                # Preparar features para el modelo
                features = self._prepare_features(market_data, symbol)
                
                if not features:
                    logger.warning(f"⚠️  Sin features para {symbol}")
                    continue
                
                # Obtener predicción del modelo
                prediction = await self.ai_model.predict_async(features)
                
                if prediction and abs(prediction.prediction) > self.config.ai_model.prediction_threshold:
                    # Convertir predicción a señal táctica
                    signal = TacticalSignal(
                        symbol=symbol,
                        signal_type="ai_model",
                        strength=prediction.prediction,  # -1 a 1
                        confidence=prediction.confidence,  # 0 a 1
                        side="buy" if prediction.prediction > 0 else "sell",
                        features=features,
                        timestamp=prediction.timestamp,
                        metadata={
                            "model_type": prediction.model_type,
                            "features_count": prediction.features_used,
                            "threshold": self.config.ai_model.prediction_threshold
                        }
                    )
                    
                    signals.append(signal)
                    logger.info(f"🎯 Señal IA generada: {symbol} {signal.side} strength={signal.strength:.3f}")
                else:
                    logger.debug(f"🤖 Sin señal para {symbol} (pred={prediction.prediction if prediction else None})")
            
            logger.info(f"🤖 Señales IA generadas: {len(signals)}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generando señales IA: {e}")
            return []
    
    async def technical_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales usando análisis técnico multi-timeframe
        """
        try:
            signals = await self.technical.generate_signals(market_data)
            logger.info(f"📊 Señales técnicas generadas: {len(signals)}")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales técnicas: {e}")
            return []
    
    async def risk_overlay(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de ajuste de riesgo
        """
        try:
            signals = await self.risk_overlay.generate_risk_signals(market_data, portfolio_data)
            logger.info(f"🛡️ Señales de riesgo generadas: {len(signals)}")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales de riesgo: {e}")
            return []
    
    def _prepare_features(self, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Prepara features para el modelo de IA
        ARREGLADO: Mejor estructura de features para FinRL
        """
        try:
            # Obtener datos del símbolo
            symbol_data = market_data.get(symbol, {})
            
            if not symbol_data:
                return None
            
            # Estructura de features compatible con FinRL
            features = {
                "symbol": symbol,
                "market_data": symbol_data,
                "ohlcv": symbol_data.get("ohlcv", {}),
                "technical_indicators": symbol_data.get("indicators", {}),
                "volume_profile": symbol_data.get("volume", {}),
                "orderbook": symbol_data.get("orderbook", {}),
                "metadata": {
                    "timestamp": symbol_data.get("timestamp"),
                    "source": "L2_tactic"
                }
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error preparando features para {symbol}: {e}")
            return None
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Estado del modelo de IA
        """
        return self.ai_model.get_model_info()
