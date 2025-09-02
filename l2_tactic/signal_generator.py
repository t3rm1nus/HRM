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
from .signal_composer import SignalComposer
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
        self.signal_composer = SignalComposer(config)
        logger.info("🎯 L2TacticProcessor inicializado correctamente")
    
    async def ai_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        signals = []
        try:
            universe = self.config.signals.universe
            logger.debug(f"🤖 Generando señales IA para universo: {universe}")
            
            for symbol in universe:
                if symbol == "USDT":
                    continue
                    
                features = self._prepare_features(market_data, symbol)
                if not features:
                    logger.warning(f"⚠️ Sin features para {symbol}")
                    continue
                
                try:
                    prediction = await self.ai_model.predict_async(features)
                    if prediction is None:
                        logger.debug(f"🤖 Sin predicción para {symbol}")
                        continue
                    if not hasattr(prediction, 'prediction') or not hasattr(prediction, 'confidence'):
                        logger.warning(f"⚠️ Predicción inválida para {symbol}")
                        continue
                    
                    pred_value = prediction.prediction
                    if hasattr(pred_value, 'shape') and pred_value.shape == ():
                        pred_value = float(pred_value)
                    elif hasattr(pred_value, '__len__') and len(pred_value) > 0:
                        pred_value = float(pred_value[0])
                    else:
                        pred_value = float(pred_value) if pred_value is not None else 0.0
                    
                    if abs(pred_value) > self.config.ai_model.prediction_threshold:
                        signal = TacticalSignal(
                            symbol=symbol,
                            strength=pred_value,
                            confidence=float(prediction.confidence),
                            side="buy" if pred_value > 0 else "sell",
                            features=features,
                            timestamp=getattr(prediction, 'timestamp', pd.Timestamp.now()),
                            source="ai",
                            metadata={
                                "model_type": getattr(prediction, 'model_type', 'finrl'),
                                "features_count": getattr(prediction, 'features_used', 0),
                                "threshold": self.config.ai_model.prediction_threshold
                            }
                        )
                        signals.append(signal)
                        logger.info(f"🎯 Señal IA generada: {symbol} {signal.side} strength={pred_value:.3f}")
                    else:
                        logger.debug(f"🤖 Señal débil para {symbol} (pred={pred_value:.3f})")
                except Exception as e:
                    logger.error(f"❌ Error procesando predicción para {symbol}: {e}")
                    continue
            
            logger.info(f"🤖 Señales IA generadas: {len(signals)}")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales IA: {e}")
            return []
    
    async def technical_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        try:
            logger.debug(f"📊 Datos de mercado para señales técnicas: {market_data}")
            signals = await self.technical.generate_signals(market_data)
            logger.info(f"📊 Señales técnicas generadas: {len(signals)}")
            if not signals:
                logger.warning("⚠️ No se generaron señales técnicas, verificar datos de entrada o umbrales")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales técnicas: {e}", exc_info=True)
            return []
    
    async def risk_signals(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        try:
            logger.debug(f"🛡️ Datos para señales de riesgo - Mercado: {market_data}, Portfolio: {portfolio_data}")
            signals = await self.risk_overlay.generate_risk_signals(market_data, portfolio_data)
            logger.info(f"🛡️ Señales de riesgo generadas: {len(signals)}")
            if not signals:
                logger.warning("⚠️ No se generaron señales de riesgo, verificar datos de entrada o umbrales")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales de riesgo: {e}", exc_info=True)
            return []
    
    def _prepare_features(self, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        try:
            normalized_symbol = symbol.replace("/", "")
            symbol_data = market_data.get(symbol) or market_data.get(normalized_symbol)
            
            if not symbol_data and "/" not in symbol:
                if symbol == "BTCUSDT":
                    symbol_data = market_data.get("BTC/USDT")
                elif symbol == "ETHUSDT":
                    symbol_data = market_data.get("ETH/USDT")
            
            if not symbol_data:
                logger.warning(f"⚠️ No se encontraron datos para {symbol} (intenté: {symbol}, {normalized_symbol})")
                return None
            
            features = {
                "symbol": normalized_symbol,
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
            logger.debug(f"✅ Features preparadas para {normalized_symbol}")
            return features
        except Exception as e:
            logger.error(f"❌ Error preparando features para {symbol}: {e}")
            return None
    
    async def get_model_status(self) -> Dict[str, Any]:
        return self.ai_model.get_model_info()

    async def process(self, state: dict, market_data: dict, features_by_symbol: dict = None, bus=None) -> dict:
        """
        Ejecuta la capa Tactic:
        - Genera señales (AI + técnico + riesgo)
        - Combina y guarda en state['l2']
        - Convierte señales a órdenes para L1
        - Devuelve dict: {"signals", "orders_for_l1", "metadata"}
        """
        ai_signals = []
        technical_signals = []
        risk_signals = []
        all_signals = []

        try:
            logger.info("🎯 Iniciando procesamiento L2TacticProcessor")
            portfolio_data = state.get("portfolio", {})

            # --- Generación de señales ---
            try:
                ai_signals = await self.ai_signals(market_data)
            except Exception as e:
                logger.error(f"❌ Error en ai_signals: {e}")

            try:
                technical_signals = await self.technical_signals(market_data)
            except Exception as e:
                logger.error(f"❌ Error en technical_signals: {e}")

            try:
                risk_signals = await self.risk_signals(market_data, portfolio_data)
            except Exception as e:
                logger.error(f"❌ Error en risk_signals: {e}")

            # --- Combinar señales ---
            all_signals = ai_signals + technical_signals + risk_signals
            # --- Componer señales usando SignalComposer ---
            if all_signals:
                try:
                    all_signals = self.signal_composer.compose(all_signals)
                    logger.info(f"✅ Señales compuestas: {len(all_signals)}")
                    # Verificar que todas las señales tengan symbol y side
                    for signal in all_signals:
                        if not hasattr(signal, 'symbol') or not hasattr(signal, 'side'):
                            logger.error(f"❌ Señal inválida: {signal.__dict__}")
                            all_signals = []
                            break
                except Exception as e:
                    logger.error(f"❌ Error al componer señales: {e}", exc_info=True)
                    all_signals = []
            else:
                logger.warning("⚠️ No hay señales para componer")
        
            # --- Guardar en state['l2'] de forma segura ---
            now = pd.Timestamp.utcnow()
            state.setdefault('l2', {})
            state['l2']['signals'] = all_signals
            state['l2']['last_update'] = now

            # --- Convertir señales a órdenes para L1 ---
            orders_for_l1 = []
            for signal in all_signals:
                try:
                    order = {
                        "symbol": getattr(signal, "symbol", None),
                        "side": getattr(signal, "side", None),
                        "type": "market",
                        "strength": getattr(signal, "strength", None),
                        "confidence": getattr(signal, "confidence", None),
                        "signal_type": getattr(signal, "signal_type", "tactical"),
                        "timestamp": getattr(signal, "timestamp", now),
                        "metadata": getattr(signal, "metadata", {})
                    }
                    orders_for_l1.append(order)
                except Exception as e:
                    logger.error(f"❌ Error convirtiendo señal a orden: {e}")

            logger.info(f"✅ L2TacticProcessor completado: {len(all_signals)} señales, {len(orders_for_l1)} órdenes")

            return {
                "signals": all_signals,
                "orders_for_l1": orders_for_l1,
                "metadata": {
                    "ai_signals": len(ai_signals),
                    "technical_signals": len(technical_signals),
                    "risk_signals": len(risk_signals),
                    "total_signals": len(all_signals)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error crítico en L2TacticProcessor.process(): {e}")
            state['l2'] = {"signals": [], "orders_for_l1": [], "last_update": pd.Timestamp.utcnow()}
            return {
                "signals": [],
                "orders_for_l1": [],
                "metadata": {"error": str(e)}
            }