"""
Integración del modelo de IA con FinRL para L2_tactic
===================================================
ARREGLADO: Ahora usa FinRL real en lugar de dummy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from core.logging import logger
from .finrl_integration import FinRLProcessor
from .models import TacticalSignal

@dataclass
class ModelPrediction:
    """Resultado de predicción del modelo"""
    symbol: str
    prediction: float  # -1 a 1
    confidence: float  # 0 a 1
    features_used: int
    model_type: str
    timestamp: float

class AIModelWrapper:
    """
    Wrapper principal para el modelo de IA
    ARREGLADO: Ahora usa FinRLProcessor en lugar de dummy
    """
    
    def __init__(self, config):
        self.config = config
        self.model_path = config.ai_model.model_path
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AI")
        
        # Inicializar FinRL processor
        try:
            logger.info(f"🤖 Inicializando modelo IA desde: {self.model_path}")
            self.finrl_processor = FinRLProcessor(self.model_path)
            self.model_loaded = True
            logger.info("✅ Modelo IA cargado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            self.finrl_processor = None
            self.model_loaded = False
    
    async def predict_async(self, features: Dict[str, Any]) -> Optional[ModelPrediction]:
        """
        Predicción asíncrona usando FinRL
        """
        if not self.model_loaded:
            logger.warning("⚠️  Modelo no disponible, retornando None")
            return None
            
        try:
            # Ejecutar predicción en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                features
            )
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción async: {e}")
            return None
    
    def _predict_sync(self, features: Dict[str, Any]) -> Optional[ModelPrediction]:
        """
        Predicción síncrona usando FinRL
        """
        try:
            # Extraer symbol y market data
            symbol = features.get('symbol', 'BTC/USDT')
            market_data = features.get('market_data', {})

            if not market_data:
                logger.warning(f"⚠️  Sin datos de mercado para {symbol} | features keys: {list(features.keys())} | features: {features} | market_data: {market_data}")
                return None

            # Generar señal usando FinRL
            signal = self.finrl_processor.generate_signal(market_data, symbol)

            if signal is None:
                return None

            # Convertir a ModelPrediction
            prediction = ModelPrediction(
                symbol=symbol,
                prediction=signal.strength,  # -1 a 1
                confidence=signal.confidence,  # 0 a 1
                features_used=len(market_data),
                model_type="FinRL_PPO",
                timestamp=signal.timestamp
            )

            logger.debug(f"🎯 Predicción {symbol}: {prediction.prediction:.3f} (conf: {prediction.confidence:.3f})")
            return prediction

        except Exception as e:
            logger.error(f"❌ Error en predicción sync: {e}")
            return None
    
    def predict(self, features: Dict[str, Any]) -> Optional[ModelPrediction]:
        """
        Predicción síncrona directa
        """
        return self._predict_sync(features)
    
    async def batch_predict(self, features_batch: List[Dict[str, Any]]) -> List[Optional[ModelPrediction]]:
        """
        Predicción en lotes
        """
        if not self.model_loaded:
            return [None] * len(features_batch)
            
        tasks = [self.predict_async(features) for features in features_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Manejar excepciones
        predictions = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ Error en batch: {result}")
                predictions.append(None)
            else:
                predictions.append(result)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Información del modelo
        """
        if not self.model_loaded:
            return {"status": "not_loaded", "model_type": "none"}
            
        return {
            "status": "loaded",
            "model_type": "FinRL_PPO", 
            "model_path": self.model_path,
            "observation_space": str(self.finrl_processor.observation_space) if hasattr(self.finrl_processor, 'observation_space') else "unknown",
            "action_space": str(self.finrl_processor.action_space) if hasattr(self.finrl_processor, 'action_space') else "unknown"
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
