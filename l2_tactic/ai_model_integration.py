"""
Integración del modelo de IA para L2_tactic
===========================================

Wrapper genérico para modelos de IA empaquetados en .zip
Soporta sklearn, pytorch, tensorflow, stable_baselines3 y modelos custom.
"""

import os
import zipfile
import pickle
import joblib
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from .config import AIModelConfig
from .models import TacticalSignal, SignalDirection, SignalSource

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Error al cargar el modelo"""
    pass

class PredictionError(Exception):
    """Error durante predicción"""
    pass

class AIModelWrapper:
    """
    Wrapper para integrar el modelo de IA dentro de L2_tactic.

    - Recibe la configuración desde AIModelConfig
    - Carga el modelo (dummy o real)
    - Expone predict() para generar señales tácticas
    """

    def __init__(self, config: AIModelConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.prediction_cache: Dict[str, List[TacticalSignal]] = {}
        self.preprocessor = None  # Stub: puede ser un scaler o transformador real
        self._load_model()

    def _load_model(self):
        """
        Inicializa el modelo en memoria.
        Aquí puedes conectar con un modelo real (ML/DL), un servicio remoto, etc.
        """
        try:
            # Placeholder: cargar modelo dummy
            self.model = f"Modelo {getattr(self.config, 'model_name', 'desconocido')} cargado"
            logger.info(f"✅ AIModelWrapper cargado: {self.model}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo IA: {e}")
            self.model = None

    def predict(self, features: pd.DataFrame, symbol: str) -> List[TacticalSignal]:
        """
        Genera señales usando el modelo de IA.
        Args:
            features: DataFrame con datos del mercado (OHLCV)
            symbol: Símbolo del activo (e.g., BTC/USDT)
        Returns:
            Lista de TacticalSignal
        """
        if self.model is None:
            logger.warning("⚠️ Modelo IA no cargado, no se generan señales")
            return []

        try:
            if features.empty or "close" not in features.columns:
                logger.warning(f"⚠️ Datos inválidos o vacíos para {symbol}")
                return []

            # Mock prediction: LONG si el precio actual > media móvil de 20 períodos
            latest = features.iloc[-1]
            price = float(latest["close"])
            moving_average = features["close"].rolling(window=20).mean().iloc[-1] if len(features) >= 20 else price
            direction = SignalDirection.LONG if price > moving_average else SignalDirection.SHORT

            signal = TacticalSignal(
                symbol=symbol,
                direction=direction,
                strength=0.7,
                confidence=0.65,
                price=price,
                timestamp=pd.Timestamp.now(tz="UTC"),
                source=SignalSource.AI_MODEL,
                metadata={
                    "model": getattr(self.config, "model_name", "dummy"),
                    "condition": f"price_{'above' if price > moving_average else 'below'}_ma20"
                },
                expires_at=pd.Timestamp.now(tz="UTC") + pd.Timedelta(
                    minutes=getattr(self.config, "signal_horizon_minutes", 5)
                )
            )

            # Cachear señal
            self.prediction_cache[symbol] = [signal]
            logger.info(f"✅ Generada señal AI para {symbol}: {direction.value}")
            return [signal]

        except Exception as e:
            logger.error(f"❌ Error generando predicciones IA para {symbol}: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo cargado
        """
        return {
            "model_name": getattr(self.config, "model_name", "unknown"),
            "params": getattr(self.config, "model_params", {}),
            "loaded": self.model is not None,
            "cache_size": len(self.prediction_cache)
        }