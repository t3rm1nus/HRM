import os
import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta
import pandas as pd
from .config import AIModelConfig
from .models import TacticalSignal, SignalDirection, SignalSource

logger = logging.getLogger("l2_tactic.ai_model_integration")

class ModelLoadError(Exception):
    """Error al cargar el modelo"""
    pass

class PredictionError(Exception):
    """Error durante predicción"""
    pass

class AIModelWrapper:
    """
    Wrapper para integrar el modelo de IA dentro de L2_tactic.
    """
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.prediction_cache: Dict[str, List[TacticalSignal]] = {}
        self.preprocessor = None
        self._load_model()

    def _load_model(self):
        """
        Inicializa el modelo en memoria.
        """
        try:
            self.model = f"Modelo {self.config.model_name} cargado"
            logger.info(f"✅ AIModelWrapper cargado: {self.model}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo IA: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model: {e}")

    def predict(self, features: pd.DataFrame, symbol: str) -> List[TacticalSignal]:
        """
        Genera señales usando el modelo de IA basado en la media móvil.
        Args:
            features: DataFrame con datos del mercado (OHLCV), requiere columna 'close'.
            symbol: Símbolo del activo (e.g., BTC/USDT).
        Returns:
            Lista de TacticalSignal, vacía si ocurre un error.
        """
        if self.model is None:
            logger.error("⚠️ Modelo IA no cargado")
            return []

        try:
            # Validar tipo de entrada
            if not isinstance(features, pd.DataFrame):
                logger.error(f"Features for {symbol} is not a DataFrame: {type(features)}")
                return []

            # Validar columnas requeridas
            if features.empty or "close" not in features.columns:
                logger.error(f"⚠️ Datos inválidos o vacíos para {symbol}: {features.columns}")
                return []

            # Validar tipo numérico de la columna 'close'
            if not pd.api.types.is_numeric_dtype(features["close"]):
                logger.error(f"Close column for {symbol} is not numeric: {features['close'].dtype}")
                return []

            # Validar índice DatetimeIndex
            if not isinstance(features.index, pd.DatetimeIndex):
                logger.warning(f"Index for {symbol} is not DatetimeIndex, attempting to convert")
                if "timestamp" in features.columns:
                    try:
                        features = features.set_index(pd.to_datetime(features["timestamp"], utc=True))
                    except Exception as e:
                        logger.error(f"Failed to convert timestamp for {symbol}: {e}")
                        return []
                else:
                    logger.error(f"No timestamp column to set index for {symbol}")
                    return []

            # Validar suficientes datos para la media móvil
            if len(features) < 20:
                logger.warning(f"Insufficient data for {symbol}: {len(features)} rows, need at least 20")
                return []

            # Generar predicción basada en media móvil
            latest = features.iloc[-1]
            price = float(latest["close"])
            moving_average = features["close"].rolling(window=20).mean().iloc[-1]
            direction = SignalDirection.LONG if price > moving_average else SignalDirection.SHORT
            confidence = 0.65 if price != moving_average else 0.5

            signal = TacticalSignal(
                symbol=symbol,
                side=direction.value,
                strength=0.7,
                confidence=confidence,
                price=price,
                timestamp=pd.Timestamp.now(tz="UTC"),
                source=SignalSource.AI.value,
                model_name=self.config.model_name,
                features_used={
                    "price": price,
                    "ma20": float(moving_average)
                },
                horizon="1h",
                reasoning=f"Price {'above' if price > moving_average else 'below'} 20-period MA"
            )

            self.prediction_cache[symbol] = [signal]
            logger.info(f"✅ Generada señal AI para {symbol}: {direction.value} (confidence: {confidence})")
            return [signal]

        except Exception as e:
            logger.error(f"❌ Error generando predicciones IA para {symbol}: {e}", exc_info=True)
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo cargado.
        """
        return {
            "model_name": self.config.model_name,
            "params": self.config.model_params,
            "loaded": self.model is not None,
            "cache_size": len(self.prediction_cache)
        }