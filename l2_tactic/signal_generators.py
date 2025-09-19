# l2_tactic/signal_generators.py
"""
Signal generation utilities for different FinRL models
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

# Handle relative imports for when running as script
try:
    from .models import TacticalSignal
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2_tactic.models import TacticalSignal


class SignalGenerators:
    """Collection of methods for generating signals from different models"""

    @staticmethod
    def prepare_observation(data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare observation as a flat np.ndarray[float32] matching Box obs_space (13 features).
        """
        try:
            # Si viene como dict anidado desde feature_engineering
            if isinstance(data, dict):
                if "ohlcv" in data and "indicators" in data:
                    flat = {**data["ohlcv"], **data["indicators"]}
                else:
                    flat = data
                data = pd.Series(flat)
            elif isinstance(data, pd.DataFrame):
                data = data.iloc[-1]

            # Define the 13 features expected by the model
            feature_names = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'rsi',
                'bollinger_upper', 'bollinger_lower',
                'ema_12', 'ema_26', 'macd'
            ]
            obs_values = []
            for f in feature_names:
                try:
                    obs_values.append(float(data.get(f, 0.0)))
                except (ValueError, TypeError):
                    obs_values.append(0.0)

            obs = np.array(obs_values, dtype=np.float32).reshape(1, -1)
            return obs
        except Exception as e:
            logger.error(f"Error preparing observation: {e}", exc_info=True)
            # Return zero vector matching expected size
            return np.zeros((1, 13), dtype=np.float32)

    @staticmethod
    def action_to_signal(action_value, symbol: str, model_name: str = "unknown", value=None):
        """
        Convert model action value to a tactical signal with improved numerical stability

        Args:
            action_value: Action value, can be tensor or float, representing model's action
            symbol: The trading symbol
            model_name: Name of the model for specific handling
            value: Optional value prediction from the model's value head
        """
        try:
            # Handle tensor inputs
            if torch.is_tensor(action_value):
                if action_value.numel() > 1:
                    # If action_value is a probability vector, get the highest prob action
                    action_val = action_value.detach().cpu().max().item()
                else:
                    action_val = action_value.detach().cpu().item()
            else:
                action_val = float(action_value)

            # Clamp to valid range [0,1]
            action_val = max(0.0, min(1.0, action_val))

            # Thresholds ajustados por modelo para evitar solo "hold"
            if model_name == "grok":
                # Thresholds más agresivos para Grok
                sell_threshold = 0.45  # Más restrictivo para evitar señales falsas
                buy_threshold = 0.55   # Más restrictivo para evitar señales falsas
                min_confidence = 0.6   # Confianza mínima más alta
            else:
                # Thresholds estándar para otros modelos
                sell_threshold = 0.4
                buy_threshold = 0.6
                min_confidence = 0.5

            if action_val <= sell_threshold:
                side = "sell"
                confidence = min(0.9, (sell_threshold - action_val) / sell_threshold + min_confidence)
                strength = confidence
            elif action_val >= buy_threshold:
                side = "buy"
                confidence = min(0.9, (action_val - buy_threshold) / (1.0 - buy_threshold) + min_confidence)
                strength = confidence
            else:
                # Zona media - generar señal basada en proximidad a thresholds
                if action_val < 0.5:
                    # Más cerca de sell
                    side = "sell"
                    confidence = min(0.7, min_confidence + (0.5 - action_val) * 0.4)
                    strength = confidence
                else:
                    # Más cerca de buy
                    side = "buy"
                    confidence = min(0.7, min_confidence + (action_val - 0.5) * 0.4)
                    strength = confidence

            # Logging específico para Grok
            if model_name == "grok":
                print(f"[DEBUG] Grok action_to_signal: action_val={action_val:.3f}, "
                      f"side={side}, confidence={confidence:.3f}")

            return TacticalSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                confidence=confidence,
                signal_type='finrl',
                source='finrl',
                timestamp=pd.Timestamp.utcnow(),
                features={'model': model_name, 'action_value': action_val}
            )

        except Exception as e:
            logger.error(f"❌ Error convirtiendo acción a señal: {e}")
            return TacticalSignal(
                symbol=symbol,
                side="hold",
                strength=0.1,
                confidence=0.1,
                signal_type="finrl_fallback",
                source="finrl",
                timestamp=pd.Timestamp.utcnow(),
                features={'error': str(e)}
            )

    @staticmethod
    def generate_finrl_signal(model, symbol: str, market_data: Optional[Dict[str, Any]] = None,
                             features: Optional[Dict[str, Any]] = None,
                             indicators: Optional[Dict[str, Any]] = None) -> Optional[TacticalSignal]:
        """
        Generate tactical signal using FinRL model
        """
        try:
            # 1️⃣ Preparar observación
            obs = SignalGenerators.prepare_observation(market_data or features or indicators)

            # 2️⃣ Llamada al modelo PPO (Stable Baselines3)
            action, _states = model.predict(obs, deterministic=True)
            # action puede ser un array, tomar el valor si es necesario
            if isinstance(action, np.ndarray):
                action_value = action.item() if action.size == 1 else float(action[0])
            elif isinstance(action, (list, tuple)):
                action_value = float(action[0])
            else:
                action_value = float(action)
            logger.debug(f"Action value: {action_value}")
            # No hay value head accesible directamente, así que se pasa None
            signal = SignalGenerators.action_to_signal(action_value, symbol, value=None)
            return signal

        except Exception as e:
            logger.error(f"❌ Error procesando señal para {symbol}: {e}")
            # Fallback to a neutral signal
            return TacticalSignal(
                symbol=symbol,
                strength=0.1,
                confidence=0.1,
                side="hold",
                type="market",
                signal_type="hold",
                source="ai_fallback",
                features={},
                metadata={'error': str(e)}
            )

    @staticmethod
    def calculate_stop_loss(price: float, is_long: bool, stop_pct: float = 0.02) -> float:
        """Calculate stop loss price"""
        if price <= 0:
            return 0.0
        if is_long:
            return price * (1 - stop_pct)
        else:
            return price * (1 + stop_pct)
