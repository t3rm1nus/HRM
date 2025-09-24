# l2_tactic/finrl_wrapper.py
"""
FinRL Processor Wrapper - handles different model types and observation formats
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from loguru import logger

# Import our modular components
from .finrl_processor import FinRLProcessor
from .observation_builders import ObservationBuilders
from .signal_generators import SignalGenerators
from .utils import safe_float

# Handle relative imports for when running as script
try:
    from .models import TacticalSignal
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2_tactic.models import TacticalSignal


class FinRLProcessorWrapper:
    """
    Wrapper inteligente para FinRLProcessor que:
    - Detecta el método correcto (predict / get_action)
    - Ajusta automáticamente la forma de las observaciones según modelo
    - Mantiene compatibilidad con BlenderEnsemble y L2 pipeline
    """

    def __init__(self, processor, model_name: str):
        self.processor = processor
        self.model_name = model_name.lower()
        self.expected_dims = processor.observation_space_info.get('expected_dims', 257) if processor.observation_space_info else 257
        logger.info(f"🔧 FinRLProcessorWrapper inicializado para modelo {model_name} con {self.expected_dims} dimensiones esperadas")

    def _prepare_obs(self, market_data: dict, symbol: str, indicators: dict = None):
        """
        Ajusta las observaciones según el modelo:
        - Gemini: shape (1, 13) - legacy single-asset
        - DeepSeek/Claude/Kimi: shape (1, 971) - risk-aware multiasset
        - Grok: shape variable según modelo
        - Otros: shape según expected_dims
        """
        try:
            if self.model_name == "gemini":
                # Gemini legacy single-asset - 13 features
                obs = ObservationBuilders.build_gemini_obs(market_data, symbol, indicators)
                if obs is None:
                    raise ValueError("No se pudo construir observación Gemini")
                logger.debug(f"[DEBUG] Modelo: {self.model_name}, shape obs: {obs.shape}")
                return obs.reshape(1, 13)

            elif self.model_name in ["deepseek", "claude", "kimi"]:
                # DeepSeek / Claude / Kimi - risk-aware multiasset
                obs = ObservationBuilders.build_multiasset_obs(market_data, symbol, indicators)
                if obs is None:
                    raise ValueError(f"No se pudo construir observación {self.model_name}")
                logger.debug(f"[DEBUG] Modelo: {self.model_name}, shape obs: {obs.shape}")
                return obs.reshape(1, 971)

            elif self.model_name == "grok":
                # Grok - manejo especial con array 1D o escalar
                obs = market_data.get("grok_features")
                if obs is None:
                    # Fallback: construir features desde market_data
                    obs = ObservationBuilders.build_grok_obs(market_data, symbol, indicators)
                    if obs is None:
                        raise ValueError("Faltan 'grok_features' en market_data y no se pudo construir")

                # Grok espera un array 1D o escalar - aplanar si es necesario
                if hasattr(obs, 'shape') and len(obs.shape) > 1:
                    obs = obs.flatten()

                logger.debug(f"[DEBUG] Modelo: {self.model_name}, shape obs: {obs.shape if hasattr(obs, 'shape') else 'scalar'}")
                return obs

            else:
                # Modelo genérico - usar dimensiones esperadas
                obs = ObservationBuilders.build_generic_obs(market_data, symbol, indicators, self.expected_dims)
                if obs is None:
                    raise ValueError(f"No se pudo construir observación genérica para {self.expected_dims} dims")
                logger.debug(f"[DEBUG] Modelo: {self.model_name}, shape obs: {obs.shape}")
                return obs.reshape(1, self.expected_dims)

        except Exception as e:
            logger.error(f"❌ Error preparando observación para {self.model_name}: {e}")
            return None

    async def generate_signal(self, market_data: dict, symbol: str, indicators: dict = None):
        """
        Genera señal usando generate_signal como método principal
        """
        try:
            # Preparar observación
            obs = self._prepare_obs(market_data, symbol, indicators)
            if obs is None:
                logger.error(f"❌ Failed to prepare observation for {symbol}")
                return None

            # Usar generate_signal como método principal
            if hasattr(self.processor, 'generate_signal'):
                logger.debug(f"🔍 Usando generate_signal para {symbol} ({self.model_name})")

                # Preparar datos para generate_signal
                # Convertir DataFrame a dict si es necesario
                if hasattr(market_data.get(symbol, {}), 'iloc'):
                    data_dict = market_data[symbol].iloc[-1].to_dict()
                else:
                    data_dict = market_data.get(symbol, {})

                # Combinar con indicadores
                combined_data = {**data_dict}
                if indicators:
                    for key, value in indicators.items():
                        if hasattr(value, 'iloc'):
                            combined_data[key] = safe_float(value.iloc[-1]) if not value.empty else 0.0
                        else:
                            combined_data[key] = safe_float(value) if value is not None else 0.0

                signal = self.processor.generate_signal(symbol, market_data=combined_data)

                # Debug logging para Grok
                if self.model_name == "grok":
                    print(f"[DEBUG] Grok signal generated: side={signal.side if signal else 'None'}, "
                          f"strength={signal.strength if signal else 'N/A'}, "
                          f"confidence={signal.confidence if signal else 'N/A'}")

                return signal
            else:
                available_methods = [m for m in dir(self.processor) if not m.startswith('_')]
                raise AttributeError(
                    f"FinRLProcessor {type(self.processor)} no tiene generate_signal. "
                    f"Métodos disponibles: {available_methods}"
                )

        except Exception as e:
            logger.error(f"❌ Error generando señal FinRL para {symbol}: {e}")
            return None

    def _action_to_signal(self, action, symbol: str):
        """
        Convierte acción del modelo a señal táctica con lógica simplificada
        """
        try:
            # Handle tensor inputs
            if hasattr(action, 'detach'):  # torch tensor
                if action.numel() > 1:
                    action_val = safe_float(action.detach().cpu().max().item())
                else:
                    action_val = safe_float(action.detach().cpu().item())
            else:
                action_val = safe_float(action)

            # Normalize action to 0-1 range if needed
            if action_val < 0:
                action_val = 0.0
            elif action_val > 1:
                action_val = 1.0

            # Simple thresholds for signal generation
            if action_val < 0.4:
                side = "sell"
                confidence = 0.6 + (0.4 - action_val) * 0.5  # Higher confidence for stronger signals
                strength = 0.5 + (0.4 - action_val) * 0.5
            elif action_val > 0.6:
                side = "buy"
                confidence = 0.6 + (action_val - 0.6) * 0.5  # Higher confidence for stronger signals
                strength = 0.5 + (action_val - 0.6) * 0.5
            else:
                # Neutral zone - hold
                side = "hold"
                confidence = 0.5
                strength = 0.3

            # Ensure reasonable bounds
            confidence = max(0.3, min(0.9, confidence))
            strength = max(0.2, min(0.9, strength))

            logger.debug(f"📊 FinRL Signal: {symbol} {side} (action={action_val:.3f}, conf={confidence:.3f}, strength={strength:.3f})")

            return TacticalSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                confidence=confidence,
                signal_type='finrl_standard',
                source='finrl',
                timestamp=pd.Timestamp.utcnow(),
                features={'model': self.model_name, 'action_value': action_val}
            )

        except Exception as e:
            logger.error(f"❌ Error convirtiendo acción a señal: {e}")
            return TacticalSignal(
                symbol=symbol,
                side="hold",
                strength=0.3,
                confidence=0.4,
                signal_type="finrl_fallback",
                source="finrl",
                timestamp=pd.Timestamp.utcnow(),
                features={'error': str(e)}
            )
