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
from .l2_utils import safe_float

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
        Build observations based on expected dimensions rather than hardcoded model names.
        Uses expected_dims to determine observation building strategy and proper reshaping.
        """
        try:
            if self.expected_dims == 13:
                # Legacy 13-dimensional observation (Gemini and similar legacy models)
                obs = ObservationBuilders.build_legacy_observation(market_data, symbol, indicators)
                if obs is None:
                    raise ValueError("Failed to build legacy 13-dimensional observation")
                logger.debug(f"[DEBUG] Model: {self.model_name}, dims: {self.expected_dims}, shape: {obs.shape}")
                return obs.reshape(1, 13)

            elif self.expected_dims == 85:
                # HRM native 85-dimensional observation (DeepSeek native)
                obs = ObservationBuilders.build_hrm_native_obs(market_data, symbol, indicators)
                if obs is None:
                    raise ValueError(f"Failed to build HRM native {self.expected_dims}-dimensional observation")
                logger.debug(f"[DEBUG] Model: {self.model_name}, dims: {self.expected_dims}, shape: {obs.shape}")
                return obs.reshape(1, 85)

            elif self.expected_dims == 971:
                # Risk-aware multiasset observation (DeepSeek/Claude/Kimi)
                obs = ObservationBuilders.build_multiasset_obs(market_data, symbol, indicators)
                if obs is None:
                    raise ValueError(f"Failed to build multiasset {self.expected_dims}-dimensional observation")
                logger.debug(f"[DEBUG] Model: {self.model_name}, dims: {self.expected_dims}, shape: {obs.shape}")
                return obs.reshape(1, 971)

            elif self.expected_dims == 257:
                # Multiasset observation (standard FinRL)
                obs = ObservationBuilders.build_generic_obs(market_data, symbol, indicators, 257)
                if obs is None:
                    raise ValueError(f"Failed to build 257-dimensional observation")
                logger.debug(f"[DEBUG] Model: {self.model_name}, dims: {self.expected_dims}, shape: {obs.shape}")
                return obs.reshape(1, 257)

            elif self.model_name == "grok":
                # Grok - special handling with 1D array or scalar
                obs = market_data.get("grok_features")
                if obs is None:
                    # Fallback: build features from market_data
                    obs = ObservationBuilders.build_grok_obs(market_data, symbol, indicators)
                    if obs is None:
                        raise ValueError("Missing 'grok_features' in market_data and failed to build fallback")

                # Grok expects 1D array or scalar - flatten if necessary
                if hasattr(obs, 'shape') and len(obs.shape) > 1:
                    obs = obs.flatten()

                logger.debug(f"[DEBUG] Grok model: shape {obs.shape if hasattr(obs, 'shape') else 'scalar'}")
                return obs

            else:
                # Generic observation for custom dimensions
                obs = ObservationBuilders.build_generic_obs(market_data, symbol, indicators, self.expected_dims)
                if obs is None:
                    raise ValueError(f"Failed to build {self.expected_dims}-dimensional observation")
                logger.debug(f"[DEBUG] Generic model: {self.model_name}, dims: {self.expected_dims}, shape: {obs.shape}")
                return obs.reshape(1, self.expected_dims)

        except Exception as e:
            logger.error(f"❌ Error preparando observación para {self.model_name}: {e}")
            return None

    async def generate_signal(self, market_data: dict, symbol: str, indicators: dict = None, l3_context: dict = None):
        """
        PRIORITY 2: Make DeepSeek truly aggressive
        - Accept l3_context parameter for aggressive behavior in setups
        - Apply BUY logit scaling and HOLD penalization

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

                # PRIORITY 2: Apply aggressive post-processing for DeepSeek in setup conditions
                if signal and self.model_name.lower() == "deepseek" and l3_context:
                    signal = self._apply_aggressive_postprocessing(signal, symbol, l3_context)

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

    def _apply_aggressive_postprocessing(self, signal, symbol: str, l3_context: dict):
        """
        PRIORITY 2: Apply aggressive post-processing to DeepSeek signals in setup conditions

        - Scale BUY logits (boost confidence for BUY signals)
        - Penalize HOLD in setup conditions
        - Reduce punishment for short drawdown
        """
        try:
            if not signal or not l3_context:
                return signal

            # Extract signal properties
            side = getattr(signal, 'side', 'hold')
            confidence = getattr(signal, 'confidence', 0.5)
            strength = getattr(signal, 'strength', 0.5)

            # Check if we're in setup condition
            setup_type = l3_context.get('setup_type')
            allow_setup_trades = l3_context.get('allow_setup_trades', False) or l3_context.get('setup_active', False)
            l3_signal = l3_context.get('signal', 'hold')
            l3_regime = l3_context.get('regime', 'unknown')

            is_setup_condition = (
                allow_setup_trades or
                setup_type in ['oversold', 'overbought'] or
                (l3_signal == 'buy' and l3_regime in ['TRENDING', 'BREAKOUT']) or
                l3_regime == 'RANGE'  # Be more aggressive in range for mean reversion
            )

            if not is_setup_condition:
                # No aggressive processing needed
                return signal

            logger.info(f"🚀 DEEPSEEK AGGRESSIVE MODE: Processing {symbol} {side} in setup condition")

            # AGGRESSIVE BUY SCALING: Boost confidence for BUY signals
            if side == "buy":
                original_confidence = confidence
                confidence = min(0.95, confidence * 1.3)  # Scale up by 30%
                strength = min(0.9, strength * 1.2)     # Scale up by 20%
                logger.info(f"🚀 DEEPSEEK BUY SCALING: {symbol} confidence {original_confidence:.2f} → {confidence:.2f}")

            # PENALIZE HOLD: Convert weak HOLD signals to BUY/SELL in setup conditions
            elif side == "hold" and confidence < 0.7:
                logger.warning(f"🚨 DEEPSEEK HOLD PENALTY: Converting weak HOLD to BUY in setup condition")

                # Convert HOLD to BUY in setup conditions (bias towards action)
                side = "buy"
                confidence = 0.55  # Moderate confidence for forced decision
                strength = 0.5

                # Update signal metadata to reflect the change
                if hasattr(signal, 'metadata'):
                    signal.metadata = signal.metadata or {}
                    signal.metadata.update({
                        'original_side': 'hold',
                        'converted_by_aggressive_mode': True,
                        'setup_condition': True,
                        'priority_2_activated': True
                    })

            # Update signal with aggressive adjustments
            signal.side = side
            signal.confidence = confidence
            signal.strength = strength

            # Add aggressive mode metadata
            if hasattr(signal, 'features'):
                signal.features = signal.features or {}
                signal.features.update({
                    'aggressive_mode_applied': True,
                    'setup_condition': is_setup_condition,
                    'l3_regime': l3_regime,
                    'l3_signal': l3_signal
                })

            # Update signal type to reflect aggressive processing
            signal.signal_type = 'finrl_aggressive_deepseek'

            logger.info(f"✅ DEEPSEEK AGGRESSIVE RESULT: {symbol} {side.upper()} (conf={confidence:.2f}, strength={strength:.2f})")
            return signal

        except Exception as e:
            logger.error(f"❌ Error in aggressive post-processing for {symbol}: {e}")
            return signal
