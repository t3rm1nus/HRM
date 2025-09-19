# l2_tactic/finrl_processor.py
"""
Main FinRL Processor class - refactored from monolithic finrl_integration.py
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

# Import our modular components
from .model_loaders import ModelLoaders
from .signal_generators import SignalGenerators
from .observation_builders import ObservationBuilders

# Handle relative imports for when running as script
try:
    from .models import TacticalSignal
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2_tactic.models import TacticalSignal


def safe_float(x):
    """
    Convierte a float el último valor de un array, lista o Serie.
    Evita el error "only length-1 arrays can be converted to Python scalars".
    """
    if isinstance(x, (list, np.ndarray, pd.Series)):
        if len(x) == 0:
            return np.nan
        return float(x[-1])  # último valor
    try:
        return float(x)
    except Exception:
        return np.nan


class FinRLProcessor:
    """
    Main FinRL Processor class - handles model loading and signal generation
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.observation_space_info = None
        self.last_action_value = None

        # Load the model
        self._load_model()

        # Inspect observation space
        self._inspect_observation_space()

        logger.info(f"✅ FinRL model loaded successfully from {model_path}")

    def _load_model(self):
        """Load the model using the unified loader"""
        self.model = ModelLoaders.load_model_by_type(self.model_path)
        if self.model is not None:
            self.is_loaded = True
        else:
            raise RuntimeError(f"Failed to load FinRL model: {self.model_path}")

    def _inspect_observation_space(self):
        """Inspect the model's observation space to understand expected format"""
        try:
            if hasattr(self.model, 'observation_space'):
                obs_space = self.model.observation_space
                logger.info(f"Observation space type: {type(obs_space)}")
                logger.info(f"Observation space: {obs_space}")

                # Extract dimensions for different model types
                expected_dims = None
                if hasattr(obs_space, 'shape'):
                    expected_dims = obs_space.shape[0] if len(obs_space.shape) > 0 else None
                    logger.info(f"Expected observation dimensions: {expected_dims}")
                elif hasattr(obs_space, 'spaces'):
                    # Handle Dict/MultiInput spaces
                    total_dims = 0
                    for key, subspace in obs_space.spaces.items():
                        if hasattr(subspace, 'shape') and len(subspace.shape) > 0:
                            dims = subspace.shape[0]
                            total_dims += dims
                            logger.info(f"  {key}: {dims} dimensions")
                    expected_dims = total_dims
                    logger.info(f"Total expected dimensions: {expected_dims}")

                self.observation_space_info = {
                    'type': type(obs_space).__name__,
                    'space': obs_space,
                    'expected_dims': expected_dims
                }

                # Set model-specific configuration based on detected dimensions
                self._configure_model_specifics(expected_dims)

        except Exception as e:
            logger.warning(f"Could not inspect observation space: {e}")
            self.observation_space_info = {'type': 'unknown', 'expected_dims': None}

    def _configure_model_specifics(self, expected_dims: int):
        """Configure model-specific settings based on observation dimensions"""
        if expected_dims == 257:
            logger.info("✅ Model configured for 257-dimensional observations (FinRL multiasset)")
        elif expected_dims == 13:
            logger.info("ℹ️ Model configured for 13-dimensional observations (legacy)")
        elif expected_dims == 971:
            logger.info("🎯 Model configured for 971-dimensional observations (Claude risk-aware)")
        elif expected_dims <= 13 or (expected_dims > 13 and expected_dims < 257):
            logger.info(f"📊 Model configured for {expected_dims}-dimensional observations (Kimi custom)")
        else:
            logger.warning(f"⚠️ Unexpected observation dimensions: {expected_dims}")

    def generate_signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None,
                       features: Optional[Dict[str, Any]] = None, indicators: Optional[Dict[str, Any]] = None) -> Optional[TacticalSignal]:
        """
        Generate tactical signal using FinRL model
        """
        try:
            # 1️⃣ Preparar observación
            obs = SignalGenerators.prepare_observation(market_data or features or indicators)

            # 2️⃣ Llamada al modelo PPO (Stable Baselines3)
            action, _states = self.model.predict(obs, deterministic=True)

            # DEBUG: Log raw action details
            logger.debug(f"Raw action for {symbol}: {action}, type={type(action)}, shape={getattr(action, 'shape', 'no shape')}")

            # action puede ser un array, usar safe_float para conversión segura
            action_value = safe_float(action)
            logger.debug(f"Action converted using safe_float: {action_value} (original: {action}, type: {type(action)})")

            logger.debug(f"Final action value: {action_value}")
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

    async def get_action(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> TacticalSignal:
        """
        Legacy method for backward compatibility - generates action from state
        """
        try:
            if not self.is_loaded:
                logger.error("❌ Model not loaded")
                return None

            # Check expected dimensions and build appropriate observation
            expected_dims = self.observation_space_info.get('expected_dims', 257) if self.observation_space_info else 257

            if expected_dims == 13:
                observation = ObservationBuilders.build_legacy_observation(state, symbol, indicators)
            elif expected_dims == 971:
                observation = ObservationBuilders.build_multiasset_obs(state, symbol, indicators)
            elif expected_dims <= 13 or (expected_dims > 13 and expected_dims < 257):
                observation = ObservationBuilders.build_generic_obs(state, symbol, indicators, expected_dims)
            else:
                observation = ObservationBuilders.build_generic_obs(state, symbol, indicators, expected_dims)

            if observation is None:
                logger.error("❌ Failed to build observation")
                return None

            logger.debug(f"Observation shape: {observation.shape} (expected dims: {expected_dims})")

            # Convert to tensor
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0)

            # Get action from model
            with torch.no_grad():
                try:
                    raw_output = self.model.policy.forward(observation_tensor)

                    if isinstance(raw_output, tuple):
                        logits, value = raw_output[0], raw_output[1]
                    else:
                        logits, value = raw_output, None

                    logits = logits.to(dtype=torch.float32)

                    # --- Caso 1: escalar (acción continua tipo PPO) ---
                    if logits.ndim == 0 or (logits.ndim == 1 and logits.shape[0] == 1):
                        logger.debug(f"Raw logits for {symbol}: {logits}, shape={logits.shape}, ndim={logits.ndim}")
                        val = float(logits.item())
                        logger.debug(f"Converted logits to scalar: {val}")
                        if val > 0.05:
                            action_type = "buy"
                        elif val < -0.05:
                            action_type = "sell"
                        else:
                            action_type = "hold"
                        confidence = min(1.0, abs(val))
                        max_prob = confidence
                        action = None

                    # --- Caso 2: 2 clases (binario buy/sell) ---
                    elif logits.shape[-1] == 2:
                        logits = logits - logits.max(dim=-1, keepdim=True)[0]
                        probs = torch.softmax(logits, dim=-1)
                        action = torch.argmax(probs, dim=-1).item()
                        action_type = "buy" if action == 1 else "sell"
                        max_prob = float(probs.max().item())
                        confidence = max_prob

                    # --- Caso 3: 3 clases (hold/buy/sell) ---
                    elif logits.shape[-1] == 3:
                        logits = logits - logits.max(dim=-1, keepdim=True)[0]
                        probs = torch.softmax(logits, dim=-1)
                        action = torch.argmax(probs, dim=-1).item()
                        action_map = {0: "hold", 1: "buy", 2: "sell"}
                        action_type = action_map.get(action, "hold")
                        max_prob = float(probs.max().item())
                        confidence = max_prob

                        # Log de probabilidades
                        probs_np = probs.squeeze().detach().cpu().numpy()
                        logger.debug(f"Action probabilities - Hold: {probs_np[0]:.3f}, Buy: {probs_np[1]:.3f}, Sell: {probs_np[2]:.3f}")

                    else:
                        logger.error(f"❌ Unexpected logits shape: {logits.shape}")
                        return None

                    # Ajustar confianza si hay value head
                    if value is not None:
                        try:
                            val = value.squeeze().item()
                            normalized_value = (val + 1) / 2
                            confidence = (confidence + normalized_value) / 2
                        except Exception:
                            pass

                except Exception as e:
                    logger.error(f"❌ Error in forward pass: {e}")
                    logger.info(f"Observation shape: {observation_tensor.shape}")
                    logger.info(f"Features available: {len(observation)}")
                    return None

            # Construir señal táctica
            signal = TacticalSignal(
                symbol=symbol,
                strength=max_prob,
                confidence=confidence,
                side=action_type,
                signal_type='finrl',
                source="finrl",
                timestamp=pd.Timestamp.utcnow(),
                features={'observation_shape': len(observation), 'max_prob': max_prob}
            )

            self.last_action_value = action
            return signal

        except Exception as e:
            logger.error(f"❌ Error getting FinRL action: {e}")
            return None
