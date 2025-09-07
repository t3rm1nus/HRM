# l2_tactic/finrl_integration.py
"""
FinRL signal generator - FIXED for MultiInputActorCriticPolicy and PyTorch
"""
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import os
import torch
from loguru import logger
from .models import TacticalSignal
from datetime import datetime
from .models import TacticalSignal
from datetime import datetime

class FinRLProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.observation_space_info = None
        self.last_action_value = None
        
        if not self.load_real_model(model_path):
            raise RuntimeError(f"FAILED TO LOAD FINRL MODEL: {model_path}")
        
        self.inspect_observation_space()
        logger.info(f"✅ FinRL model loaded successfully from {model_path}")

    def inspect_observation_space(self):
        """Inspect the model's observation space to understand expected format"""
        try:
            if hasattr(self.model, 'observation_space'):
                obs_space = self.model.observation_space
                logger.info(f"Observation space type: {type(obs_space)}")
                logger.info(f"Observation space: {obs_space}")
                self.observation_space_info = {
                    'type': type(obs_space).__name__,
                    'space': obs_space
                }
                if hasattr(obs_space, 'spaces'):
                    logger.info("Sub-spaces:")
                    for key, subspace in obs_space.spaces.items():
                        logger.info(f"  {key}: {subspace}")
        except Exception as e:
            logger.warning(f"Could not inspect observation space: {e}")

    def check_model_file(self, model_path: str) -> bool:
        """Check if model file exists and is valid"""
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return False
        try:
            file_size = os.path.getsize(model_path)
            if file_size < 1000:
                logger.error(f"Model file too small: {file_size} bytes")
                return False
            logger.info(f"Model file check passed: {file_size/1024:.1f}KB")
            return True
        except Exception as e:
            logger.error(f"Error checking model file: {e}")
            return False

    def load_real_model(self, model_path: str) -> bool:
        """Load FinRL model"""
        if not self.check_model_file(model_path):
            return False
        try:
            logger.info(f"Loading FinRL model from {model_path}...")
            if model_path.endswith('.zip'):
                return self.load_stable_baselines3_model(model_path)
            elif model_path.endswith('.pkl'):
                return self.load_pickle_model(model_path)
            elif model_path.endswith('.pth'):
                return self.load_torch_model(model_path)
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load FinRL model: {e}", exc_info=True)
            return False

    def load_stable_baselines3_model(self, zip_path: str) -> bool:
        """Load stable_baselines3 PPO model from ZIP"""
        try:
            from stable_baselines3 import PPO
            logger.info(f"Loading stable_baselines3 PPO model from: {zip_path}")
            self.model = PPO.load(zip_path, device='cpu')
            self.is_loaded = True
            logger.info(f"PPO model loaded successfully via stable_baselines3! Policy: {type(self.model.policy)}")
            return True
        except ImportError as e:
            logger.error(f"stable_baselines3 not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading stable_baselines3 model: {e}", exc_info=True)
            return False

    def load_pickle_model(self, pkl_path: str) -> bool:
        """Load pickled model"""
        try:
            self.model = pickle.load(open(pkl_path, 'rb'))
            self.is_loaded = True
            logger.info(f"Pickled model loaded successfully: {type(self.model)}")
            return True
        except Exception as e:
            logger.error(f"Error loading pickled model: {e}", exc_info=True)
            return False

    def load_torch_model(self, pth_path: str) -> bool:
        """Load PyTorch model"""
        try:
            # Create model instance with the right architecture
            from stable_baselines3.ppo.policies import ActorCriticPolicy
            
            # Define observation space (matching the saved model)
            from gymnasium.spaces import Box
            import numpy as np
            obs_space = Box(low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32)  # 63 features
            action_space = Box(low=0, high=2, shape=(2,), dtype=np.float32)  # 2 outputs
            
            # Create policy with matching architecture
            policy = ActorCriticPolicy(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=lambda _: 0.0,  # Dummy schedule since we're just using for inference
                net_arch=[dict(pi=[256, 128], vf=[256, 128])]  # Match saved architecture
            )
            
            # Load state dict
            state_dict = torch.load(pth_path, map_location='cpu')
            policy.load_state_dict(state_dict)
            policy.eval()  # Set to evaluation mode
            
            self.model = policy
            self.is_loaded = True
            logger.info(f"PyTorch model loaded successfully and reconstructed as ActorCriticPolicy")
            return True
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}", exc_info=True)
            return False

    def generate_signal(self, symbol: str, market_data: Optional[Dict[str, Any]] = None, 
                       features: Optional[Dict[str, Any]] = None, indicators: Optional[Dict[str, Any]] = None) -> Optional[TacticalSignal]:
        """
        Generate tactical signal using FinRL model
        """
        try:
            # 1️⃣ Preparar observación
            obs = self.prepare_observation(market_data or features or indicators)

            # 2️⃣ Llamada al modelo (policy)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                
                # Forward pass through the policy network
                # This gets both the features and action distribution
                features, dist = self.model.mlp_extractor(obs_tensor)
                
                # Get value prediction
                value = self.model.value_net(features).cpu().numpy()[0]
                
                # Get action logits and convert to probabilities
                action_logits = self.model.action_net(features)
                probs = torch.softmax(action_logits, dim=-1)
                action_probs = probs.cpu().numpy()[0]  # Remove batch dimension
                
                logger.debug(f"Action probs: {action_probs}, Value: {value}")
                
            # 3️⃣ Convert probabilities to signal and return
            signal = self._action_to_signal(action_probs, symbol, value)
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

    def prepare_observation(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare observation as a flat np.ndarray[float32] matching Box obs_space.
        Extends basic features to match the 63-dimensional input expected by the model.
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

            # Base features (19 dimensions)
            base_features = [
                'open','high','low','close','volume',
                'sma_20','sma_50','ema_12','ema_26','macd','macd_signal','rsi',
                'bollinger_middle','bollinger_std','bollinger_upper','bollinger_lower',
                'vol_mean_20','vol_std_20','vol_zscore'
            ]
            
            # Get base feature values
            base_values = [float(data.get(f, 0.0)) for f in base_features]
            
            # Add derived features to reach 63 dimensions
            derived_values = []
            
            # Price momentum features
            if len(base_values) >= 4:  # If we have OHLCV data
                price = base_values[3]  # Close price
                derived_values.extend([
                    price / base_values[0] - 1,  # Returns vs open
                    price / base_values[1] - 1,  # Returns vs high
                    price / base_values[2] - 1,  # Returns vs low
                ])
            else:
                derived_values.extend([0.0] * 3)
                
            # Normalized indicators
            if len(base_values) >= 12:  # If we have technical indicators
                rsi = base_values[11]
                derived_values.extend([
                    (rsi - 50) / 50,  # Normalized RSI
                    base_values[9] / abs(base_values[10]) if abs(base_values[10]) > 0 else 0,  # MACD ratio
                ])
            else:
                derived_values.extend([0.0] * 2)
                
            # Pad remaining dimensions with zeros
            remaining_dims = 63 - (len(base_values) + len(derived_values))
            derived_values.extend([0.0] * remaining_dims)
            
            # Combine base and derived features
            obs_values = base_values + derived_values
            obs = np.array(obs_values, dtype=np.float32)
            
            # Add batch dimension and validate shape
            obs = obs.reshape(1, -1)
            if obs.shape[1] != 63:
                raise ValueError(f"Invalid observation shape: {obs.shape}, expected (1, 63)")
                
            return obs
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}", exc_info=True)
            # Return zero vector as fallback
            return np.zeros((1, 63), dtype=np.float32)
        except Exception as e:
            logger.error(f"Error preparing flat observation: {e}", exc_info=True)
            return np.zeros((1, 19), dtype=np.float32)

    def _action_to_signal(self, action_probs, symbol: str, value: float = None):
        """
        Convert model outputs to a tactical signal
        
        Args:
            action_probs: Probabilities from the model's action head
            symbol: The trading symbol
            value: Optional value prediction from the model's value head
        """
        try:
            # Convert to numpy array if needed
            probs = np.array(action_probs).flatten() if isinstance(action_probs, (list, np.ndarray)) else np.array([action_probs])
            
            # Convert numpy values to native Python types and get action strength
            probs = [abs(float(p)) for p in probs]
            action_strength = max(probs)  # Get raw action strength
            
            # Scale action strength to [0,1]
            action_strength = min(1.0, (np.tanh(action_strength) + 1) / 2)
            
            # Handle different output formats
            # Calculate probabilities based on action space
            if len(probs) == 2:  # Binary action space (buy/sell)
                buy_prob, sell_prob = probs
                hold_prob = max(0.0, 1.0 - (buy_prob + sell_prob))  # Implicit hold probability
            elif len(probs) == 3:  # Trinary action space (buy/hold/sell)
                buy_prob, hold_prob, sell_prob = probs
            elif len(probs) == 1:  # Continuous action space [-1, 1]
                action_val = probs[0]
                # Convert to probabilities while maintaining strength
                if action_val > 0.2:
                    buy_prob = action_strength  # Use scaled strength
                    hold_prob = 0.2
                    sell_prob = 1.0 - buy_prob - hold_prob
                elif action_val < -0.2:
                    sell_prob = action_strength  # Use scaled strength
                    hold_prob = 0.2
                    buy_prob = 1.0 - sell_prob - hold_prob
                else:
                    hold_prob = 0.4
                    buy_prob = sell_prob = (1.0 - hold_prob) / 2
            else:
                logger.warning(f"Unexpected probability shape: {len(probs)}")
                buy_prob = sell_prob = action_strength / 3
                hold_prob = 1.0 - (buy_prob + sell_prob)
            
            # Get action value/strength directly from probabilities
            action_strength = max(buy_prob, sell_prob)  # Hold prob doesn't affect strength
            
            # Determine action based on highest probability while preserving PPO strength
            if buy_prob > max(sell_prob, hold_prob) and buy_prob > 0.4:
                side = "buy"
                strength = action_strength  # Use scaled action strength
            elif sell_prob > max(buy_prob, hold_prob) and sell_prob > 0.4:
                side = "sell" 
                strength = action_strength  # Use scaled action strength
            else:
                side = "hold"
                strength = min(0.4, action_strength)  # Limit hold strength

            # Use value prediction to scale confidence if available
            base_confidence = max(buy_prob, sell_prob, hold_prob)
            if value is not None:
                # Convert value to native Python float and scale to [0, 1]
                value = float(value)
                value_confidence = (np.tanh(value / 2) + 1) / 2
                confidence = (base_confidence + float(value_confidence)) / 2
            else:
                confidence = base_confidence

            # Create metadata with native Python types
            metadata = {
                "source": "finrl",
                "probabilities": {
                    "buy": buy_prob,
                    "hold": hold_prob,
                    "sell": sell_prob
                }
            }
            
            if value is not None:
                metadata["value"] = value

            return TacticalSignal(
                symbol=symbol,
                side=side,
                type="market",
                strength=strength,
                confidence=confidence,
                signal_type=side,
                timestamp=datetime.utcnow().timestamp(),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error converting action to signal: {e}")
            return TacticalSignal(
                symbol=symbol,
                side="hold",
                type="market",
                strength=0.1,
                confidence=0.1,
                signal_type="hold",
                timestamp=datetime.utcnow().timestamp(),
                metadata={"error": str(e)}
            )

    def _calculate_stop_loss(self, price: float, is_long: bool, stop_pct: float = 0.02) -> float:
        """Calculate stop loss price"""
        if price <= 0:
            return 0.0
        if is_long:
            return price * (1 - stop_pct)
        else:
            return price * (1 + stop_pct)

if __name__ == "__main__":
    try:
        processor = FinRLProcessor('models/L2/ai_model_data_multiasset/policy.pth')
        print("SUCCESS: Model loaded")
        test_data = {
            'open': 108000.0, 'high': 109000.0, 'low': 107500.0, 'close': 108790.92,
            'volume': 1500000.0, 'rsi': 45.0, 'macd': -50.0, 'bollinger_upper': 110000.0,
            'bollinger_lower': 107000.0, 'ema_12': 108500.0, 'ema_26': 108200.0,
        }
        signal = processor.generate_signal(test_data)
        print(f"Test signal: {signal}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()