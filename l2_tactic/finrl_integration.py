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
from datetime import datetime
from .models import TacticalSignal

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
        """Load stable_baselines3 PPO model from ZIP with correct architecture and device"""
        try:
            from stable_baselines3 import PPO
            import torch
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading stable_baselines3 PPO model from: {zip_path} with device={device} and policy_kwargs={policy_kwargs}")
            self.model = PPO.load(zip_path, device=device, policy_kwargs=policy_kwargs)
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
            obs_space = Box(low=-np.inf, high=np.inf, shape=(257,), dtype=np.float32)  # Match saved model
            action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Single output
            
            # Create policy matching saved architecture
            policy = ActorCriticPolicy(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=lambda _: 0.0,  # Dummy schedule since we're just using for inference
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Match saved 64-unit architecture
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

            # 2️⃣ Llamada al modelo PPO (Stable Baselines3)
            action, _states = self.model.predict(obs, deterministic=True)
            # action puede ser un array, tomar el valor si es necesario
            if isinstance(action, (list, tuple, np.ndarray)):
                action_value = float(action[0])
            else:
                action_value = float(action)
            logger.debug(f"Action value: {action_value}")
            # No hay value head accesible directamente, así que se pasa None
            signal = self._action_to_signal(action_value, symbol, value=None)
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

    def _action_to_signal(self, action_value: float, symbol: str, value: float = None):
        """
        Convert model action value to a tactical signal
        
        Args:
            action_value: Single value between 0-1 representing model's action
            symbol: The trading symbol
            value: Optional value prediction from the model's value head
        """
        try:
            # Map action value to probabilities:
            # 0.0-0.33: Strong sell
            # 0.33-0.66: Hold
            # 0.66-1.0: Strong buy
            
            # Convert action value to normalized probabilities
            action_val = float(action_value)  # Ensure Python float
            
            # Calculate probabilities based on action value ranges
            if action_val <= 0.33:
                sell_prob = 1.0 - (action_val * 3)  # Decreases as we approach 0.33
                hold_prob = action_val * 3          # Increases as we approach 0.33
                buy_prob = 0.0
            elif action_val <= 0.66:
                sell_prob = 0.0
                hold_prob = 1.0 - ((action_val - 0.33) * 3)  # Decreases from 1.0
                buy_prob = (action_val - 0.33) * 3           # Increases toward 1.0
            else:
                sell_prob = 0.0
                hold_prob = 0.0
                buy_prob = action_val
            
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
        processor = FinRLProcessor('models/L2/ai_model_data_multiasset.zip')
        print("SUCCESS: Model loaded")
        test_data = {
            'open': 108000.0, 'high': 109000.0, 'low': 107500.0, 'close': 108790.92,
            'volume': 1500000.0, 'rsi': 45.0, 'macd': -50.0, 'bollinger_upper': 110000.0,
            'bollinger_lower': 107000.0, 'ema_12': 108500.0, 'ema_26': 108200.0,
            'sma_20': 108000.0, 'sma_50': 107500.0, 'vol_mean_20': 1200000.0,
            'vol_std_20': 200000.0, 'vol_zscore': 1.5
        }
        signal = processor.generate_signal("BTCUSDT", market_data=test_data)
        print(f"Test signal: {signal}")

        # Print debug info
        obs = processor.prepare_observation(test_data)
        print(f"\nObservation shape: {obs.shape}")
        print(f"Non-zero features: {np.count_nonzero(obs)}")
        print(f"Value range: [{obs.min():.2f}, {obs.max():.2f}]")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()