# l2_tactic/generators/finrl.py
"""
FinRL signal generator - FIXED for MultiInputActorCriticPolicy
"""
import pickle
import joblib
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import os
import zipfile
import tempfile

logger = logging.getLogger(__name__)

class FinRLProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.observation_space_info = None
        
        if not self.load_real_model(model_path):
            raise RuntimeError(f"FAILED TO LOAD FINRL MODEL: {model_path}")
        
        # Inspect observation space after loading
        self.inspect_observation_space()
        
        logger.info(f"FinRL PPO model loaded successfully from {model_path}")

    def inspect_observation_space(self):
        """Inspect the model's observation space to understand expected format"""
        try:
            if hasattr(self.model, 'observation_space'):
                obs_space = self.model.observation_space
                logger.info(f"Observation space type: {type(obs_space)}")
                logger.info(f"Observation space: {obs_space}")
                
                # Store info for feature preparation
                self.observation_space_info = {
                    'type': type(obs_space).__name__,
                    'space': obs_space
                }
                
                # If it's a Dict space, log the sub-spaces
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
        """Load REAL FinRL model"""
        try:
            if not self.check_model_file(model_path):
                return False
            
            logger.info(f"Loading FinRL PPO model from {model_path}...")
            
            if model_path.endswith('.zip'):
                return self.load_stable_baselines3_model(model_path)
            elif model_path.endswith('.pkl'):
                return self.load_pickle_model(model_path)
            else:
                logger.error(f"Unsupported model format: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load FinRL model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def load_stable_baselines3_model(self, zip_path: str) -> bool:
        """Load stable_baselines3 PPO model from ZIP"""
        try:
            from stable_baselines3 import PPO
            
            logger.info(f"Loading stable_baselines3 PPO model from: {zip_path}")
            
            self.model = PPO.load(zip_path)
            self.is_loaded = True
            
            logger.info("PPO model loaded successfully via stable_baselines3!")
            logger.info(f"Model policy: {type(self.model.policy)}")
            
            return True
                    
        except ImportError as e:
            logger.error(f"stable_baselines3 not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading stable_baselines3 model: {e}")
            return False

    def load_pickle_model(self, pkl_path: str) -> bool:
        """Load pickle model with multiple methods"""
        methods = [
            ('pickle', lambda: pickle.load(open(pkl_path, 'rb'))),
            ('joblib', lambda: joblib.load(pkl_path)),
        ]
        
        for method_name, load_func in methods:
            try:
                logger.info(f"Trying {method_name} for {pkl_path}")
                self.model = load_func()
                self.is_loaded = True
                logger.info(f"Model loaded with {method_name}")
                return True
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                continue
        
        return False

    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict]:
        """Generate signals using REAL FinRL model"""
        if not self.is_loaded or not self.model:
            logger.error("FinRL model not loaded - cannot generate signals")
            raise RuntimeError("FinRL model not available")
        
        signals = []
        
        try:
            for symbol, data in market_data.items():
                signal_strength = self.generate_signal(data)
                
                if signal_strength is not None and abs(signal_strength) > 0.05:  # Lower threshold
                    side = 'buy' if signal_strength > 0 else 'sell'
                    signals.append({
                        'symbol': symbol,
                        'side': side,
                        'confidence': abs(signal_strength),
                        'strength': signal_strength,
                        'source': 'finrl_ppo',
                        'price': data.get('close', 0),
                        'stop_loss': self._calculate_stop_loss(data.get('close', 0), signal_strength > 0),
                        'metadata': {
                            'model_type': 'PPO',
                            'is_real_model': True,
                            'model_class': str(type(self.model))
                        }
                    })
                    
                    logger.info(f"FinRL signal: {symbol} {side} conf={abs(signal_strength):.3f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating FinRL signals: {e}")
            raise

    def generate_signal(self, data: Dict[str, Any]) -> Optional[float]:
        """Generate signal using REAL FinRL PPO model with proper observation format"""
        if not self.is_loaded:
            raise RuntimeError("FinRL model not loaded")
        
        try:
            # Prepare observation in the format expected by the model
            observation = self.prepare_observation(data)
            
            logger.debug(f"Observation type: {type(observation)}")
            logger.debug(f"Observation keys: {list(observation.keys()) if isinstance(observation, dict) else 'Not dict'}")
            
            # Get prediction from PPO model
            if hasattr(self.model, 'predict'):
                action, _ = self.model.predict(observation, deterministic=True)
                
                # Convert action to signal strength
                if isinstance(action, np.ndarray):
                    if len(action) > 0:
                        signal = float(action[0])
                    else:
                        signal = 0.0
                else:
                    signal = float(action)
                
                # Normalize to [-1, 1] range for trading signal
                signal = np.tanh(signal)
                
                logger.debug(f"PPO raw action: {action}, normalized signal: {signal:.3f}")
                
                return signal
            else:
                logger.error("Model doesn't have predict method")
                return None
                
        except Exception as e:
            logger.error(f"Error in PPO signal generation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def prepare_observation(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepare observation for MultiInputActorCriticPolicy
        This model expects a dictionary of observations, not a single array
        """
        try:
            # Common FinRL observation structure for multi-asset models
            close = float(data.get('close', 0.0))
            open_price = float(data.get('open', close))
            high = float(data.get('high', close))
            low = float(data.get('low', close))
            volume = float(data.get('volume', 0.0))
            
            # Normalize prices
            if close > 0:
                # Price features (normalized returns)
                price_features = np.array([
                    (open_price - close) / close,    # Open return
                    (high - close) / close,          # High return  
                    (low - close) / close,           # Low return
                    volume / 1e6,                    # Volume (scaled)
                ], dtype=np.float32)
                
                # Technical indicators (normalized)
                technical_features = np.array([
                    (data.get('rsi', 50.0) - 50.0) / 50.0,           # RSI normalized to [-1,1]
                    np.tanh(data.get('macd', 0.0) / 100.0),          # MACD normalized
                    (data.get('bb_upper', close*1.02) - close) / close,   # BB upper
                    (data.get('bb_lower', close*0.98) - close) / close,   # BB lower
                    (data.get('ema_12', close) - close) / close,          # EMA 12
                    (data.get('ema_26', close) - close) / close,          # EMA 26
                ], dtype=np.float32)
                
            else:
                price_features = np.zeros(4, dtype=np.float32)
                technical_features = np.zeros(6, dtype=np.float32)
            
            # Try different possible observation formats
            possible_formats = [
                # Format 1: Separate price and technical features
                {
                    'price': price_features.reshape(1, -1),
                    'technical': technical_features.reshape(1, -1)
                },
                # Format 2: Combined features
                {
                    'features': np.concatenate([price_features, technical_features]).reshape(1, -1)
                },
                # Format 3: Individual feature groups (common in FinRL)
                {
                    'ohlcv': price_features.reshape(1, -1),
                    'indicators': technical_features.reshape(1, -1)
                },
                # Format 4: Single observation array (fallback)
                {
                    'observation': np.concatenate([price_features, technical_features]).reshape(1, -1)
                }
            ]
            
            # Try each format until one works with the model
            for i, obs_format in enumerate(possible_formats):
                try:
                    # Test if this format is compatible
                    _ = self.model.predict(obs_format, deterministic=True)
                    logger.debug(f"Using observation format {i+1}: {list(obs_format.keys())}")
                    return obs_format
                except Exception as e:
                    logger.debug(f"Format {i+1} failed: {str(e)[:100]}")
                    continue
            
            # If all formats fail, raise the last error
            raise RuntimeError("No compatible observation format found")
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            # Return minimal fallback observation
            return {
                'observation': np.zeros((1, 10), dtype=np.float32)
            }

    def _calculate_stop_loss(self, price: float, is_long: bool, stop_pct: float = 0.02) -> float:
        """Calculate stop loss price"""
        if price <= 0:
            return 0.0
            
        if is_long:
            return price * (1 - stop_pct)
        else:
            return price * (1 + stop_pct)

# Debug script
if __name__ == "__main__":
    try:
        processor = FinRLProcessor('models/L2/ai_model_data_multiasset.zip')
        print("SUCCESS: Model loaded")
        
        # Test prediction with real market data structure
        test_data = {
            'open': 108000.0,
            'high': 109000.0,
            'low': 107500.0,
            'close': 108790.92,
            'volume': 1500000.0,
            'rsi': 45.0,
            'macd': -50.0,
            'bb_upper': 110000.0,
            'bb_lower': 107000.0,
            'ema_12': 108500.0,
            'ema_26': 108200.0,
        }
        
        signal = processor.generate_signal(test_data)
        print(f"Test signal: {signal}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()