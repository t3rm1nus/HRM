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
import torch.nn as nn
from loguru import logger
from datetime import datetime
from .models import TacticalSignal

# Import for Claude model custom feature extractor
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym
except ImportError:
    BaseFeaturesExtractor = None
    gym = None


if BaseFeaturesExtractor is not None:
    class RiskAwareExtractor(BaseFeaturesExtractor):
        """
        Custom feature extractor inspired by the paper's risk-aware architecture
        Used for Claude model training and inference
        """

        def __init__(self, observation_space, features_dim=512):
            if gym is None:
                raise ImportError("gymnasium not available")
            if hasattr(gym, 'spaces') and hasattr(gym.spaces, 'Box'):
                if not isinstance(observation_space, gym.spaces.Box):
                    raise ValueError("observation_space must be a gym.spaces.Box")
            super(RiskAwareExtractor, self).__init__(observation_space, features_dim)

            n_input_features = observation_space.shape[0]

            # Multi-layer feature extraction network
            self.feature_net = nn.Sequential(
                nn.Linear(n_input_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
                nn.ReLU()
            )

        def forward(self, observations):
            return self.feature_net(observations)
else:
    # Fallback class when BaseFeaturesExtractor is not available
    class RiskAwareExtractor:
        def __init__(self, observation_space, features_dim=512):
            raise ImportError("BaseFeaturesExtractor not available - gymnasium/stable_baselines3 not installed")

class FinRLProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.observation_space_info = None
        self.last_action_value = None
        
        load_result, last_error = self.load_real_model(model_path)
        if not load_result:
            import traceback
            print('--- EXCEPCIÓN ORIGINAL AL CARGAR EL MODELO ---')
            if last_error:
                if isinstance(last_error, BaseException):
                    traceback.print_exception(type(last_error), last_error, last_error.__traceback__)
                else:
                    print(str(last_error))
                raise RuntimeError(f"FAILED TO LOAD FINRL MODEL: {model_path}\nOriginal error: {last_error}")
            else:
                traceback.print_exc()
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

    def check_model_file(self, model_path: str) -> bool:
        """Check if model file exists and is valid"""
        if not os.path.exists(model_path):
            logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
            return False
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Menos de 1KB
            logger.error(f"❌ Archivo de modelo demasiado pequeño ({file_size} bytes): {model_path}")
            return False
        logger.info(f"✅ Archivo de modelo válido ({file_size/1024:.1f}KB): {model_path}")
        return True

    def load_real_model(self, model_path: str):
        """Unified loader: tries all supported formats, always returns (success, error)."""
        # Check file exists
        if not self.check_model_file(model_path):
            return False, f"Model file does not exist: {model_path}"
        try:
            logger.info(f"Loading FinRL model from {model_path}...")

            # Conditional loading based on model name
            if "gemini.zip" in model_path:
                # Load gemini.zip as stable_baselines3 model (current behavior)
                logger.info("🔍 Detected gemini.zip - loading as stable_baselines3 model")
                ok = self.load_stable_baselines3_model(model_path)
                return (ok, None) if ok else (False, f"Failed to load gemini.zip as stable_baselines3: {model_path}")

            elif "deepseek.zip" in model_path:
                # Load deepseek.zip with custom logic
                logger.info("🔍 Detected deepseek.zip - loading with custom logic")
                ok = self.load_deepseek_model(model_path)
                return (ok, None) if ok else (False, f"Failed to load deepseek.zip: {model_path}")

            elif "claude.zip" in model_path:
                # Load claude.zip with custom logic (RiskAwareExtractor)
                logger.info("🔍 Detected claude.zip - loading with custom logic")
                ok = self.load_claude_model(model_path)
                return (ok, None) if ok else (False, f"Failed to load claude.zip: {model_path}")

            elif "kimi.zip" in model_path:
                # Load kimi.zip with custom logic (default policy kwargs)
                logger.info("🔍 Detected kimi.zip - loading with custom logic")
                ok = self.load_kimi_model(model_path)
                return (ok, None) if ok else (False, f"Failed to load kimi.zip: {model_path}")

            else:
                # Default loading for other models
                if model_path.endswith('.zip'):
                    ok = self.load_stable_baselines3_model(model_path)
                    return (ok, None) if ok else (False, f"Failed to load as stable_baselines3 zip: {model_path}")
                elif model_path.endswith('.pkl'):
                    ok = self.load_pickle_model(model_path)
                    return (ok, None) if ok else (False, f"Failed to load as pickle: {model_path}")
                elif model_path.endswith('.pth'):
                    ok = self.load_torch_model(model_path)
                    return (ok, None) if ok else (False, f"Failed to load as torch: {model_path}")
                else:
                    return False, f"Unsupported model format: {model_path}"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"CRITICAL: Failed to load FinRL model: {e}\n{tb}")
            return False, e

    async def get_action(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> TacticalSignal:
        """
        Genera una acción basada en los indicadores técnicos.
        Uses appropriate observation dimensions based on model type.
        """
        try:
            if not self.is_loaded:
                logger.error("❌ Model not loaded")
                return None

            # Check expected dimensions and build appropriate observation
            expected_dims = self.observation_space_info.get('expected_dims', 257) if self.observation_space_info else 257

            if expected_dims == 13:
                # Legacy 13-dimensional observation for Gemini
                observation = self._build_legacy_observation(state, symbol, indicators)
            elif expected_dims == 971:
                # Large 971-dimensional observation for Claude risk-aware model
                observation = self._build_claude_observation(state, symbol, indicators)
            elif expected_dims <= 13 or (expected_dims > 13 and expected_dims < 257):
                # Custom dimensions for Kimi and other small-dimension models
                observation = self._build_kimi_observation(state, symbol, indicators, expected_dims)
            else:
                # Full 257-dimensional observation for DeepSeek and other multiasset models
                observation = self._build_full_observation(state, symbol, indicators)

            if observation is None:
                logger.error("❌ Failed to build observation")
                return None

            logger.debug(f"Observation shape: {observation.shape} (expected dims: {expected_dims})")

            # Convert to tensor with proper dimensions and ensure float32
            observation_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension

            # Get action from model with proper error handling
            with torch.no_grad():
                try:
                    # Forward pass through policy network with explicit type control
                    action_logits = self.model.policy.forward(observation_tensor)

                    # Handle different output formats
                    if isinstance(action_logits, tuple):
                        # If output is (logits, value)
                        logits = action_logits[0]
                        value = action_logits[1]
                    else:
                        logits = action_logits
                        value = None

                    # Ensure logits is a float tensor
                    logits = logits.to(dtype=torch.float32)

                    # Ensure logits have correct shape (batch_size, num_actions)
                    if len(logits.shape) == 1:
                        logits = logits.unsqueeze(0)

                    # Apply softmax with numerical stability and proper type
                    logits = logits.to(dtype=torch.float32)  # Ensure float type
                    logits_max = logits.max(dim=-1, keepdim=True)[0]  # For numerical stability
                    logits = logits - logits_max

                    # Compute softmax probabilities
                    exp_logits = torch.exp(logits)
                    action_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

                    # Get action with highest probability
                    action = torch.argmax(action_probs, dim=-1).item()

                    # Optional value head output
                    if value is not None:
                        value = value.squeeze().item()
                except Exception as e:
                    logger.error(f"❌ Error in forward pass: {e}")
                    logger.info(f"Observation shape: {observation_tensor.shape}")
                    logger.info(f"Features available: {len(observation)}")
                    return None

            # Map action to tactical signal with confidence calculation
            action_map = {0: "hold", 1: "buy", 2: "sell"}
            action_type = action_map.get(action, "hold")

            # Calculate confidence based on action probability and value if available
            max_prob = float(action_probs.max())
            confidence = max_prob
            if value is not None:
                # Blend probability with normalized value for confidence
                normalized_value = (value + 1) / 2  # Assuming value is in [-1, 1]
                confidence = (max_prob + normalized_value) / 2

            signal = TacticalSignal(
                symbol=symbol,
                strength=max_prob,
                confidence=confidence,
                side=action_type,  # Required parameter
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

    def _build_legacy_observation(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build legacy 13-dimensional observation for Gemini and other 13-feature models.
        """
        try:
            # Get market data from state
            market_data = state.get("market_data", state.get("mercado", {}))
            if not market_data:
                logger.error("❌ No market data available in state")
                return None

            # Get data for the specific symbol
            symbol_data = market_data.get(symbol)
            if symbol_data is None or (isinstance(symbol_data, pd.DataFrame) and symbol_data.empty):
                logger.error(f"❌ No market data available for {symbol}")
                return None

            # Extract the last row of data
            if isinstance(symbol_data, pd.DataFrame):
                last_row = symbol_data.iloc[-1]
            else:
                last_row = symbol_data

            # Define the 13 features expected by legacy models
            feature_names = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'rsi',
                'bollinger_upper', 'bollinger_lower',
                'ema_12', 'ema_26', 'macd'
            ]

            obs_values = []
            for f in feature_names:
                try:
                    # First try to get from indicators
                    if indicators and f in indicators:
                        indicator_series = indicators[f]
                        if hasattr(indicator_series, 'iloc'):
                            value = float(indicator_series.iloc[-1])
                        else:
                            value = float(indicator_series)
                    # Then try from market data
                    elif hasattr(last_row, 'get'):
                        value = float(last_row.get(f, 0.0))
                    elif isinstance(last_row, pd.Series) and f in last_row.index:
                        value = float(last_row[f])
                    else:
                        value = 0.0

                    obs_values.append(value if np.isfinite(value) else 0.0)
                except (ValueError, TypeError, KeyError):
                    obs_values.append(0.0)

            if len(obs_values) != 13:
                logger.error(f"❌ Feature dimension mismatch: got {len(obs_values)}, expected 13")
                return None

            logger.debug(f"✅ Built 13-dimensional legacy observation for {symbol}")
            return np.array(obs_values, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building legacy observation: {e}")
            return None

    def _build_claude_observation(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build large 971-dimensional observation for Claude risk-aware model.
        Expands available features to match the model's expected input size.
        """
        try:
            # Get market data from state
            market_data = state.get("market_data", state.get("mercado", {}))
            if not market_data:
                logger.error("❌ No market data available in state")
                return None

            # Build features DataFrame for current symbol
            features_df = self._build_features_dataframe(symbol, market_data, indicators)
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # Build features_by_symbol dict for cross-asset features
            features_by_symbol = {}
            for sym, df in market_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    features_by_symbol[sym] = df.copy()

            # Start with base features (expand to ~800 dimensions)
            base_features = self._select_claude_base_features(features_df)

            # Add cross/L3 features (11 dimensions)
            cross_features = self._build_cross_l3_features(market_data, features_by_symbol)

            # Add risk-aware features (expand to reach 971 total)
            risk_features = self._build_risk_aware_features(state, symbol, market_data, indicators)

            # Combine all features
            all_features = base_features + cross_features + risk_features

            # Ensure exactly 971 features
            if len(all_features) < 971:
                # Pad with zeros
                padding = [0.0] * (971 - len(all_features))
                all_features.extend(padding)
            elif len(all_features) > 971:
                # Truncate if too many
                all_features = all_features[:971]

            if len(all_features) != 971:
                logger.error(f"❌ Feature dimension mismatch: got {len(all_features)}, expected 971")
                return None

            logger.debug(f"✅ Built 971-dimensional Claude observation: {len(base_features)} base + {len(cross_features)} cross + {len(risk_features)} risk")
            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building Claude observation: {e}")
            return None

    def _build_kimi_observation(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any], expected_dims: int) -> Optional[np.ndarray]:
        """
        Build custom observation for Kimi model with variable dimensions.
        Creates a flexible observation that adapts to the model's expected input size.
        """
        try:
            # Get market data from state
            market_data = state.get("market_data", state.get("mercado", {}))
            if not market_data:
                logger.error("❌ No market data available in state")
                return None

            # Build features DataFrame for current symbol
            features_df = self._build_features_dataframe(symbol, market_data, indicators)
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # Get all available numeric features
            if not isinstance(features_df, pd.DataFrame) or features_df.empty:
                # Fallback: create basic features from market data
                basic_features = self._build_basic_features(state, symbol, indicators)
                # Pad or truncate to expected dimensions
                if len(basic_features) < expected_dims:
                    basic_features.extend([0.0] * (expected_dims - len(basic_features)))
                elif len(basic_features) > expected_dims:
                    basic_features = basic_features[:expected_dims]

                logger.debug(f"✅ Built {expected_dims}-dimensional Kimi observation (basic fallback)")
                return np.array(basic_features, dtype=np.float32)

            # Extract numeric features from the last row
            last_row = features_df.iloc[-1]
            numeric_cols = [
                c for c in features_df.columns
                if pd.api.types.is_numeric_dtype(features_df[c])
            ]

            # Build feature vector
            features = []
            for c in numeric_cols:
                try:
                    val = float(last_row[c]) if np.isfinite(last_row[c]) else 0.0
                    features.append(val)
                except Exception:
                    features.append(0.0)

            # If we have too few features, add basic market features
            if len(features) < expected_dims:
                basic_features = self._build_basic_features(state, symbol, indicators)
                features.extend(basic_features)

            # Pad or truncate to expected dimensions
            if len(features) < expected_dims:
                features.extend([0.0] * (expected_dims - len(features)))
            elif len(features) > expected_dims:
                features = features[:expected_dims]

            if len(features) != expected_dims:
                logger.error(f"❌ Feature dimension mismatch: got {len(features)}, expected {expected_dims}")
                return None

            logger.debug(f"✅ Built {expected_dims}-dimensional Kimi observation: {len(numeric_cols)} numeric + basic features")
            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building Kimi observation: {e}")
            return None

    def _build_basic_features(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> List[float]:
        """
        Build basic market features as fallback for Kimi model
        """
        features = []

        try:
            # Get market data
            market_data = state.get("market_data", state.get("mercado", {}))
            symbol_data = market_data.get(symbol)

            if isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                last_row = symbol_data.iloc[-1]

                # Basic OHLCV features
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    try:
                        val = float(last_row.get(col, 0.0))
                        features.append(val if np.isfinite(val) else 0.0)
                    except Exception:
                        features.append(0.0)

                # Add indicators if available
                for ind_name in ['rsi', 'macd', 'sma_20', 'sma_50', 'bollinger_upper', 'bollinger_lower']:
                    if indicators and ind_name in indicators:
                        try:
                            ind_series = indicators[ind_name]
                            if hasattr(ind_series, 'iloc'):
                                val = float(ind_series.iloc[-1])
                            else:
                                val = float(ind_series)
                            features.append(val if np.isfinite(val) else 0.0)
                        except Exception:
                            features.append(0.0)
                    else:
                        features.append(0.0)

            # Portfolio features
            portfolio = state.get("portfolio", {})
            if isinstance(portfolio.get("USDT"), dict):
                balance = portfolio.get("USDT", {}).get("free", 0.0)
                position = portfolio.get(symbol, {}).get("position", 0.0)
            else:
                balance = portfolio.get("USDT", 0.0)
                positions = portfolio.get("positions", {})
                position = positions.get(symbol, {}).get("size", 0.0)

            features.extend([balance, position, abs(position)])

        except Exception as e:
            logger.warning(f"Error building basic features: {e}")

        return features

    def _select_claude_base_features(self, features_df: pd.DataFrame) -> List[float]:
        """Select and expand base features for Claude model (~800 dimensions)"""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * 800

        last_row = features_df.iloc[-1]

        # Get all numeric columns
        numeric_cols = [
            c for c in features_df.columns
            if pd.api.types.is_numeric_dtype(features_df[c])
        ]

        values = []
        # Expand each numeric feature by creating variations
        for c in numeric_cols:
            try:
                base_val = float(last_row[c]) if np.isfinite(last_row[c]) else 0.0

                # Create multiple variations of each feature
                values.append(base_val)  # Original
                values.append(base_val * base_val)  # Squared
                values.append(np.sqrt(abs(base_val)) if base_val >= 0 else 0.0)  # Square root
                values.append(np.log(abs(base_val) + 1e-8))  # Log (with small epsilon)
                values.append(np.sin(base_val))  # Sine
                values.append(np.cos(base_val))  # Cosine

                # Rolling statistics if we have historical data
                if len(features_df) > 5:
                    col_data = features_df[c].tail(5).astype(float)
                    values.append(col_data.mean())  # 5-period mean
                    values.append(col_data.std())   # 5-period std
                    values.append(col_data.max())   # 5-period max
                    values.append(col_data.min())   # 5-period min

            except Exception:
                # Add zeros for failed features
                values.extend([0.0] * 10)  # 10 variations per feature

        # Pad or truncate to 800 dimensions
        if len(values) < 800:
            values.extend([0.0] * (800 - len(values)))
        else:
            values = values[:800]

        return values

    def _build_risk_aware_features(self, state: Dict[str, Any], symbol: str,
                                  market_data: Dict[str, pd.DataFrame],
                                  indicators: Dict[str, Any]) -> List[float]:
        """
        Build additional risk-aware features for Claude model (~160 dimensions)
        """
        features = []

        try:
            # Portfolio risk features
            portfolio = state.get("portfolio", {})
            if isinstance(portfolio.get("USDT"), dict):
                # New portfolio structure
                balance = portfolio.get("USDT", {}).get("free", 0.0)
                btc_position = portfolio.get("BTCUSDT", {}).get("position", 0.0)
                eth_position = portfolio.get("ETHUSDT", {}).get("position", 0.0)
            else:
                # Old portfolio structure
                balance = portfolio.get("USDT", 0.0)
                positions = portfolio.get("positions", {})
                btc_position = positions.get("BTCUSDT", {}).get("size", 0.0)
                eth_position = positions.get("ETHUSDT", {}).get("size", 0.0)

            features.extend([
                balance, btc_position, eth_position,
                balance * 0.01,  # Scaled balance
                abs(btc_position), abs(eth_position),  # Absolute positions
                btc_position + eth_position,  # Total position
            ])

            # Market volatility features
            for sym in ['BTCUSDT', 'ETHUSDT']:
                df = market_data.get(sym)
                if isinstance(df, pd.DataFrame) and not df.empty and len(df) > 10:
                    close_prices = df['close'].tail(20).astype(float)
                    returns = close_prices.pct_change().dropna()

                    features.extend([
                        returns.std(),  # Volatility
                        returns.mean(),  # Mean return
                        returns.skew(),  # Skewness
                        returns.kurtosis(),  # Kurtosis
                        (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0],  # Total return
                    ])
                else:
                    features.extend([0.0] * 5)

            # Cross-market correlation features
            btc_df = market_data.get('BTCUSDT')
            eth_df = market_data.get('ETHUSDT')
            if (isinstance(btc_df, pd.DataFrame) and not btc_df.empty and
                isinstance(eth_df, pd.DataFrame) and not eth_df.empty and
                len(btc_df) > 10 and len(eth_df) > 10):

                btc_returns = btc_df['close'].tail(20).pct_change().dropna()
                eth_returns = eth_df['close'].tail(20).pct_change().dropna()

                if len(btc_returns) > 5 and len(eth_returns) > 5:
                    corr = btc_returns.corr(eth_returns)
                    features.extend([
                        corr,  # Correlation
                        corr * corr,  # Squared correlation
                        abs(corr),  # Absolute correlation
                        np.sign(corr),  # Correlation direction
                    ])
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 4)

            # Time-based features
            import datetime
            now = datetime.datetime.now()
            features.extend([
                now.hour / 24.0,  # Hour of day (normalized)
                now.weekday() / 7.0,  # Day of week (normalized)
                np.sin(2 * np.pi * now.hour / 24.0),  # Cyclic hour
                np.cos(2 * np.pi * now.hour / 24.0),  # Cyclic hour
            ])

        except Exception as e:
            logger.warning(f"Error building risk-aware features: {e}")
            features = [0.0] * 160

        # Ensure exactly 160 features
        if len(features) < 160:
            features.extend([0.0] * (160 - len(features)))
        elif len(features) > 160:
            features = features[:160]

        return features

    def _build_full_observation(self, state: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build complete 257-dimensional observation for FinRL model.
        """
        try:
            # Get market data from state
            market_data = state.get("market_data", state.get("mercado", {}))
            if not market_data:
                logger.error("❌ No market data available in state")
                return None

            # Build features DataFrame for current symbol
            features_df = self._build_features_dataframe(symbol, market_data, indicators)
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # Build features_by_symbol dict
            features_by_symbol = {}
            for sym, df in market_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # For simplicity, use market data as features (could be enhanced)
                    features_by_symbol[sym] = df.copy()

            # Select base features (246 dimensions)
            base_features = self._select_base_features_row(features_df)

            # Build cross/L3 features (11 dimensions)
            cross_features = self._build_cross_l3_features(market_data, features_by_symbol)

            # Combine features: 246 base + 11 cross = 257 total
            full_features = base_features + cross_features

            if len(full_features) != 257:
                logger.error(f"❌ Feature dimension mismatch: got {len(full_features)}, expected 257")
                return None

            logger.debug(f"✅ Built 257-dimensional observation: {len(base_features)} base + {len(cross_features)} cross")
            return np.array(full_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building full observation: {e}")
            return None

    def _build_features_dataframe(self, symbol: str, market_data: Dict[str, pd.DataFrame],
                                 indicators: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Build features DataFrame for a symbol combining market data and indicators.
        """
        try:
            df = market_data.get(symbol)
            # ✅ FIXED: Check if df is dict first, then convert to DataFrame
            if isinstance(df, dict):
                if not df:
                    return None
                df = pd.DataFrame([df])  # Convert dict to DataFrame
            elif df is None:
                return None
            elif hasattr(df, 'empty') and df.empty:
                return None

            # Start with market data
            features_df = df.copy()

            # Add indicators as columns
            for indicator_name, indicator_series in indicators.items():
                if hasattr(indicator_series, 'iloc'):
                    try:
                        features_df[indicator_name] = indicator_series
                    except Exception as e:
                        logger.warning(f"Failed to add indicator {indicator_name}: {e}")

            return features_df

        except Exception as e:
            logger.error(f"Error building features DataFrame for {symbol}: {e}")
            return None

    # Feature engineering constants
    FINRL_OBS_DIM = 257
    BASE_FEATURES_DIM = 246
    CROSS_FEATURES_DIM = 11

    def _select_base_features_row(self, features_df: pd.DataFrame) -> List[float]:
        """Selects up to 246 numeric features from the last row of the features DataFrame."""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * self.BASE_FEATURES_DIM

        last_row = features_df.iloc[-1]

        # Get all numeric columns
        numeric_cols = [
            c for c in features_df.columns
            if pd.api.types.is_numeric_dtype(features_df[c])
        ]

        values = []
        for c in numeric_cols[:self.BASE_FEATURES_DIM]:
            try:
                v = last_row[c]
                values.append(float(v) if np.isfinite(v) else 0.0)
            except Exception:
                values.append(0.0)

        # Pad with zeros if needed
        if len(values) < self.BASE_FEATURES_DIM:
            values.extend([0.0] * (self.BASE_FEATURES_DIM - len(values)))

        return values[:self.BASE_FEATURES_DIM]

    def _compute_eth_btc_ratio(self, market_data: Dict[str, pd.DataFrame],
                              features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute ETH/BTC close ratio with fallback"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = float(eth["close"].iloc[-1])
                btc_close = float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Compute 30-sample correlation between BTC and ETH close returns"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0
            if eth.empty or btc.empty or "close" not in eth.columns or "close" not in btc.columns:
                return 0.0
            n = 30
            eth_close = eth["close"].astype(float).tail(n)
            btc_close = btc["close"].astype(float).tail(n)
            common_idx = eth_close.index.intersection(btc_close.index)
            eth_close = eth_close.loc[common_idx]
            btc_close = btc_close.loc[common_idx]
            if len(eth_close) < 3:
                return 0.0
            eth_ret = eth_close.pct_change().dropna()
            btc_ret = btc_close.pct_change().dropna()
            common_idx = eth_ret.index.intersection(btc_ret.index)
            if len(common_idx) < 3:
                return 0.0
            corr = float(np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])[0, 1])
            if np.isfinite(corr):
                return corr
        except Exception:
            pass
        return 0.0

    def _compute_spread_pct(self, market_data: Dict[str, pd.DataFrame],
                           features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute (BTC - ETH)/BTC as simple spread proxy in %"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = float(eth["close"].iloc[-1])
                btc_close = float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0

    def _build_cross_l3_features(self, market_data: Dict[str, pd.DataFrame],
                                features_by_symbol: Dict[str, pd.DataFrame]) -> List[float]:
        """Build 11 cross/L3 features"""
        feats = []

        feats.append(self._compute_eth_btc_ratio(market_data, features_by_symbol))
        feats.append(self._compute_btc_eth_corr30(market_data))
        feats.append(self._compute_spread_pct(market_data, features_by_symbol))

        def pick_feature(df_map: Dict[str, pd.DataFrame], key: str, default: float) -> float:
            try:
                btc_df = df_map.get("BTCUSDT")
                if isinstance(btc_df, pd.DataFrame) and key in btc_df.columns and not btc_df.empty:
                    v = float(btc_df[key].iloc[-1])
                    return v if np.isfinite(v) else default
            except Exception:
                pass
            return default

        # L3 features (with defaults since L3 context may not be available)
        feats.append(pick_feature(features_by_symbol, "l3_regime", 0.5))  # Neutral regime
        feats.append(pick_feature(features_by_symbol, "l3_risk_appetite", 0.5))  # Moderate risk
        feats.append(pick_feature(features_by_symbol, "l3_alloc_BTC", 0.5))  # 50% BTC
        feats.append(pick_feature(features_by_symbol, "l3_alloc_ETH", 0.3))  # 30% ETH
        feats.append(pick_feature(features_by_symbol, "l3_alloc_CASH", 0.2))  # 20% CASH

        # Volume ratio and correlation features
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame):
                n = 20
                v_eth = eth["volume"].astype(float).tail(n)
                v_btc = btc["volume"].astype(float).tail(n)
                common_idx = v_eth.index.intersection(v_btc.index)
                v_eth = v_eth.loc[common_idx]
                v_btc = v_btc.loc[common_idx]
                ratio = float(v_eth.mean() / v_btc.mean()) if v_btc.mean() != 0 else 0.0
                feats.append(ratio if np.isfinite(ratio) else 0.0)
                v_eth_ret = v_eth.pct_change().dropna()
                v_btc_ret = v_btc.pct_change().dropna()
                common_idx = v_eth_ret.index.intersection(v_btc_ret.index)
                if len(common_idx) >= 3:
                    corr = float(np.corrcoef(v_eth_ret.loc[common_idx], v_btc_ret.loc[common_idx])[0, 1])
                    feats.append(corr if np.isfinite(corr) else 0.0)
                else:
                    feats.append(0.0)
            else:
                feats.extend([0.0, 0.0])
        except Exception:
            feats.extend([0.0, 0.0])

        # MACD difference
        try:
            btc_f = features_by_symbol.get("BTCUSDT")
            eth_f = features_by_symbol.get("ETHUSDT")
            if (isinstance(btc_f, pd.DataFrame) and "macd" in btc_f.columns and not btc_f.empty and
                isinstance(eth_f, pd.DataFrame) and "macd" in eth_f.columns and not eth_f.empty):
                val = float(btc_f["macd"].iloc[-1]) - float(eth_f["macd"].iloc[-1])
                feats.append(val if np.isfinite(val) else 0.0)
            else:
                feats.append(0.0)
        except Exception:
            feats.append(0.0)

        # Ensure exactly 11 features
        if len(feats) < self.CROSS_FEATURES_DIM:
            feats.extend([0.0] * (self.CROSS_FEATURES_DIM - len(feats)))
        elif len(feats) > self.CROSS_FEATURES_DIM:
            feats = feats[:self.CROSS_FEATURES_DIM]

        return [float(x) for x in feats]

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
            if gym is None:
                logger.error("gymnasium not available for torch model loading")
                return False

            # Create model instance with the right architecture
            from stable_baselines3.ppo.policies import ActorCriticPolicy

            # Define observation space (matching the saved model)
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(257,), dtype=np.float32)  # Match saved model
            action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Single output

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

    def load_deepseek_model(self, zip_path: str) -> bool:
        """Load DeepSeek model from ZIP with custom logic - matches training configuration"""
        try:
            logger.info(f"Loading DeepSeek model from: {zip_path}")

            from stable_baselines3 import PPO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading DeepSeek model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            # The DeepSeek model was trained with specific policy_kwargs that are stored in the model
            self.model = PPO.load(zip_path, device=device)

            self.is_loaded = True
            logger.info(f"DeepSeek model loaded successfully! Policy: {type(self.model.policy)}")
            return True

        except ImportError as e:
            logger.error(f"stable_baselines3 not available for DeepSeek: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {e}", exc_info=True)
            return False

    def load_claude_model(self, zip_path: str) -> bool:
        """Load Claude model from ZIP with custom logic - matches training configuration with RiskAwareExtractor"""
        try:
            logger.info(f"Loading Claude model from: {zip_path}")

            from stable_baselines3 import PPO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading Claude model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            # The Claude model was trained with specific policy_kwargs including RiskAwareExtractor
            # that are stored in the model file
            self.model = PPO.load(zip_path, device=device)

            self.is_loaded = True
            logger.info(f"Claude model loaded successfully! Policy: {type(self.model.policy)}")
            return True

        except ImportError as e:
            logger.error(f"stable_baselines3 not available for Claude: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading Claude model: {e}", exc_info=True)
            return False

    def load_kimi_model(self, zip_path: str) -> bool:
        """Load Kimi model from ZIP with custom logic - matches training configuration with default policy kwargs"""
        try:
            logger.info(f"Loading Kimi model from: {zip_path}")

            from stable_baselines3 import PPO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Loading Kimi model with device={device}")

            # Load without policy_kwargs to avoid mismatch with stored model
            # The Kimi model was trained with default policy_kwargs (empty dict)
            # that are stored in the model file
            self.model = PPO.load(zip_path, device=device)

            self.is_loaded = True
            logger.info(f"Kimi model loaded successfully! Policy: {type(self.model.policy)}")
            return True

        except ImportError as e:
            logger.error(f"stable_baselines3 not available for Kimi: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading Kimi model: {e}", exc_info=True)
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

    def _action_to_signal(self, action_value, symbol: str, value=None):
        """
        Convert model action value to a tactical signal with improved numerical stability
        
        Args:
            action_value: Action value, can be tensor or float, representing model's action
            symbol: The trading symbol
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
            
            # ✅ FIXED: More aggressive thresholds to generate buy/sell signals
            # Use sigmoid-like smoothing for transitions
            def smooth_prob(x, center, width=0.2):  # Wider width for smoother transitions
                return 1 / (1 + np.exp(-(x - center) / width))

            # ✅ FIXED: More aggressive thresholds - smaller hold zone
            sell_threshold = 0.4  # Was 0.33
            buy_threshold = 0.6   # Was 0.66

            # Calculate probabilities with more aggressive decision making
            if action_val <= sell_threshold:
                # Strong sell signal
                sell_prob = max(0.6, 1.0 - (action_val / sell_threshold))
                buy_prob = 0.0
                hold_prob = min(0.4, action_val / sell_threshold)
            elif action_val >= buy_threshold:
                # Strong buy signal
                buy_prob = max(0.6, (action_val - buy_threshold) / (1.0 - buy_threshold) + 0.4)
                sell_prob = 0.0
                hold_prob = min(0.4, (1.0 - action_val) / (1.0 - buy_threshold))
            else:
                # Middle zone - still allow some trading with lower confidence
                mid_point = (sell_threshold + buy_threshold) / 2
                if action_val < mid_point:
                    # Lean towards sell
                    sell_prob = 0.4
                    buy_prob = 0.1
                    hold_prob = 0.5
                else:
                    # Lean towards buy
                    sell_prob = 0.1
                    buy_prob = 0.4
                    hold_prob = 0.5

            # Ensure probabilities are properly bounded
            sell_prob = max(0.0, min(1.0, sell_prob))
            buy_prob = max(0.0, min(1.0, buy_prob))
            hold_prob = max(0.0, min(1.0, hold_prob))

            # Re-normalize to ensure they sum to 1
            total = sell_prob + buy_prob + hold_prob
            if total > 0:
                sell_prob /= total
                buy_prob /= total
                hold_prob /= total
            
            # Ensure probabilities sum to 1 and are bounded [0,1]
            total = sell_prob + buy_prob + hold_prob
            if total > 0:
                sell_prob /= total
                buy_prob /= total
                hold_prob /= total
            
            # Calculate base confidence from probabilities
            action_strength = max(buy_prob, sell_prob)  # Hold prob doesn't affect strength
            base_confidence = action_strength
            
            # Validate and incorporate value head prediction if available
            value_confidence = None
            if value is not None:
                try:
                    if torch.is_tensor(value):
                        val = value.detach().cpu().item()
                    else:
                        val = float(value)
                    # Scale value to [0,1] confidence range
                    value_confidence = (np.tanh(val / 2) + 1) / 2
                except (ValueError, TypeError, torch.cuda.CUDAError) as e:
                    logger.warning(f"Value head validation failed: {e}")
            
            # Combine base confidence with value prediction
            confidence = base_confidence
            if value_confidence is not None:
                confidence = (base_confidence + value_confidence) / 2
                
            # ✅ FIXED: Determine action based on highest probability with fallback
            if buy_prob > max(sell_prob, hold_prob) and (buy_prob > 0.3 or confidence > 0.5):
                side = "buy"
                strength = confidence
                logger.debug(f"🟢 BUY signal: prob={buy_prob:.3f}, confidence={confidence:.3f}")
            elif sell_prob > max(buy_prob, hold_prob) and (sell_prob > 0.3 or confidence > 0.5):
                side = "sell"
                strength = action_strength
                logger.debug(f"🔴 SELL signal: prob={sell_prob:.3f}, confidence={confidence:.3f}")
            else:
                # Fallback: if all probabilities are very low, make a decision based on action_val
                if action_val < 0.45:
                    side = "sell"
                    strength = 0.6
                    logger.debug(f"🟠 FALLBACK SELL: action_val={action_val:.3f} too low")
                elif action_val > 0.55:
                    side = "buy"
                    strength = 0.6
                    logger.debug(f"🟠 FALLBACK BUY: action_val={action_val:.3f} too high")
                else:
                    side = "hold"
                    strength = min(0.4, action_strength)
                    logger.debug(f"⚪ HOLD signal: all probs low, action_val={action_val:.3f}")

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
