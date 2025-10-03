# l2_tactic/observation_builders.py
"""
Observation building methods for different FinRL models
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger

# Import safe_float for robust array handling
from .utils import safe_float


class ObservationBuilders:
    """Collection of methods for building observations for different models"""

    @staticmethod
    def _build_features_dataframe(symbol: str, market_data: Dict[str, pd.DataFrame],
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

    @staticmethod
    def build_legacy_observation(state_or_market_data: Dict[str, Any], symbol: str, indicators: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build legacy 13-dimensional observation for Gemini and other 13-feature models.

        Args:
            state_or_market_data: Either a full state dict (containing 'market_data'/'mercado') or direct market_data dict
            symbol: Trading symbol
            indicators: Technical indicators dict
        """
        try:
            # Handle both parameter formats: state dict or direct market_data dict
            if isinstance(state_or_market_data, dict) and ('market_data' in state_or_market_data or 'mercado' in state_or_market_data):
                # Full state dict format
                market_data = state_or_market_data.get("market_data", state_or_market_data.get("mercado", {}))
            else:
                # Direct market_data dict format
                market_data = state_or_market_data

            if not market_data:
                logger.error("❌ No market data available")
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
                            value = safe_float(indicator_series.iloc[-1])
                        else:
                            value = safe_float(indicator_series)
                    # Then try from market data
                    elif hasattr(last_row, 'get'):
                        value = safe_float(last_row.get(f, 0.0))
                    elif isinstance(last_row, pd.Series) and f in last_row.index:
                        value = safe_float(last_row[f])
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

    @staticmethod
    def build_gemini_obs(market_data: dict, symbol: str, indicators: dict = None):
        """
        Deprecated: Use build_legacy_observation instead.
        Builds legacy 13-dimensional observation for Gemini models.
        This function exists for backward compatibility.
        """
        logger.warning("⚠️ build_gemini_obs is deprecated. Use build_legacy_observation instead.")
        return ObservationBuilders.build_legacy_observation(market_data, symbol, indicators or {})

    @staticmethod
    def build_grok_obs(market_data: dict, symbol: str, indicators: dict = None):
        """Construye observación para Grok - manejo flexible según el modelo"""
        try:
            # Get market data for symbol
            symbol_data = market_data.get(symbol)
            if symbol_data is None:
                logger.error(f"❌ No market data for {symbol}")
                return None

            # Extract last row
            if isinstance(symbol_data, pd.DataFrame):
                last_row = symbol_data.iloc[-1]
            else:
                last_row = symbol_data

            # Build flexible features for Grok
            # Grok can handle variable dimensions, so we build comprehensive features
            features = []

            # Basic OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    val = safe_float(last_row.get(col, 0.0)) if hasattr(last_row, 'get') else safe_float(last_row[col])
                    features.append(val if np.isfinite(val) else 0.0)
                except Exception:
                    features.append(0.0)

            # Technical indicators
            tech_indicators = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50',
                             'bollinger_upper', 'bollinger_lower', 'ema_12', 'ema_26']

            for ind in tech_indicators:
                if indicators and ind in indicators:
                    try:
                        ind_series = indicators[ind]
                        if hasattr(ind_series, 'iloc'):
                            val = safe_float(ind_series.iloc[-1])
                        else:
                            val = safe_float(ind_series)
                        features.append(val if np.isfinite(val) else 0.0)
                    except Exception:
                        features.append(0.0)
                else:
                    features.append(0.0)

            # Portfolio features
            portfolio = market_data.get('portfolio', {})
            if isinstance(portfolio.get("USDT"), dict):
                balance = portfolio.get("USDT", {}).get("free", 0.0)
                position = portfolio.get(symbol, {}).get("position", 0.0)
            else:
                balance = portfolio.get("USDT", 0.0)
                positions = portfolio.get("positions", {})
                position = positions.get(symbol, {}).get("size", 0.0)

            features.extend([balance, position, abs(position)])

            # Convert to numpy array
            obs = np.array(features, dtype=np.float32)

            logger.debug(f"✅ Built Grok observation: {len(features)} features")
            return obs

        except Exception as e:
            logger.error(f"❌ Error building Grok observation: {e}")
            return None

    @staticmethod
    def build_multiasset_obs(market_data: dict, symbol: str, indicators: dict = None):
        """Construye observación multiasset de 971 features para Claude/DeepSeek/Kimi"""
        try:
            # Build features DataFrame
            features_df = ObservationBuilders._build_features_dataframe(symbol, market_data, indicators or {})
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # Build cross-asset features
            features_by_symbol = {}
            for sym, df in market_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    features_by_symbol[sym] = df.copy()

            # Select base features (800 dimensions)
            base_features = ObservationBuilders._select_claude_base_features(features_df)

            # Build cross/L3 features (11 dimensions)
            cross_features = ObservationBuilders._build_cross_l3_features(market_data, features_by_symbol)

            # Build risk-aware features (160 dimensions)
            risk_features = ObservationBuilders._build_risk_aware_features({}, symbol, market_data, indicators or {})

            # Combine: 800 + 11 + 160 = 971
            all_features = base_features + cross_features + risk_features

            if len(all_features) != 971:
                logger.error(f"❌ Multiasset feature dimension mismatch: got {len(all_features)}, expected 971")
                return None

            logger.debug(f"✅ Built multiasset observation: {len(base_features)} base + {len(cross_features)} cross + {len(risk_features)} risk")
            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building multiasset observation: {e}")
            return None

    @staticmethod
    def build_hrm_native_obs(market_data: dict, symbol: str, indicators: dict = None):
        """Construye observación HRM nativa de 85 dimensiones para DeepSeek"""
        try:
            # Build features DataFrame
            features_df = ObservationBuilders._build_features_dataframe(symbol, market_data, indicators or {})
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # HRM native features (85 dimensions) - comprehensive market and technical features
            last_row = features_df.iloc[-1]

            # Basic price features (10 dimensions)
            basic_features = []
            for col in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    val = safe_float(last_row.get(col, 0.0))
                    basic_features.append(val if np.isfinite(val) else 0.0)
                except Exception:
                    basic_features.append(0.0)

            # Add price ratios and changes
            try:
                close = safe_float(last_row.get('close', 0.0))
                open_price = safe_float(last_row.get('open', 0.0))
                high = safe_float(last_row.get('high', 0.0))
                low = safe_float(last_row.get('low', 0.0))

                # Intraday change
                intraday_change = (close - open_price) / open_price if open_price != 0 else 0.0
                basic_features.append(intraday_change)

                # Daily range
                daily_range = (high - low) / close if close != 0 else 0.0
                basic_features.append(daily_range)

                # Volume intensity (normalized)
                volume = safe_float(last_row.get('volume', 0.0))
                volume_intensity = volume / close if close != 0 else 0.0
                basic_features.append(volume_intensity)

                # Gap indicator
                basic_features.append(0.0)  # Placeholder for gap
            except Exception:
                basic_features.extend([0.0, 0.0, 0.0, 0.0])

            # Technical indicators (50 dimensions)
            tech_features = []
            tech_indicators = [
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi', 'roc', 'mom', 'adx', 'di_plus', 'di_minus',
                'trix', 'keltner_upper', 'keltner_middle', 'keltner_lower', 'ichimoku_tenkan',
                'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b', 'parabolic_sar',
                'dpo', 'vortex_pos', 'vortex_neg', 'chande_kroll_stop_long', 'chande_kroll_stop_short',
                'supertrend', 'aroon_up', 'aroon_down', 'tsf', 'special_k', 'special_d',
                'elder_force_index', 'elder_thermometer', 'market_mechanics', 'gopalakrishnan_range_index',
                'balance_of_power', 'volume_price_trend', 'ease_of_movement', 'negative_volume_index'
            ]

            for ind in tech_indicators:
                if indicators and ind in indicators:
                    try:
                        ind_series = indicators[ind]
                        if hasattr(ind_series, 'iloc'):
                            val = safe_float(ind_series.iloc[-1])
                        else:
                            val = safe_float(ind_series)
                        tech_features.append(val if np.isfinite(val) else 0.0)
                    except Exception:
                        tech_features.append(0.0)
                else:
                    tech_features.append(0.0)

            # Market regime features (10 dimensions)
            regime_features = []

            # Volatility measures
            try:
                if len(features_df) > 5:
                    close_prices = features_df['close'].tail(10).astype(float)
                    returns = close_prices.pct_change().dropna()
                    if len(returns) > 0:
                        regime_features.append(returns.std())  # Volatility
                        regime_features.append(returns.mean())  # Mean return
                        regime_features.append(returns.skew())  # Skewness
                        regime_features.append(returns.kurtosis())  # Kurtosis
                    else:
                        regime_features.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    regime_features.extend([0.0, 0.0, 0.0, 0.0])
            except Exception:
                regime_features.extend([0.0, 0.0, 0.0, 0.0])

            # Trend strength
            try:
                if len(features_df) > 20:
                    sma20 = features_df['close'].tail(20).mean()
                    sma50 = features_df['close'].tail(50).mean() if len(features_df) > 50 else features_df['close'].mean()
                    trend_strength = (sma20 - sma50) / sma50 if sma50 != 0 else 0.0
                    regime_features.append(trend_strength)
                else:
                    regime_features.append(0.0)
            except Exception:
                regime_features.append(0.0)

            # Momentum indicators
            regime_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # Placeholders

            # Cross-asset features (15 dimensions) - simplified
            cross_features = []

            # BTC-ETH ratio and correlation
            cross_features.extend([
                ObservationBuilders._compute_eth_btc_ratio(market_data, {symbol: features_df}),
                ObservationBuilders._compute_btc_eth_corr30(market_data),
                ObservationBuilders._compute_spread_pct(market_data, {symbol: features_df})
            ])

            # Additional cross-market features
            cross_features.extend([0.0] * 12)  # Placeholders for additional cross-market features

            # Combine all features: 10 + 50 + 10 + 15 = 85
            all_features = basic_features + tech_features + regime_features + cross_features

            # Ensure exactly 85 dimensions
            if len(all_features) < 85:
                all_features.extend([0.0] * (85 - len(all_features)))
            elif len(all_features) > 85:
                all_features = all_features[:85]

            if len(all_features) != 85:
                logger.error(f"❌ HRM native feature dimension mismatch: got {len(all_features)}, expected 85")
                return None

            logger.debug(f"✅ Built HRM native observation: {len(basic_features)} basic + {len(tech_features)} tech + {len(regime_features)} regime + {len(cross_features)} cross")
            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building HRM native observation: {e}")
            return None

    @staticmethod
    def build_generic_obs(market_data: dict, symbol: str, indicators: dict = None, expected_dims: int = 257):
        """Construye observación genérica para otros modelos"""
        try:
            # Build features DataFrame
            features_df = ObservationBuilders._build_features_dataframe(symbol, market_data, indicators or {})
            if features_df is None or features_df.empty:
                logger.error(f"❌ Failed to build features DataFrame for {symbol}")
                return None

            # Select base features (246 dimensions for standard FinRL)
            if expected_dims == 257:
                base_features = ObservationBuilders._select_base_features_row(features_df)
                cross_features = ObservationBuilders._build_cross_l3_features(market_data, {})
                all_features = base_features + cross_features
            else:
                # Generic: use all numeric features, pad/truncate to expected_dims
                last_row = features_df.iloc[-1]
                numeric_cols = [c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])]
                all_features = []
                for c in numeric_cols[:expected_dims]:
                    try:
                        v = safe_float(last_row[c])
                        all_features.append(v if np.isfinite(v) else 0.0)
                    except Exception:
                        all_features.append(0.0)

            # Pad or truncate to expected dimensions
            if len(all_features) < expected_dims:
                all_features.extend([0.0] * (expected_dims - len(all_features)))
            elif len(all_features) > expected_dims:
                all_features = all_features[:expected_dims]

            if len(all_features) != expected_dims:
                logger.error(f"❌ Generic feature dimension mismatch: got {len(all_features)}, expected {expected_dims}")
                return None

            logger.debug(f"✅ Built generic observation: {expected_dims} features")
            return np.array(all_features, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Error building generic observation: {e}")
            return None

    @staticmethod
    def _select_claude_base_features(features_df: pd.DataFrame) -> List[float]:
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
                base_val = safe_float(last_row[c]) if np.isfinite(last_row[c]) else 0.0

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

    @staticmethod
    def _select_base_features_row(features_df: pd.DataFrame) -> List[float]:
        """Selects up to 246 numeric features from the last row of the features DataFrame."""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * 246

        last_row = features_df.iloc[-1]

        # Get all numeric columns
        numeric_cols = [
            c for c in features_df.columns
            if pd.api.types.is_numeric_dtype(features_df[c])
        ]

        values = []
        for c in numeric_cols[:246]:
            try:
                v = last_row[c]
                values.append(safe_float(v) if np.isfinite(v) else 0.0)
            except Exception:
                values.append(0.0)

        # Pad with zeros if needed
        if len(values) < 246:
            values.extend([0.0] * (246 - len(values)))

        return values[:246]

    @staticmethod
    def _build_cross_l3_features(market_data: Dict[str, pd.DataFrame],
                                features_by_symbol: Dict[str, pd.DataFrame]) -> List[float]:
        """Build 11 cross/L3 features"""
        feats = []

        feats.append(ObservationBuilders._compute_eth_btc_ratio(market_data, features_by_symbol))
        feats.append(ObservationBuilders._compute_btc_eth_corr30(market_data))
        feats.append(ObservationBuilders._compute_spread_pct(market_data, features_by_symbol))

        def pick_feature(df_map: Dict[str, pd.DataFrame], key: str, default: float) -> float:
            try:
                btc_df = df_map.get("BTCUSDT")
                if isinstance(btc_df, pd.DataFrame) and key in btc_df.columns and not btc_df.empty:
                    v = safe_float(btc_df[key].iloc[-1])
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
                ratio = safe_float(v_eth.mean() / v_btc.mean()) if v_btc.mean() != 0 else 0.0
                feats.append(ratio if np.isfinite(ratio) else 0.0)
                v_eth_ret = v_eth.pct_change().dropna()
                v_btc_ret = v_btc.pct_change().dropna()
                common_idx = v_eth_ret.index.intersection(v_btc_ret.index)
                if len(common_idx) >= 3:
                    corr = safe_float(np.corrcoef(v_eth_ret.loc[common_idx], v_btc_ret.loc[common_idx])[0, 1])
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
                val = safe_float(btc_f["macd"].iloc[-1]) - safe_float(eth_f["macd"].iloc[-1])
                feats.append(val if np.isfinite(val) else 0.0)
            else:
                feats.append(0.0)
        except Exception:
            feats.append(0.0)

        # Ensure exactly 11 features
        if len(feats) < 11:
            feats.extend([0.0] * (11 - len(feats)))
        elif len(feats) > 11:
            feats = feats[:11]

        return [safe_float(x) for x in feats]

    @staticmethod
    def _build_risk_aware_features(state: Dict[str, Any], symbol: str,
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

    @staticmethod
    def _compute_eth_btc_ratio(market_data: Dict[str, pd.DataFrame],
                              features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute ETH/BTC close ratio with fallback"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = safe_float(eth["close"].iloc[-1])
                btc_close = safe_float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _compute_btc_eth_corr30(market_data: Dict[str, pd.DataFrame]) -> float:
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
            corr = safe_float(np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])[0, 1])
            if np.isfinite(corr):
                return corr
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _compute_spread_pct(market_data: Dict[str, pd.DataFrame],
                           features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute (BTC - ETH)/BTC as simple spread proxy in %"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = safe_float(eth["close"].iloc[-1])
                btc_close = safe_float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0
