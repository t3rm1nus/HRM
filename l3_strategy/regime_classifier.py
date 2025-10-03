# regime_classifier.py
"""
Comprehensive Market Regime Classifier - ENHANCED VERSION with Setup Detection

Now includes oversold/overbought setup detection within range regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from core.logging import logger
from .regime_features import calculate_regime_features

class MarketRegimeClassifier:
    """
    Advanced market regime classifier with setup detection for range markets.
    """

    def __init__(self):
        """Initialize classifier with calibrated crypto thresholds"""
        self.target_hours = 6
        self.min_data_points = 48
        
        self.calculation_window = None
        self.detected_timeframe = None

        # CALIBRATED THRESHOLDS FOR CRYPTO
        self.thresholds = {
            'trend': {
                'strong_change': 0.020,
                'moderate_change': 0.010,
                'weak_change': 0.004,
                'min_r2': 0.4,
                'min_adx': 20
            },
            'range': {
                'max_directional_move': 0.003,  # 0.3% for strict range
                'tight_bb_width': 0.004,
                'normal_bb_width': 0.010,
                'min_touches': 2,
                # NEW: Setup detection thresholds
                'oversold_rsi': 40,
                'overbought_rsi': 60,
                'setup_min_adx': 20,
                'setup_bb_width': 0.005
            },
            'volatile': {
                'volatility_multiplier': 2.0,
                'min_vol_periods': 24
            },
            'breakout': {
                'bb_break_threshold': 0.015,
                'volume_spike': 1.5,
                'momentum_conf': 0.6
            }
        }

    def _detect_timeframe(self, df: pd.DataFrame) -> int:
        """Detect actual timeframe from data in minutes"""
        try:
            if len(df) >= 2 and isinstance(df.index, pd.DatetimeIndex):
                time_diff = (df.index[-1] - df.index[-2]).total_seconds() / 60
                return max(1, int(time_diff))
            return 1
        except Exception as e:
            logger.warning(f"Could not detect timeframe: {e}, defaulting to 1min")
            return 1

    def _calculate_dynamic_window(self, timeframe_minutes: int) -> int:
        """Calculate window size for target hours"""
        target_minutes = self.target_hours * 60
        window = int(target_minutes / timeframe_minutes)
        return max(self.min_data_points, window)

    def classify_market_regime(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> Dict:
        """Main classification with dynamic timeframe detection and setup detection"""
        try:
            self.detected_timeframe = self._detect_timeframe(df)
            self.calculation_window = self._calculate_dynamic_window(self.detected_timeframe)
            
            logger.info(f"Detected timeframe: {self.detected_timeframe}min, using {self.calculation_window} candles for {self.target_hours}h analysis")

            if not self._validate_input_data(df):
                return self._create_error_result("insufficient_data")

            features_df = self._calculate_analysis_features(df)
            window_data = features_df.tail(self.calculation_window)
            
            price_change = (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]

            results = {}
            results['trending'] = self._classify_trending_regime(window_data, price_change)
            results['breakout'] = self._classify_breakout_regime(window_data)
            results['range'] = self._classify_range_regime(window_data, price_change)
            results['volatile'] = self._classify_volatile_regime(window_data)

            primary_regime, confidence, subtype = self._determine_primary_regime(results, price_change)

            result = {
                'primary_regime': primary_regime,
                'subtype': subtype,
                'confidence': confidence,
                'regime_scores': {
                    'TRENDING': results['trending']['score'],
                    'RANGE': results['range']['score'],
                    'VOLATILE': results['volatile']['score'],
                    'BREAKOUT': results['breakout']['score']
                },
                'metrics': self._extract_key_metrics(window_data, price_change),
                'metadata': {
                    'calculation_window': self.calculation_window,
                    'timeframe_minutes': self.detected_timeframe,
                    'data_points': len(window_data),
                    'price_change_pct': price_change,
                    'timestamp': pd.Timestamp.now()
                }
            }

            self._log_regime_classification(result, symbol)
            return result

        except Exception as e:
            logger.error(f"Error in regime classification for {symbol}: {e}")
            return self._create_error_result("classification_error")

    def _classify_trending_regime(self, window_data: pd.DataFrame, price_change: float) -> Dict:
        """Classify TRENDING with lowered thresholds for crypto"""
        try:
            prices = window_data['close'].values
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            r_squared = r_value ** 2

            adx = window_data['adx'].iloc[-1] if 'adx' in window_data.columns else 20
            rsi = window_data['rsi'].iloc[-1] if 'rsi' in window_data.columns else 50

            direction = "UP" if price_change > 0 else "DOWN"
            abs_change = abs(price_change)
            
            if abs_change > self.thresholds['trend']['strong_change'] and r_squared > 0.5:
                subtype = f"STRONG_{'BULL' if direction == 'UP' else 'BEAR'}"
                score = min(1.0, (abs_change / 0.03) * (r_squared / 0.7) * (adx / 35))
                
            elif abs_change > self.thresholds['trend']['moderate_change'] and r_squared > 0.4:
                subtype = f"MODERATE_{'BULL' if direction == 'UP' else 'BEAR'}"
                score = 0.70
                
            elif abs_change > self.thresholds['trend']['weak_change'] and r_squared > 0.3:
                subtype = f"WEAK_{'BULL' if direction == 'UP' else 'BEAR'}"
                score = 0.55
                
            else:
                subtype = None
                score = max(0.0, abs_change / 0.008)

            return {
                'score': score,
                'subtype': subtype,
                'metrics': {
                    'slope': slope,
                    'price_change_pct': price_change,
                    'r_squared': r_squared,
                    'adx': adx,
                    'rsi': rsi,
                    'direction': direction
                }
            }

        except Exception as e:
            logger.error(f"Error in trending regime classification: {e}")
            return {'score': 0.0, 'subtype': None, 'metrics': {}}

    def _classify_range_regime(self, window_data: pd.DataFrame, price_change: float) -> Dict:
        """Classify RANGE with SETUP DETECTION for oversold/overbought conditions"""
        try:
            abs_price_change = abs(price_change)
            
            # STRICT: Only < 0.3% movement allowed for range
            if abs_price_change > self.thresholds['range']['max_directional_move']:
                logger.info(f"Range rejected: directional move {abs_price_change:.2%} > {self.thresholds['range']['max_directional_move']:.2%}")
                return {
                    'score': 0.1,
                    'subtype': None,
                    'metrics': {
                        'rejection_reason': 'directional_movement_detected',
                        'price_change': price_change
                    }
                }

            # Calculate range metrics
            bb_upper = window_data['bollinger_upper'].iloc[-1]
            bb_lower = window_data['bollinger_lower'].iloc[-1]
            bb_middle = window_data['bollinger_middle'].iloc[-1]
            current_price = window_data['close'].iloc[-1]
            
            rsi = window_data['rsi'].iloc[-1] if 'rsi' in window_data.columns else 50
            adx = window_data['adx'].iloc[-1] if 'adx' in window_data.columns else 20

            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0

            high_period = min(20, len(window_data))
            low_period = min(20, len(window_data))
            high_20 = window_data['high'].tail(high_period).max()
            low_20 = window_data['low'].tail(low_period).min()
            range_width = (high_20 - low_20) / ((high_20 + low_20) / 2)

            tolerance_upper = bb_upper * 0.998
            tolerance_lower = bb_lower * 1.002
            touches_upper = sum(window_data['high'].tail(min(24, len(window_data))) >= tolerance_upper)
            touches_lower = sum(window_data['low'].tail(min(24, len(window_data))) <= tolerance_lower)

            volatility = window_data['close'].pct_change().std()

            # NEW: SETUP DETECTION within range
            setup_detected = False
            setup_type = None
            
            if bb_width < self.thresholds['range']['setup_bb_width']:
                # Oversold setup: tight range + low RSI + moderate ADX
                if rsi < self.thresholds['range']['oversold_rsi'] and adx > self.thresholds['range']['setup_min_adx']:
                    subtype = "OVERSOLD_SETUP"
                    score = 0.85
                    setup_detected = True
                    setup_type = "oversold"
                    logger.info(f"🎯 OVERSOLD SETUP detected: RSI={rsi:.1f}, ADX={adx:.1f}, BB_width={bb_width:.4f}")
                
                # Overbought setup: tight range + high RSI + moderate ADX
                elif rsi > self.thresholds['range']['overbought_rsi'] and adx > self.thresholds['range']['setup_min_adx']:
                    subtype = "OVERBOUGHT_SETUP"
                    score = 0.85
                    setup_detected = True
                    setup_type = "overbought"
                    logger.info(f"🎯 OVERBOUGHT SETUP detected: RSI={rsi:.1f}, ADX={adx:.1f}, BB_width={bb_width:.4f}")
            
            # Standard range classification if no setup detected
            if not setup_detected:
                if bb_width < self.thresholds['range']['tight_bb_width'] and touches_upper >= 2 and touches_lower >= 2:
                    subtype = "TIGHT_RANGE"
                    score = 0.85
                    
                elif bb_width < self.thresholds['range']['normal_bb_width'] and range_width < 0.02:
                    subtype = "NORMAL_RANGE"
                    score = 0.70
                    
                elif range_width < 0.04:
                    subtype = "WIDE_RANGE"
                    score = 0.55
                    
                else:
                    subtype = None
                    score = 0.3

            return {
                'score': score,
                'subtype': subtype,
                'setup_detected': setup_detected,
                'setup_type': setup_type,
                'metrics': {
                    'bb_width': bb_width,
                    'range_width': range_width,
                    'touches_upper': touches_upper,
                    'touches_lower': touches_lower,
                    'volatility': volatility,
                    'rsi': rsi,
                    'adx': adx,
                    'price_position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5,
                    'directional_move': abs_price_change
                }
            }

        except Exception as e:
            logger.error(f"Error in range regime classification: {e}")
            return {'score': 0.0, 'subtype': None, 'metrics': {}}

    def _classify_volatile_regime(self, window_data: pd.DataFrame) -> Dict:
        """Classify VOLATILE regime"""
        try:
            min_vol_periods = min(24, len(window_data))
            current_volatility = window_data['close'].pct_change().tail(min_vol_periods).std()
            historical_volatility = window_data['close'].pct_change().tail(min(72, len(window_data))).std()

            if 'volume' in window_data.columns:
                volume_ma = window_data['volume'].tail(min(20, len(window_data))).mean()
                volume_current = window_data['volume'].iloc[-1]
            else:
                volume_ma = 1.0
                volume_current = 1.0

            lookback = min(72, len(window_data))
            price_move = abs(window_data['close'].iloc[-1] - window_data['close'].iloc[-lookback]) / window_data['close'].iloc[-lookback]

            if 'rsi' in window_data.columns:
                rsi_tail = window_data['rsi'].tail(min(24, len(window_data)))
                rsi_min = rsi_tail.min()
                rsi_max = rsi_tail.max()
                rsi_range = rsi_max - rsi_min
            else:
                rsi_range = 0

            vol_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0

            if vol_ratio > self.thresholds['volatile']['volatility_multiplier'] and rsi_range > 35:
                subtype = "HIGH_VOLATILITY"
                score = min(1.0, vol_ratio / 3.0)
            else:
                subtype = None
                score = max(0.0, (vol_ratio - 1.0) / 2.0)

            return {
                'score': score,
                'subtype': subtype,
                'metrics': {
                    'current_volatility': current_volatility,
                    'historical_volatility': historical_volatility,
                    'vol_ratio': vol_ratio,
                    'price_move': price_move,
                    'rsi_range': rsi_range,
                    'volume_ratio': volume_current / volume_ma if volume_ma > 0 else 1.0
                }
            }

        except Exception as e:
            logger.error(f"Error in volatile regime classification: {e}")
            return {'score': 0.0, 'subtype': None, 'metrics': {}}

    def _classify_breakout_regime(self, window_data: pd.DataFrame) -> Dict:
        """Classify BREAKOUT regime"""
        try:
            current_price = window_data['close'].iloc[-1]
            bb_upper = window_data['bollinger_upper'].iloc[-1]
            bb_lower = window_data['bollinger_lower'].iloc[-1]

            breakout_up = current_price > bb_upper * (1 + self.thresholds['breakout']['bb_break_threshold'])
            breakout_down = current_price < bb_lower * (1 - self.thresholds['breakout']['bb_break_threshold'])

            if 'volume' in window_data.columns:
                volume_ma = window_data['volume'].tail(min(20, len(window_data))).mean()
                volume_current = window_data['volume'].iloc[-1]
                volume_spike = volume_current / volume_ma if volume_ma > 0 else 1.0
            else:
                volume_spike = 1.0

            momentum_5 = window_data['momentum_5'].iloc[-1] if 'momentum_5' in window_data.columns else 0
            momentum_10 = window_data['momentum_10'].iloc[-1] if 'momentum_10' in window_data.columns else 0

            consolidation_periods = min(12, len(window_data))
            recent_high = window_data['high'].tail(consolidation_periods).max()
            recent_low = window_data['low'].tail(consolidation_periods).min()
            recent_mean = window_data['close'].tail(consolidation_periods).mean()
            recent_range = (recent_high - recent_low) / recent_mean if recent_mean != 0 else 0

            if (breakout_up or breakout_down) and volume_spike > self.thresholds['breakout']['volume_spike'] and recent_range < 0.04:
                subtype = "BULL_BREAKOUT" if breakout_up else "BEAR_BREAKOUT"
                momentum_conf = abs(momentum_5) / current_price * 100 if current_price != 0 else 0
                score = min(1.0, (volume_spike / 2.0) * 0.7 + (momentum_conf / 3.0) * 0.3)
            else:
                subtype = None
                score = 0.0

            return {
                'score': score,
                'subtype': subtype,
                'metrics': {
                    'breakout_up': breakout_up,
                    'breakout_down': breakout_down,
                    'volume_spike': volume_spike,
                    'recent_range': recent_range,
                    'momentum_5': momentum_5,
                    'momentum_10': momentum_10
                }
            }

        except Exception as e:
            logger.error(f"Error in breakout regime classification: {e}")
            return {'score': 0.0, 'subtype': None, 'metrics': {}}

    def _determine_primary_regime(self, results: Dict, price_change: float) -> Tuple[str, float, Optional[str]]:
        """Determine primary regime with proper hierarchy"""
        try:
            abs_price_change = abs(price_change)

            # 1. BREAKOUT priority
            if results['breakout']['score'] > 0.7:
                return 'BREAKOUT', results['breakout']['score'], results['breakout']['subtype']

            # 2. TRENDING priority
            if abs_price_change > self.thresholds['trend']['weak_change'] and results['trending']['score'] > 0.45:
                return 'TRENDING', results['trending']['score'], results['trending']['subtype']

            # 3. VOLATILE
            if results['volatile']['score'] > 0.6:
                return 'VOLATILE', results['volatile']['score'], results['volatile']['subtype']

            # 4. RANGE (including setups)
            if abs_price_change < self.thresholds['range']['max_directional_move'] and results['range']['score'] > 0.6:
                return 'RANGE', results['range']['score'], results['range']['subtype']

            # 5. Fallback
            scores = {
                'TRENDING': results['trending']['score'],
                'RANGE': results['range']['score'],
                'VOLATILE': results['volatile']['score'],
                'BREAKOUT': results['breakout']['score']
            }
            primary_regime = max(scores, key=scores.get)
            primary_score = scores[primary_regime]
            
            logger.warning(f"Regime determination fell back to highest score: {primary_regime} ({primary_score:.2f})")
            
            subtype = results[primary_regime.lower()]['subtype']
            return primary_regime, primary_score, subtype

        except Exception as e:
            logger.error(f"Error determining primary regime: {e}")
            return "UNKNOWN", 0.0, None

    def _calculate_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features required for regime analysis"""
        try:
            features_df = calculate_regime_features(df)

            required_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
                'adx', 'di_plus', 'di_minus', 'momentum_5', 'momentum_10'
            ]

            for col in required_cols:
                if col not in features_df.columns:
                    if col in ['rsi', 'adx', 'di_plus', 'di_minus']:
                        features_df[col] = 50.0
                    elif col in ['momentum_5', 'momentum_10']:
                        features_df[col] = 0.0
                    elif 'bollinger' in col:
                        features_df[col] = features_df['close']
                    logger.warning(f"Missing feature {col}, using fallback value")

            return features_df

        except Exception as e:
            logger.error(f"Error calculating analysis features: {e}")
            return pd.DataFrame({'close': df['close']})

    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """Validate input data requirements"""
        if df is None or df.empty:
            logger.warning("Input DataFrame is None or empty")
            return False

        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data points: {len(df)} < {self.min_data_points}")
            return False

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

        if df[['open', 'high', 'low', 'close']].isna().any().any():
            logger.warning("NaN values found in OHLC data")
            return False

        return True

    def _extract_key_metrics(self, window_data: pd.DataFrame, price_change: float) -> Dict:
        """Extract key metrics for result reporting"""
        try:
            metrics = {}
            latest = window_data.iloc[-1] if not window_data.empty else pd.Series()

            metrics['price_change_window'] = price_change
            metrics['price_current'] = latest.get('close', 0)
            metrics['rsi'] = latest.get('rsi', 50)
            metrics['adx'] = latest.get('adx', 20)
            
            bb_upper = latest.get('bollinger_upper', 0)
            bb_lower = latest.get('bollinger_lower', 0)
            bb_middle = latest.get('bollinger_middle', 1)
            metrics['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0

            vol_periods = min(24, len(window_data))
            metrics['volatility'] = window_data['close'].pct_change().tail(vol_periods).std()

            if 'volume' in window_data.columns:
                vol_ma_periods = min(20, len(window_data))
                metrics['volume_ma'] = window_data['volume'].tail(vol_ma_periods).mean()
                metrics['volume_current'] = latest.get('volume', 0)

            return metrics

        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {}

    def _log_regime_classification(self, result: Dict, symbol: str):
        """Log regime classification results with setup detection"""
        try:
            regime = result['primary_regime']
            subtype = result['subtype']
            confidence = result['confidence']
            metadata = result['metadata']

            logger.info(f"=" * 80)
            logger.info(f"{symbol} REGIME CLASSIFICATION")
            logger.info(f"Timeframe: {metadata['timeframe_minutes']}min | Window: {metadata['calculation_window']} candles ({self.target_hours}h)")
            logger.info(f"Price Change: {metadata['price_change_pct']:+.2%}")
            logger.info(f"-" * 80)
            logger.info(f"PRIMARY: {regime} | Subtype: {subtype} | Confidence: {confidence:.2%}")
            
            # Highlight setup detection
            if subtype in ['OVERSOLD_SETUP', 'OVERBOUGHT_SETUP']:
                logger.info(f"🎯 TRADING SETUP DETECTED: {subtype} - Mean reversion opportunity")
            
            logger.info(f"Scores: T:{result['regime_scores']['TRENDING']:.2f} R:{result['regime_scores']['RANGE']:.2f} V:{result['regime_scores']['VOLATILE']:.2f} B:{result['regime_scores']['BREAKOUT']:.2f}")
            
            metrics = result['metrics']
            logger.info(f"Metrics: RSI:{metrics.get('rsi', 50):.1f} ADX:{metrics.get('adx', 20):.1f} BBw:{metrics.get('bb_width', 0):.2%} Vol:{metrics.get('volatility', 0):.4f}")
            logger.info(f"=" * 80)

        except Exception as e:
            logger.error(f"Error logging regime classification: {e}")

    def _create_error_result(self, error_type: str) -> Dict:
        """Create error result for failed classifications"""
        return {
            'primary_regime': 'ERROR',
            'subtype': None,
            'confidence': 0.0,
            'regime_scores': {'TRENDING': 0.0, 'RANGE': 0.0, 'VOLATILE': 0.0, 'BREAKOUT': 0.0},
            'metrics': {},
            'metadata': {
                'error_type': error_type,
                'timestamp': pd.Timestamp.now()
            }
        }


# Legacy compatibility functions
def clasificar_regimen_mejorado(datos_mercado, symbol="BTCUSDT"):
    """Legacy function for backward compatibility"""
    try:
        classifier = MarketRegimeClassifier()
        df = datos_mercado.get(symbol)

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(f"No valid data for {symbol} in legacy function")
            return "neutral"

        result = classifier.classify_market_regime(df, symbol)

        regime_mapping = {
            'TRENDING': 'bull' if 'BULL' in str(result['subtype']) else 'bear',
            'RANGE': 'range',
            'VOLATILE': 'volatile',
            'BREAKOUT': 'breakout',
            'ERROR': 'neutral'
        }

        return regime_mapping.get(result['primary_regime'], 'neutral')

    except Exception as e:
        logger.error(f"Error in legacy regime classification: {e}")
        return "neutral"


def ejecutar_estrategia_por_regimen(datos_mercado, symbol="BTCUSDT"):
    """Enhanced strategy execution with setup detection"""
    try:
        classifier = MarketRegimeClassifier()
        df = datos_mercado.get(symbol)

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return _create_fallback_strategy()

        result = classifier.classify_market_regime(df, symbol)
        strategy_result = _generate_strategy_from_regime(result)

        return strategy_result

    except Exception as e:
        logger.error(f"Error ejecutando estrategia por régimen: {e}")
        return _create_fallback_strategy()


def _generate_strategy_from_regime(regime_result: Dict) -> Dict:
    """Generate trading strategy based on regime classification with setup handling"""
    regime = regime_result['primary_regime']
    subtype = regime_result['subtype']
    confidence = regime_result['confidence']

    base_strategy = {
        'regime': regime,
        'subtype': subtype,
        'confidence': confidence,
        'strategy_type': 'regime_adaptive'
    }

    # NEW: Handle setups within range regimes
    if subtype == 'OVERSOLD_SETUP':
        base_strategy.update({
            'signal': 'buy',
            'strategy_type': 'mean_reversion_oversold',
            'profit_target': 0.015,
            'stop_loss': 0.008,
            'max_position_time': 4,
            'setup_type': 'oversold',
            'allow_l2_signal': True
        })
        return base_strategy

    elif subtype == 'OVERBOUGHT_SETUP':
        base_strategy.update({
            'signal': 'sell',
            'strategy_type': 'mean_reversion_overbought',
            'profit_target': 0.015,
            'stop_loss': 0.008,
            'max_position_time': 4,
            'setup_type': 'overbought',
            'allow_l2_signal': True
        })
        return base_strategy

    # Standard regime strategies
    if regime == 'TRENDING':
        if 'BULL' in str(subtype):
            base_strategy.update({
                'signal': 'buy',
                'profit_target': 0.04 if 'STRONG' in subtype else 0.025 if 'MODERATE' in subtype else 0.015,
                'stop_loss': 0.02 if 'STRONG' in subtype else 0.015,
                'max_position_time': 12 if 'STRONG' in subtype else 8
            })
        elif 'BEAR' in str(subtype):
            base_strategy.update({
                'signal': 'sell',
                'profit_target': 0.04 if 'STRONG' in subtype else 0.025 if 'MODERATE' in subtype else 0.015,
                'stop_loss': 0.02 if 'STRONG' in subtype else 0.015,
                'max_position_time': 12 if 'STRONG' in subtype else 8
            })

    elif regime == 'RANGE':
        if subtype == 'TIGHT_RANGE':
            base_strategy.update({
                'signal': 'hold',
                'strategy_type': 'mean_reversion_ready',
                'profit_target': 0.008,
                'stop_loss': 0.006
            })
        else:
            base_strategy.update({
                'signal': 'hold',
                'profit_target': 0.012,
                'stop_loss': 0.01
            })

    elif regime == 'VOLATILE':
        base_strategy.update({
            'signal': 'hold',
            'strategy_type': 'volatility_avoidance',
            'profit_target': 0.03,
            'stop_loss': 0.025
        })

    elif regime == 'BREAKOUT':
        if 'BULL' in str(subtype):
            base_strategy.update({
                'signal': 'buy',
                'strategy_type': 'breakout_momentum',
                'profit_target': 0.05,
                'stop_loss': 0.025,
                'max_position_time': 8
            })
        elif 'BEAR' in str(subtype):
            base_strategy.update({
                'signal': 'sell',
                'strategy_type': 'breakout_momentum',
                'profit_target': 0.05,
                'stop_loss': 0.025,
                'max_position_time': 8
            })
        else:
            base_strategy.update({
                'signal': 'hold',
                'strategy_type': 'breakout_waiting'
            })

    elif regime == 'ERROR' or regime == 'UNKNOWN':
        base_strategy.update({
            'signal': 'hold',
            'strategy_type': 'classification_error'
        })

    else:
        # Default fallback for any unhandled regime
        base_strategy.update({
            'signal': 'hold',
            'strategy_type': 'default_hold',
            'allow_l2_signal': False
        })

    return base_strategy


def _create_fallback_strategy() -> Dict:
    """Create fallback strategy for error cases"""
    return {
        'regime': 'ERROR',
        'subtype': None,
        'confidence': 0.0,
        'signal': 'hold',
        'strategy_type': 'classification_error',
        'profit_target': 0.0,
        'stop_loss': 0.0,
        'max_position_time': 0,
        'allow_l2_signal': False
    }
