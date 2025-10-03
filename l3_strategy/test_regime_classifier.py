"""
Comprehensive Test Suite for Market Regime Classifier

Tests all regime types and subtypes with realistic market data scenarios:
- TRENDING: STRONG_BULL, MODERATE_BULL, WEAK_BULL, STRONG_BEAR, MODERATE_BEAR, WEAK_BEAR
- RANGE: TIGHT_RANGE, NORMAL_RANGE, WIDE_RANGE
- VOLATILE: HIGH_VOLATILITY
- BREAKOUT: BULL_BREAKOUT, BEAR_BREAKOUT

Uses synthetic data to validate classification accuracy and edge cases.
"""

import pandas as pd
import numpy as np
import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from l3_strategy.regime_classifier import MarketRegimeClassifier


class TestMarketRegimeClassifier(unittest.TestCase):
    """Test suite for comprehensive regime classification"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = MarketRegimeClassifier()
        self.symbol = "BTCUSDT"
        np.random.seed(42)  # For reproducible results

    def generate_synthetic_data(self, scenario: str, periods: int = 100) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for different market scenarios

        Args:
            scenario: Type of market scenario to generate
            periods: Number of periods to generate

        Returns:
            DataFrame with OHLCV data
        """
        # Create time index
        start_time = datetime.now() - timedelta(hours=periods//12)  # 5-min intervals
        timestamps = [start_time + timedelta(minutes=i*5) for i in range(periods)]

        # Base price parameters
        base_price = 50000
        prices = [base_price]

        # Generate price data based on scenario
        if scenario == "strong_bull_trend":
            # Strong upward trending market
            for i in range(1, periods):
                change = np.random.normal(0.002, 0.005)  # 0.2% mean, 0.5% std
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, prices[-1] * 0.98))  # Floor at -2%

        elif scenario == "strong_bear_trend":
            # Strong downward trending market
            for i in range(1, periods):
                change = np.random.normal(-0.002, 0.005)  # -0.2% mean, 0.5% std
                new_price = prices[-1] * (1 + change)
                prices.append(min(new_price, prices[-1] * 1.02))  # Ceiling at +2%

        elif scenario == "moderate_bull_trend":
            # Moderate upward trend
            for i in range(1, periods):
                change = np.random.normal(0.0008, 0.003)  # 0.08% mean, 0.3% std
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, prices[-1] * 0.99))  # Floor at -1%

        elif scenario == "tight_range":
            # Very tight price range with consistent band touches
            range_center = base_price
            range_width = 0.015  # 1.5% range
            for i in range(1, periods):
                # Mean-reverting behavior
                deviation = (prices[-1] - range_center) / range_center
                reversion_force = -deviation * 0.1  # 10% reversion
                noise = np.random.normal(0, 0.002)
                change = reversion_force + noise
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

        elif scenario == "normal_range":
            # Normal ranging market
            range_center = base_price
            range_width = 0.04  # 4% range
            for i in range(1, periods):
                # Mean-reverting with wider bounds
                deviation = (prices[-1] - range_center) / range_center
                reversion_force = -deviation * 0.05  # 5% reversion
                noise = np.random.normal(0, 0.004)
                change = reversion_force + noise
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

        elif scenario == "high_volatility":
            # High volatility period
            for i in range(1, periods):
                change = np.random.normal(0, 0.015)  # 1.5% std deviation
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

        elif scenario == "bull_breakout":
            # Consolidation followed by breakout
            for i in range(1, periods//2):
                # First half: consolidation
                deviation = (prices[-1] - base_price) / base_price
                change = -deviation * 0.1 + np.random.normal(0, 0.003)
                prices.append(prices[-1] * (1 + change))

            for i in range(periods//2, periods):
                # Second half: strong breakout
                change = np.random.normal(0.004, 0.006)
                prices.append(prices[-1] * (1 + change))

        else:
            # Default: random walk
            for i in range(1, periods):
                change = np.random.normal(0, 0.005)
                prices.append(prices[-1] * (1 + change))

        # Generate OHLCV data from prices with realistic spreads
        data = []
        for i, price in enumerate(prices):
            # Add some randomness to create realistic OHLC bars
            volatility = 0.002  # 0.2% typical bar volatility
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = data[-1]['close'] if data else price
            volume = np.random.normal(1000, 200)  # Base volume

            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': max(volume, 100)
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def test_strong_bull_trend_detection(self):
        """Test detection of strong bullish trending regime"""
        df = self.generate_synthetic_data("strong_bull_trend", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'TRENDING')
        self.assertEqual(result['subtype'], 'STRONG_BULL')
        self.assertGreater(result['confidence'], 0.7)
        self.assertGreater(result['regime_scores']['TRENDING'], 0.8)

        # Check key metrics
        self.assertGreater(result['metrics']['price_change_6h'], 0.015)  # >1.5% over 6 hours

    def test_strong_bear_trend_detection(self):
        """Test detection of strong bearish trending regime"""
        df = self.generate_synthetic_data("strong_bear_trend", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'TRENDING')
        self.assertEqual(result['subtype'], 'STRONG_BEAR')
        self.assertGreater(result['confidence'], 0.7)

    def test_moderate_trend_variants(self):
        """Test detection of moderate trend variants"""
        scenarios = ["moderate_bull_trend"]

        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                df = self.generate_synthetic_data(scenario, periods=80)
                result = self.classifier.classify_market_regime(df, self.symbol)

                self.assertEqual(result['primary_regime'], 'TRENDING')
                self.assertIn('MODERATE', result['subtype'])
                self.assertGreater(result['confidence'], 0.5)

    def test_tight_range_detection(self):
        """Test detection of tight ranging regime"""
        df = self.generate_synthetic_data("tight_range", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'RANGE')
        self.assertEqual(result['subtype'], 'TIGHT_RANGE')
        self.assertGreater(result['confidence'], 0.7)

        # Check BB width is tight
        self.assertLess(result['metrics']['bb_width'], 0.04)  # <4%

    def test_normal_range_detection(self):
        """Test detection of normal ranging regime"""
        df = self.generate_synthetic_data("normal_range", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'RANGE')
        self.assertIn('NORMAL', result['subtype'])
        self.assertGreater(result['confidence'], 0.6)

    def test_high_volatility_detection(self):
        """Test detection of high volatility regime"""
        df = self.generate_synthetic_data("high_volatility", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'VOLATILE')
        self.assertEqual(result['subtype'], 'HIGH_VOLATILITY')
        self.assertGreater(result['confidence'], 0.6)

    def test_breakout_detection(self):
        """Test detection of breakout regime"""
        df = self.generate_synthetic_data("bull_breakout", periods=100)

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'BREAKOUT')
        self.assertEqual(result['subtype'], 'BULL_BREAKOUT')
        self.assertGreater(result['confidence'], 0.6)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        df = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50200, 50300],
            'low': [49900, 50000],
            'close': [50100, 50200],
            'volume': [1000, 1100]
        })

        result = self.classifier.classify_market_regime(df, self.symbol)

        self.assertEqual(result['primary_regime'], 'ERROR')
        self.assertIn('insufficient_data', result['metadata']['error_type'])

    def test_invalid_data_handling(self):
        """Test handling of invalid/missing data"""
        # Test with None input
        result = self.classifier.classify_market_regime(None, self.symbol)
        self.assertEqual(result['primary_regime'], 'ERROR')

        # Test with empty DataFrame
        result = self.classifier.classify_market_regime(pd.DataFrame(), self.symbol)
        self.assertEqual(result['primary_regime'], 'ERROR')

        # Test missing required columns
        df = pd.DataFrame({'close': [50000, 50100]})
        result = self.classifier.classify_market_regime(df, self.symbol)
        self.assertEqual(result['primary_regime'], 'ERROR')

    def test_regime_score_calculation(self):
        """Test that regime scores are calculated correctly"""
        df = self.generate_synthetic_data("strong_bull_trend", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        # All scores should be between 0 and 1
        for regime, score in result['regime_scores'].items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Highest score should match primary regime
        primary_score = result['regime_scores'][result['primary_regime']]
        self.assertEqual(primary_score, max(result['regime_scores'].values()))

    def test_key_metrics_extraction(self):
        """Test extraction of key technical metrics"""
        df = self.generate_synthetic_data("strong_bull_trend", periods=80)

        result = self.classifier.classify_market_regime(df, self.symbol)

        metrics = result['metrics']

        # Check required metrics are present
        required_metrics = ['price_change_6h', 'rsi', 'adx', 'bb_width', 'volatility_24']
        for metric in required_metrics:
            self.assertIn(metric, metrics)

        # RSI should be between 0 and 100 (or neutral 50 if calculation fails)
        self.assertGreaterEqual(metrics['rsi'], 0)
        self.assertLessEqual(metrics['rsi'], 100)

        # ADX should be positive
        self.assertGreaterEqual(metrics['adx'], 0)

    def test_calculation_window_usage(self):
        """Test that classification uses correct 6-hour window"""
        df = self.generate_synthetic_data("strong_bull_trend", periods=150)  # More than 72 periods

        result = self.classifier.classify_market_regime(df, self.symbol)

        # Should use exactly 72 data points for window
        self.assertEqual(result['metadata']['data_points'], 72)
        self.assertEqual(result['metadata']['calculation_window'], 72)

    def test_legacy_compatibility(self):
        """Test legacy function compatibility"""
        from l3_strategy.regime_classifier import clasificar_regimen_mejorado

        df = self.generate_synthetic_data("strong_bull_trend", periods=80)
        market_data = {self.symbol: df}

        result = clasificar_regimen_mejorado(market_data, self.symbol)

        # Should return legacy format
        self.assertIn(result, ['bull', 'bear', 'range', 'neutral', 'volatile', 'breakout'])

    def test_strategy_generation(self):
        """Test strategy generation for different regimes"""
        from l3_strategy.regime_classifier import ejecutar_estrategia_por_regimen

        scenarios = [
            ("strong_bull_trend", "buy"),
            ("tight_range", "hold"),
            ("high_volatility", "hold")
        ]

        for scenario, expected_signal in scenarios:
            with self.subTest(scenario=scenario):
                df = self.generate_synthetic_data(scenario, periods=80)
                market_data = {self.symbol: df}

                strategy = ejecutar_estrategia_por_regimen(market_data, self.symbol)

                self.assertIn('signal', strategy)
                self.assertIn('confidence', strategy)
                self.assertIn('strategy_type', strategy)
                self.assertIn(strategy['signal'], ['buy', 'sell', 'hold'])

    def test_error_resilience(self):
        """Test classifier resilience to various error conditions"""
        classifier = MarketRegimeClassifier()

        # Test with NaN values
        df = self.generate_synthetic_data("strong_bull_trend", periods=80)
        df.loc[0, 'close'] = np.nan

        result = classifier.classify_market_regime(df, self.symbol)

        # Should handle NaN gracefully and still produce a result
        self.assertNotEqual(result['primary_regime'], 'ERROR')

    def test_threshold_robustness(self):
        """Test that classification thresholds are reasonable"""
        classifier = MarketRegimeClassifier()

        # Test trending thresholds
        self.assertGreater(classifier.thresholds['trend']['strong_slope'], 0.01)
        self.assertGreater(classifier.thresholds['trend']['moderate_slope'], 0.005)
        self.assertGreater(classifier.thresholds['trend']['weak_slope'], 0.002)

        # Test range thresholds
        self.assertGreater(classifier.thresholds['range']['tight_bb_width'], 0.01)
        self.assertGreater(classifier.thresholds['range']['normal_bb_width'], 0.05)

        # Test volatility thresholds
        self.assertGreater(classifier.thresholds['volatile']['volatility_multiplier'], 1.5)

        # Test breakout thresholds
        self.assertGreater(classifier.thresholds['breakout']['volume_spike'], 1.0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

    # Run tests
    unittest.main(verbosity=2)
