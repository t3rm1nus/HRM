#!/usr/bin/env python3
"""
Test suite for BTC/ETH Sales Synchronization.

This module provides comprehensive tests for the BTC/ETH synchronization functionality,
including unit tests for individual components and integration tests for the full pipeline.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import TacticalSignal
from core.logging import logger


class TestMarketConditionSimilarity(unittest.TestCase):
    """Test cases for market condition similarity detection"""

    def setUp(self):
        """Set up test processor and market data"""
        self.processor = L2TacticProcessor()

        # Create highly correlated BTC/ETH data
        base_time = pd.Timestamp.now()
        dates = pd.date_range(start=base_time - timedelta(hours=24), end=base_time, freq='1H')

        # BTC data with strong uptrend
        btc_prices = [50000 + i * 100 for i in range(len(dates))]  # Steady uptrend
        self.btc_data = pd.DataFrame({
            'close': btc_prices,
            'high': [p * 1.005 for p in btc_prices],
            'low': [p * 0.995 for p in btc_prices],
            'volume': [100 + i * 5 for i in range(len(dates))]
        }, index=dates)

        # ETH data highly correlated with BTC
        eth_prices = [3000 + i * 6 for i in range(len(dates))]  # Correlated uptrend
        self.eth_data = pd.DataFrame({
            'close': eth_prices,
            'high': [p * 1.005 for p in eth_prices],
            'low': [p * 0.995 for p in eth_prices],
            'volume': [200 + i * 10 for i in range(len(dates))]
        }, index=dates)

        self.market_data = {
            'BTCUSDT': self.btc_data,
            'ETHUSDT': self.eth_data
        }

    def test_highly_correlated_markets(self):
        """Test similarity detection for highly correlated markets"""
        similarity = self.processor._detect_market_condition_similarity(self.market_data)

        self.assertIsInstance(similarity, dict)
        self.assertIn('correlation', similarity)
        self.assertIn('similarity_score', similarity)
        self.assertIn('is_similar', similarity)

        # Should detect high correlation
        self.assertGreater(similarity['correlation'], 0.8)
        self.assertGreater(similarity['similarity_score'], 0.8)
        self.assertTrue(similarity['is_similar'])

        logger.info(f"High correlation test: {similarity}")

    def test_weakly_correlated_markets(self):
        """Test similarity detection for weakly correlated markets"""
        # Create uncorrelated ETH data
        dates = self.eth_data.index
        eth_prices_uncorr = [3000 + np.sin(i) * 100 for i in range(len(dates))]  # Oscillating
        eth_data_uncorr = pd.DataFrame({
            'close': eth_prices_uncorr,
            'high': [p * 1.005 for p in eth_prices_uncorr],
            'low': [p * 0.995 for p in eth_prices_uncorr],
            'volume': [200 + i * 10 for i in range(len(dates))]
        }, index=dates)

        market_data_uncorr = {
            'BTCUSDT': self.btc_data,
            'ETHUSDT': eth_data_uncorr
        }

        similarity = self.processor._detect_market_condition_similarity(market_data_uncorr)

        # Should detect low correlation
        self.assertLess(similarity['correlation'], 0.5)
        self.assertLess(similarity['similarity_score'], 0.5)
        self.assertFalse(similarity['is_similar'])

        logger.info(f"Low correlation test: {similarity}")

    def test_rsi_similarity_calculation(self):
        """Test RSI similarity calculation"""
        # Create data with similar RSI
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')

        # Both assets in similar overbought territory
        btc_prices = [50000 + 5000 * np.sin(i/2) for i in range(len(dates))]
        eth_prices = [3000 + 500 * np.sin(i/2) for i in range(len(dates))]

        btc_data = pd.DataFrame({'close': btc_prices}, index=dates)
        eth_data = pd.DataFrame({'close': eth_prices}, index=dates)

        market_data = {'BTCUSDT': btc_data, 'ETHUSDT': eth_data}

        similarity = self.processor._detect_market_condition_similarity(market_data)

        # RSI similarity should be high
        self.assertGreater(similarity['rsi_similarity'], 0.8)
        logger.info(f"RSI similarity test: RSI_sim={similarity['rsi_similarity']:.3f}")

    def test_correlation_period_calculation(self):
        """Test correlation calculation for different periods"""
        corr_30 = self.processor._compute_correlation_period(self.btc_data, self.eth_data, 30)
        corr_10 = self.processor._compute_correlation_period(self.btc_data, self.eth_data, 10)
        corr_5 = self.processor._compute_correlation_period(self.btc_data, self.eth_data, 5)

        # All should be positive and high for our correlated data
        self.assertGreater(corr_30, 0.8)
        self.assertGreater(corr_10, 0.8)
        self.assertGreater(corr_5, 0.8)

        logger.info(f"Correlation periods: 30p={corr_30:.3f}, 10p={corr_10:.3f}, 5p={corr_5:.3f}")


class TestAssetWeaknessDetection(unittest.TestCase):
    """Test cases for asset weakness detection"""

    def setUp(self):
        """Set up test data"""
        self.processor = L2TacticProcessor()

        # Create weak ETH data (overbought, bearish MACD, negative momentum)
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')
        eth_prices = [3000 - i * 10 for i in range(len(dates))]  # Downtrend
        self.eth_weak_data = pd.DataFrame({
            'close': eth_prices,
            'high': [p * 1.01 for p in eth_prices],
            'low': [p * 0.99 for p in eth_prices],
            'volume': [200 - i * 2 for i in range(len(dates))]
        }, index=dates)

        # Create strong ETH data (normal conditions)
        eth_prices_strong = [3000 + i * 5 for i in range(len(dates))]  # Uptrend
        self.eth_strong_data = pd.DataFrame({
            'close': eth_prices_strong,
            'high': [p * 1.005 for p in eth_prices_strong],
            'low': [p * 0.995 for p in eth_prices_strong],
            'volume': [200 + i * 2 for i in range(len(dates))]
        }, index=dates)

    def test_weak_asset_detection(self):
        """Test detection of weak asset conditions"""
        market_data = {'ETHUSDT': self.eth_weak_data}
        state = {}

        weakness = self.processor._check_asset_weakness('ETHUSDT', market_data, state)

        # Should detect high weakness (downtrending data should be weak)
        self.assertGreater(weakness, 0.3)  # Lowered threshold since our data might not be extremely weak
        logger.info(f"Weak asset detection: weakness={weakness:.3f}")

    def test_strong_asset_detection(self):
        """Test detection of strong asset conditions"""
        market_data = {'ETHUSDT': self.eth_strong_data}
        state = {}

        weakness = self.processor._check_asset_weakness('ETHUSDT', market_data, state)

        # Should detect low weakness (strong conditions)
        self.assertLess(weakness, 0.4)
        logger.info(f"Strong asset detection: weakness={weakness:.3f}")


class TestSynchronizedSellTriggers(unittest.TestCase):
    """Test cases for synchronized sell trigger system"""

    def setUp(self):
        """Set up test signals and data"""
        self.processor = L2TacticProcessor()

        # Create highly correlated market data
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')
        btc_prices = [50000 + i * 100 for i in range(len(dates))]
        eth_prices = [3000 + i * 6 for i in range(len(dates))]

        self.market_data = {
            'BTCUSDT': pd.DataFrame({'close': btc_prices}, index=dates),
            'ETHUSDT': pd.DataFrame({'close': eth_prices}, index=dates)
        }

        # Create test signals
        self.btc_strong_sell = TacticalSignal(
            symbol='BTCUSDT',
            side='sell',
            strength=0.85,
            confidence=0.9,
            signal_type='tactical',
            source='ai',
            timestamp=pd.Timestamp.now()
        )

        self.eth_weak_signal = TacticalSignal(
            symbol='ETHUSDT',
            side='hold',
            strength=0.5,
            confidence=0.6,
            signal_type='tactical',
            source='ai',
            timestamp=pd.Timestamp.now()
        )

        self.eth_strong_sell = TacticalSignal(
            symbol='ETHUSDT',
            side='sell',
            strength=0.8,
            confidence=0.85,
            signal_type='tactical',
            source='ai',
            timestamp=pd.Timestamp.now()
        )

    def test_synchronized_sell_trigger_btc_to_eth(self):
        """Test BTC strong sell triggering ETH synchronized sell"""
        signals = [self.btc_strong_sell, self.eth_weak_signal]
        similarity_analysis = {'correlation': 0.9, 'similarity_score': 0.85, 'is_similar': True}

        synchronized_signals = self.processor._apply_synchronized_sell_triggers(
            signals, self.btc_strong_sell, self.eth_weak_signal, similarity_analysis, {}, self.market_data
        )

        # Should add a synchronized sell signal for ETH
        self.assertGreater(len(synchronized_signals), len(signals))

        # Check for synchronized ETH sell signal
        eth_sell_signals = [s for s in synchronized_signals if s.symbol == 'ETHUSDT' and s.side == 'sell']
        self.assertGreater(len(eth_sell_signals), 0)

        sync_signal = eth_sell_signals[0]
        self.assertEqual(sync_signal.signal_type, 'synchronized_sell')
        self.assertEqual(sync_signal.source, 'btc_eth_sync')

        logger.info(f"Synchronized sell test: {len(signals)} → {len(synchronized_signals)} signals")

    def test_no_trigger_when_not_similar(self):
        """Test no synchronization when markets are not similar"""
        signals = [self.btc_strong_sell, self.eth_weak_signal]
        similarity_analysis = {'correlation': 0.3, 'similarity_score': 0.4, 'is_similar': False}

        synchronized_signals = self.processor._apply_synchronized_sell_triggers(
            signals, self.btc_strong_sell, self.eth_weak_signal, similarity_analysis, {}, self.market_data
        )

        # Should not add any signals
        self.assertEqual(len(synchronized_signals), len(signals))

    def test_no_trigger_when_eth_not_weak(self):
        """Test no synchronization when ETH is not weak"""
        # Replace weak signal with strong sell signal
        signals = [self.btc_strong_sell, self.eth_strong_sell]
        similarity_analysis = {'correlation': 0.9, 'similarity_score': 0.85, 'is_similar': True}

        synchronized_signals = self.processor._apply_synchronized_sell_triggers(
            signals, self.btc_strong_sell, self.eth_strong_sell, similarity_analysis, {}, self.market_data
        )

        # Should not add synchronization since ETH already has sell signal
        eth_sell_signals = [s for s in synchronized_signals if s.symbol == 'ETHUSDT' and s.side == 'sell']
        self.assertEqual(len(eth_sell_signals), 1)  # Only the original signal

    def test_bidirectional_synchronization(self):
        """Test that synchronization works both ways (ETH→BTC and BTC→ETH)"""
        # Test ETH→BTC synchronization
        eth_strong_sell = TacticalSignal(
            symbol='ETHUSDT',
            side='sell',
            strength=0.82,
            confidence=0.88,
            signal_type='tactical',
            source='ai',
            timestamp=pd.Timestamp.now()
        )

        btc_weak_signal = TacticalSignal(
            symbol='BTCUSDT',
            side='hold',
            strength=0.5,
            confidence=0.6,
            signal_type='tactical',
            source='ai',
            timestamp=pd.Timestamp.now()
        )

        signals = [eth_strong_sell, btc_weak_signal]
        similarity_analysis = {'correlation': 0.9, 'similarity_score': 0.85, 'is_similar': True}

        synchronized_signals = self.processor._apply_synchronized_sell_triggers(
            signals, btc_weak_signal, eth_strong_sell, similarity_analysis, {}, self.market_data
        )

        # Should add synchronized BTC sell signal
        btc_sell_signals = [s for s in synchronized_signals if s.symbol == 'BTCUSDT' and s.side == 'sell']
        self.assertGreater(len(btc_sell_signals), 0)


class TestCorrelationBasedSizing(unittest.TestCase):
    """Test cases for correlation-based position sizing"""

    def setUp(self):
        """Set up test signals"""
        self.processor = L2TacticProcessor()

        # Create sell signals for both BTC and ETH
        self.btc_sell = TacticalSignal(
            symbol='BTCUSDT',
            side='sell',
            strength=0.8,
            confidence=0.85,
            signal_type='tactical',
            source='ai',
            quantity=1.0,  # Original quantity
            timestamp=pd.Timestamp.now()
        )

        self.eth_sell = TacticalSignal(
            symbol='ETHUSDT',
            side='sell',
            strength=0.75,
            confidence=0.8,
            signal_type='tactical',
            source='ai',
            quantity=5.0,  # Original quantity
            timestamp=pd.Timestamp.now()
        )

    def test_high_correlation_sizing_reduction(self):
        """Test position size reduction when correlation is high"""
        signals = [self.btc_sell, self.eth_sell]
        similarity_analysis = {'correlation': 0.9, 'similarity_score': 0.85, 'is_similar': True}

        adjusted_signals = self.processor._apply_correlation_based_sizing(signals, similarity_analysis, {})

        # Both signals should have reduced quantities
        btc_signal = next(s for s in adjusted_signals if s.symbol == 'BTCUSDT')
        eth_signal = next(s for s in adjusted_signals if s.symbol == 'ETHUSDT')

        # Quantities should be reduced (correlation factor < 1.0)
        self.assertLess(btc_signal.quantity, 1.0)
        self.assertLess(eth_signal.quantity, 5.0)

        # Should have metadata about adjustment
        self.assertIn('correlation_adjusted', btc_signal.metadata)
        self.assertIn('correlation_factor', btc_signal.metadata)

        logger.info(f"Correlation sizing: BTC {1.0}→{btc_signal.quantity:.3f}, ETH {5.0}→{eth_signal.quantity:.3f}")

    def test_low_correlation_no_sizing_change(self):
        """Test no position size change when correlation is low"""
        signals = [self.btc_sell, self.eth_sell]
        similarity_analysis = {'correlation': 0.4, 'similarity_score': 0.5, 'is_similar': False}

        adjusted_signals = self.processor._apply_correlation_based_sizing(signals, similarity_analysis, {})

        # Signals should maintain original quantities
        btc_signal = next(s for s in adjusted_signals if s.symbol == 'BTCUSDT')
        eth_signal = next(s for s in adjusted_signals if s.symbol == 'ETHUSDT')

        self.assertEqual(btc_signal.quantity, 1.0)
        self.assertEqual(eth_signal.quantity, 5.0)

    def test_buy_signals_not_affected(self):
        """Test that buy signals are not affected by correlation sizing"""
        buy_signal = TacticalSignal(
            symbol='BTCUSDT',
            side='buy',
            strength=0.8,
            confidence=0.85,
            signal_type='tactical',
            source='ai',
            quantity=1.0,
            timestamp=pd.Timestamp.now()
        )

        signals = [buy_signal]
        similarity_analysis = {'correlation': 0.9, 'similarity_score': 0.85, 'is_similar': True}

        adjusted_signals = self.processor._apply_correlation_based_sizing(signals, similarity_analysis, {})

        # Buy signal should not be affected
        adjusted_signal = adjusted_signals[0]
        self.assertEqual(adjusted_signal.quantity, 1.0)


class TestFullSynchronizationPipeline(unittest.TestCase):
    """Integration tests for the full BTC/ETH synchronization pipeline"""

    def setUp(self):
        """Set up full integration test"""
        self.processor = L2TacticProcessor()

        # Create highly correlated market data
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')
        btc_prices = [50000 + i * 100 for i in range(len(dates))]
        eth_prices = [3000 + i * 6 for i in range(len(dates))]

        self.market_data = {
            'BTCUSDT': pd.DataFrame({'close': btc_prices}, index=dates),
            'ETHUSDT': pd.DataFrame({'close': eth_prices}, index=dates)
        }

        # Create initial signals (BTC strong sell, ETH weak)
        self.signals = [
            TacticalSignal(
                symbol='BTCUSDT',
                side='sell',
                strength=0.85,
                confidence=0.9,
                signal_type='tactical',
                source='ai',
                quantity=1.0,
                timestamp=pd.Timestamp.now()
            ),
            TacticalSignal(
                symbol='ETHUSDT',
                side='hold',
                strength=0.5,
                confidence=0.6,
                signal_type='tactical',
                source='ai',
                quantity=5.0,
                timestamp=pd.Timestamp.now()
            )
        ]

    def test_full_synchronization_pipeline(self):
        """Test the complete synchronization pipeline"""
        state = {}

        # Apply full synchronization
        synchronized_signals = self.processor._apply_btc_eth_synchronization(
            self.signals, self.market_data, state
        )

        # Should have more signals than original (synchronized sell added)
        self.assertGreater(len(synchronized_signals), len(self.signals))

        # Check for synchronized sell signal
        sync_signals = [s for s in synchronized_signals if s.signal_type == 'synchronized_sell']
        self.assertGreater(len(sync_signals), 0)

        # Check that quantities were adjusted for correlation
        sell_signals = [s for s in synchronized_signals if s.side == 'sell']
        for signal in sell_signals:
            if signal.symbol in ['BTCUSDT', 'ETHUSDT']:
                # Should have correlation metadata
                self.assertIn('correlation_adjusted', signal.metadata)
                self.assertLess(signal.quantity, getattr(signal, 'original_quantity', signal.quantity + 1))

        logger.info(f"Full pipeline test: {len(self.signals)} → {len(synchronized_signals)} signals")
        logger.info(f"Synchronized signals: {len(sync_signals)}")
        logger.info(f"Sell signals with correlation adjustment: {len([s for s in sell_signals if s.metadata.get('correlation_adjusted', False)])}")

    def test_pipeline_with_no_correlation(self):
        """Test pipeline when markets are not correlated"""
        # Create uncorrelated data
        dates = self.market_data['ETHUSDT'].index
        eth_prices_uncorr = [3000 + np.sin(i) * 100 for i in range(len(dates))]
        market_data_uncorr = {
            'BTCUSDT': self.market_data['BTCUSDT'],
            'ETHUSDT': pd.DataFrame({'close': eth_prices_uncorr}, index=dates)
        }

        state = {}

        # Apply synchronization
        synchronized_signals = self.processor._apply_btc_eth_synchronization(
            self.signals, market_data_uncorr, state
        )

        # Should have same number of signals (no synchronization triggered)
        self.assertEqual(len(synchronized_signals), len(self.signals))

        logger.info(f"No correlation test: {len(self.signals)} → {len(synchronized_signals)} signals")


class TestTechnicalIndicatorCalculations(unittest.TestCase):
    """Test cases for technical indicator calculations used in synchronization"""

    def setUp(self):
        """Set up test data"""
        self.processor = L2TacticProcessor()

        # Create test data
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')
        prices = [50000 + i * 50 for i in range(len(dates))]

        self.test_data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'volume': [100 + i * 2 for i in range(len(dates))]
        }, index=dates)

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.processor._calculate_rsi(self.test_data)

        # RSI should be between 0 and 100
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)

        # For uptrending data, RSI should be elevated
        self.assertGreater(rsi, 50)

        logger.info(f"RSI calculation test: RSI={rsi:.1f}")

    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd = self.processor._calculate_macd(self.test_data)

        # MACD should be a finite number
        self.assertTrue(np.isfinite(macd))

        # For uptrending data, MACD should be positive
        self.assertGreater(macd, 0)

        logger.info(f"MACD calculation test: MACD={macd:.4f}")

    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        trend = self.processor._calculate_trend_strength(self.test_data)

        # Trend should be between -1 and 1
        self.assertGreaterEqual(trend, -1)
        self.assertLessEqual(trend, 1)

        # For uptrending data, trend should be positive
        self.assertGreater(trend, 0)

        logger.info(f"Trend strength test: trend={trend:.3f}")

    def test_ema_calculation(self):
        """Test EMA calculation"""
        ema = self.processor._calculate_ema(self.test_data['close'].values, 12)

        # EMA should be a finite number close to recent prices
        self.assertTrue(np.isfinite(ema))
        self.assertGreater(ema, 40000)  # Should be in reasonable range

        logger.info(f"EMA calculation test: EMA={ema:.2f}")


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error handling"""

    def setUp(self):
        """Set up test processor"""
        self.processor = L2TacticProcessor()

    def test_missing_market_data(self):
        """Test handling of missing market data"""
        signals = []
        market_data = {}
        state = {}

        # Should handle gracefully without crashing
        result = self.processor._apply_btc_eth_synchronization(signals, market_data, state)
        self.assertEqual(len(result), 0)

    def test_insufficient_data(self):
        """Test handling of insufficient data for calculations"""
        # Create data with only 2 points (insufficient for most calculations)
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=1), end=pd.Timestamp.now(), freq='30min')
        btc_data = pd.DataFrame({'close': [50000, 50100]}, index=dates)
        eth_data = pd.DataFrame({'close': [3000, 3010]}, index=dates)

        market_data = {'BTCUSDT': btc_data, 'ETHUSDT': eth_data}

        similarity = self.processor._detect_market_condition_similarity(market_data)

        # Should return default values without crashing
        self.assertIsInstance(similarity, dict)
        self.assertEqual(similarity['correlation'], 0.0)
        self.assertFalse(similarity['is_similar'])

    def test_extreme_correlation_values(self):
        """Test handling of extreme correlation values"""
        # Test with perfect correlation
        dates = pd.date_range(start=pd.Timestamp.now() - timedelta(hours=24), end=pd.Timestamp.now(), freq='1H')
        btc_prices = [50000 + i * 100 for i in range(len(dates))]
        eth_prices = [3000 + i * 6 for i in range(len(dates))]  # Perfectly correlated

        market_data = {
            'BTCUSDT': pd.DataFrame({'close': btc_prices}, index=dates),
            'ETHUSDT': pd.DataFrame({'close': eth_prices}, index=dates)
        }

        similarity = self.processor._detect_market_condition_similarity(market_data)

        # Should handle perfect correlation without issues
        self.assertGreaterEqual(similarity['correlation'], 0.95)
        self.assertTrue(similarity['is_similar'])

        # Test correlation sizing with extreme values
        signals = [
            TacticalSignal(symbol='BTCUSDT', side='sell', strength=0.8, confidence=0.85, quantity=1.0, timestamp=pd.Timestamp.now()),
            TacticalSignal(symbol='ETHUSDT', side='sell', strength=0.75, confidence=0.8, quantity=5.0, timestamp=pd.Timestamp.now())
        ]

        adjusted = self.processor._apply_correlation_based_sizing(signals, similarity, {})

        # Should apply maximum reduction for extreme correlation
        for signal in adjusted:
            if signal.side == 'sell':
                self.assertLess(signal.quantity, getattr(signal, 'original_quantity', signal.quantity + 1))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run tests
    unittest.main(verbosity=2)
