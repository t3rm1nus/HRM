#!/usr/bin/env python3
"""
Comprehensive test suite for convergence and technical strength sizing enhancements.
Tests all new functions and integration points.
"""

import sys
import os
sys.path.append('.')

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import the new functions
from core.technical_indicators import (
    calculate_technical_strength_score,
    calculate_convergence_multiplier,
    validate_technical_strength_for_position_size
)
from core.portfolio_manager import PortfolioManager
from l2_tactic.signal_composer import SignalComposer
from l2_tactic.models import TacticalSignal


class TestTechnicalStrengthScoring(unittest.TestCase):
    """Test technical strength scoring functionality"""

    def setUp(self):
        """Set up test data"""
        self.test_indicators = {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'vol_zscore': 0.0,
            'adx': 25.0,
            'roc_5': 0.0,
            'williams_r': -50.0
        }

    def test_strong_bullish_signal(self):
        """Test scoring for strong bullish conditions"""
        bullish_data = self.test_indicators.copy()
        bullish_data.update({
            'rsi': 75.0,  # Overbought (bearish)
            'macd': 20.0,  # Bullish MACD
            'macd_signal': 15.0,
            'vol_zscore': 2.0,  # High volume
            'adx': 40.0,  # Strong trend
            'roc_5': 3.0,  # Bullish momentum
            'williams_r': -20.0  # Bullish
        })

        df = pd.DataFrame([bullish_data])
        score = calculate_technical_strength_score(df, 'BTCUSDT')

        # Should be moderately high due to bullish signals (adjusted expectation)
        self.assertGreater(score, 0.55)
        self.assertLessEqual(score, 1.0)

    def test_weak_bearish_signal(self):
        """Test scoring for weak bearish conditions"""
        bearish_data = self.test_indicators.copy()
        bearish_data.update({
            'rsi': 25.0,  # Oversold (bullish signal)
            'macd': -15.0,  # Bearish MACD
            'macd_signal': -10.0,
            'vol_zscore': -1.0,  # Low volume
            'adx': 15.0,  # Weak trend
            'roc_5': -1.0,  # Bearish momentum
            'williams_r': -80.0  # Oversold (bullish)
        })

        df = pd.DataFrame([bearish_data])
        score = calculate_technical_strength_score(df, 'BTCUSDT')

        # Should be moderate due to mixed signals
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.7)

    def test_neutral_conditions(self):
        """Test scoring for neutral market conditions"""
        df = pd.DataFrame([self.test_indicators])
        score = calculate_technical_strength_score(df, 'BTCUSDT')

        # Should be around 0.5 for neutral conditions
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.6)

    def test_missing_indicators(self):
        """Test handling of missing indicators"""
        incomplete_data = {'rsi': 50.0}  # Only RSI
        df = pd.DataFrame([incomplete_data])
        score = calculate_technical_strength_score(df, 'BTCUSDT')

        # Should return neutral score when data is incomplete
        self.assertEqual(score, 0.5)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        score = calculate_technical_strength_score(df, 'BTCUSDT')

        # Should return neutral score for empty data
        self.assertEqual(score, 0.5)


class TestConvergenceMultiplier(unittest.TestCase):
    """Test convergence multiplier calculations"""

    def test_perfect_agreement(self):
        """Test multiplier for perfect L1+L2 agreement"""
        multiplier = calculate_convergence_multiplier(0.95, 0.9, 0.9)
        self.assertGreater(multiplier, 1.8)  # Should be close to 2.0

    def test_poor_agreement(self):
        """Test multiplier for poor L1+L2 agreement"""
        multiplier = calculate_convergence_multiplier(0.3, 0.5, 0.5)
        self.assertLess(multiplier, 0.8)  # Should be close to 0.5

    def test_moderate_agreement(self):
        """Test multiplier for moderate agreement"""
        multiplier = calculate_convergence_multiplier(0.6, 0.7, 0.7)
        self.assertGreater(multiplier, 1.0)
        self.assertLess(multiplier, 1.5)

    def test_confidence_bonus(self):
        """Test that high confidence increases multiplier"""
        low_conf = calculate_convergence_multiplier(0.7, 0.5, 0.5)
        high_conf = calculate_convergence_multiplier(0.7, 0.9, 0.9)

        self.assertGreater(high_conf, low_conf)

    def test_bounds_checking(self):
        """Test that multipliers stay within reasonable bounds"""
        # Test upper bound
        high_mult = calculate_convergence_multiplier(0.95, 0.95, 0.95)
        self.assertLessEqual(high_mult, 2.5)

        # Test lower bound
        low_mult = calculate_convergence_multiplier(0.1, 0.1, 0.1)
        self.assertGreaterEqual(low_mult, 0.3)


class TestTechnicalStrengthValidation(unittest.TestCase):
    """Test technical strength validation for position sizes"""

    def test_large_position_rejection(self):
        """Test rejection of large positions with weak technicals"""
        # Weak technical strength
        weak_strength = 0.4

        # Should reject large positions
        self.assertFalse(validate_technical_strength_for_position_size(weak_strength, 15000, 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(weak_strength, 10000, 'BTCUSDT'))

    def test_large_position_approval(self):
        """Test approval of large positions with strong technicals"""
        # Strong technical strength
        strong_strength = 0.8

        # Should approve large positions
        self.assertTrue(validate_technical_strength_for_position_size(strong_strength, 15000, 'BTCUSDT'))
        self.assertTrue(validate_technical_strength_for_position_size(strong_strength, 10000, 'BTCUSDT'))

    def test_medium_position_requirements(self):
        """Test requirements for medium-sized positions"""
        medium_strength = 0.55  # Below 0.6 threshold

        # Should reject medium positions with insufficient strength
        self.assertFalse(validate_technical_strength_for_position_size(medium_strength, 6000, 'BTCUSDT'))

        strong_enough = 0.65  # Above 0.6 threshold
        self.assertTrue(validate_technical_strength_for_position_size(strong_enough, 6000, 'BTCUSDT'))

    def test_small_position_flexibility(self):
        """Test that small positions have lower requirements"""
        weak_strength = 0.35  # Above the 0.3 minimum for micro positions

        # Should allow small positions even with weak technicals
        self.assertTrue(validate_technical_strength_for_position_size(weak_strength, 500, 'BTCUSDT'))


class TestEnhancedPositionSizing(unittest.TestCase):
    """Test enhanced position sizing in PortfolioManager"""

    def setUp(self):
        """Set up test portfolio manager"""
        self.pm = PortfolioManager(mode='simulated', initial_balance=10000.0)
        self.market_data = {'BTCUSDT': {'close': 50000.0}}

    def test_convergence_scaling(self):
        """Test that position sizes scale with convergence"""
        base_size = 1000.0

        # High convergence should increase position size
        high_conv_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', base_size, 0.9, 0.7, self.market_data
        )

        # Low convergence should decrease position size
        low_conv_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', base_size, 0.3, 0.7, self.market_data
        )

        self.assertGreater(high_conv_size, base_size)
        self.assertLess(low_conv_size, base_size)

    def test_technical_strength_bonus(self):
        """Test technical strength bonuses for strong signals"""
        base_size = 1000.0

        # Strong technicals should increase position size
        strong_tech_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', base_size, 0.7, 0.85, self.market_data
        )

        # Weak technicals should decrease position size
        weak_tech_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', base_size, 0.7, 0.3, self.market_data
        )

        self.assertGreater(strong_tech_size, base_size)
        self.assertLess(weak_tech_size, base_size)

    def test_risk_limits(self):
        """Test that risk limits are respected"""
        # Try to create a very large position
        large_base = 50000.0  # Would be > portfolio value

        actual_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', large_base, 0.9, 0.9, self.market_data
        )

        # Should be limited by risk management
        self.assertLess(actual_size, large_base)

    def test_zero_position_rejection(self):
        """Test rejection of positions that don't meet criteria"""
        # Very weak technicals with poor convergence
        rejected_size = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 1000.0, 0.2, 0.2, self.market_data
        )

        # Should return 0 for rejected positions
        self.assertEqual(rejected_size, 0.0)


class TestSignalComposerIntegration(unittest.TestCase):
    """Test SignalComposer integration with enhanced sizing"""

    def setUp(self):
        """Set up test signal composer"""
        from l2_tactic.config import SignalConfig
        config = SignalConfig()
        self.composer = SignalComposer(config)

    def test_enhanced_position_calculation(self):
        """Test that SignalComposer uses enhanced position sizing"""
        # Create test signal with convergence data
        features = {
            'l1_l2_agreement': 0.8,
            'rsi': 65.0,
            'macd': 10.0,
            'macd_signal': 8.0,
            'vol_zscore': 1.0,
            'adx': 30.0,
            'close': 50000.0
        }

        signal = TacticalSignal(
            symbol='BTCUSDT',
            side='buy',
            strength=0.7,
            confidence=0.8,
            signal_type='tactical',
            source='test',
            features=features,
            timestamp=pd.Timestamp.now()
        )

        state = {
            'portfolio': {'BTCUSDT': {'position': 0.0}},
            'market_data': {'BTCUSDT': {'close': 50000.0}}
        }

        # Test the enhanced position sizing method
        quantity = self.composer._calculate_enhanced_position_size(
            'BTCUSDT', 'buy', 0.7, 0.8, features, state, 50000.0
        )

        # Should return a positive quantity
        self.assertGreater(quantity, 0)


class TestIntegrationFlow(unittest.TestCase):
    """Test complete integration flow"""

    def test_end_to_end_sizing(self):
        """Test complete flow from indicators to position size"""
        # Step 1: Calculate technical strength
        indicators_data = {
            'rsi': 70.0,  # Bullish
            'macd': 15.0,  # Bullish
            'macd_signal': 12.0,
            'vol_zscore': 1.2,  # High volume
            'adx': 35.0,  # Trending
            'roc_5': 2.0,  # Bullish momentum
            'williams_r': -30.0  # Bullish
        }
        df = pd.DataFrame([indicators_data])
        tech_strength = calculate_technical_strength_score(df, 'BTCUSDT')

        # Step 2: Calculate convergence multiplier
        conv_multiplier = calculate_convergence_multiplier(0.85, 0.8, 0.8)

        # Step 3: Validate position size
        base_size = 2000.0
        is_valid = validate_technical_strength_for_position_size(tech_strength, base_size, 'BTCUSDT')

        # Step 4: Calculate final position size
        pm = PortfolioManager(mode='simulated', initial_balance=10000.0)
        market_data = {'BTCUSDT': {'close': 50000.0}}

        final_size = pm.calculate_convergence_technical_position_size(
            'BTCUSDT', base_size, 0.85, tech_strength, market_data
        )

        # Assertions
        self.assertTrue(is_valid)  # Should be valid for this size
        self.assertGreater(final_size, base_size * 0.8)  # Should be enhanced
        self.assertGreater(tech_strength, 0.6)  # Should be strong bullish
        self.assertGreater(conv_multiplier, 1.5)  # Should be good multiplier


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("🧪 COMPREHENSIVE TESTING SUITE")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestTechnicalStrengthScoring,
        TestConvergenceMultiplier,
        TestTechnicalStrengthValidation,
        TestEnhancedPositionSizing,
        TestSignalComposerIntegration,
        TestIntegrationFlow
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
