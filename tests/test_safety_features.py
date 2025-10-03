#!/usr/bin/env python3
"""
Comprehensive test suite for convergence and technical strength safety features.
Tests circuit breakers, configuration system, and gradual rollout controls.
"""

import sys
import os
sys.path.append('.')

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import safety-enhanced functions
from core.technical_indicators import (
    calculate_technical_strength_score,
    calculate_convergence_multiplier,
    validate_technical_strength_for_position_size,
    apply_convergence_safety_limits,
    get_convergence_safety_mode
)
from core.portfolio_manager import PortfolioManager
from core.convergence_config import (
    ConvergenceConfig,
    get_convergence_config,
    enable_convergence_features,
    emergency_disable_convergence,
    get_convergence_status
)


class TestCircuitBreakers(unittest.TestCase):
    """Test circuit breaker functionality"""

    def test_invalid_strength_score_circuit_breaker(self):
        """Test circuit breaker for invalid strength scores"""
        # Test invalid types
        self.assertFalse(validate_technical_strength_for_position_size("invalid", 1000, 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(None, 1000, 'BTCUSDT'))

        # Test out of range values
        self.assertFalse(validate_technical_strength_for_position_size(-0.1, 1000, 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(1.1, 1000, 'BTCUSDT'))

    def test_invalid_position_size_circuit_breaker(self):
        """Test circuit breaker for invalid position sizes"""
        # Test invalid types
        self.assertFalse(validate_technical_strength_for_position_size(0.5, "invalid", 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(0.5, None, 'BTCUSDT'))

        # Test negative values
        self.assertFalse(validate_technical_strength_for_position_size(0.5, -100, 'BTCUSDT'))

    def test_emergency_extreme_weak_strength(self):
        """Test emergency circuit breaker for extremely weak technical strength"""
        # Strength below 0.1 should always be rejected
        self.assertFalse(validate_technical_strength_for_position_size(0.05, 1000, 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(0.01, 500, 'BTCUSDT'))

    def test_emergency_extreme_large_positions(self):
        """Test emergency circuit breaker for extremely large positions"""
        # Positions over $100K should always be rejected regardless of strength
        self.assertFalse(validate_technical_strength_for_position_size(0.9, 150000, 'BTCUSDT'))
        self.assertFalse(validate_technical_strength_for_position_size(1.0, 200000, 'BTCUSDT'))


class TestSafetyLimits(unittest.TestCase):
    """Test convergence safety limits"""

    def test_conservative_safety_limits(self):
        """Test conservative safety mode limits"""
        # Test upper bounds
        limited = apply_convergence_safety_limits(3.0, "conservative")
        self.assertEqual(limited, 1.5)  # Should be capped at 1.5

        # Test lower bounds
        limited = apply_convergence_safety_limits(0.3, "conservative")
        self.assertEqual(limited, 0.6)  # Should be floored at 0.6

    def test_moderate_safety_limits(self):
        """Test moderate safety mode limits"""
        # Test upper bounds
        limited = apply_convergence_safety_limits(3.0, "moderate")
        self.assertEqual(limited, 2.0)  # Should be capped at 2.0

        # Test lower bounds
        limited = apply_convergence_safety_limits(0.2, "moderate")
        self.assertEqual(limited, 0.5)  # Should be floored at 0.5

    def test_emergency_safety_limits(self):
        """Test emergency safety mode limits"""
        # Test upper bounds
        limited = apply_convergence_safety_limits(3.0, "emergency")
        self.assertEqual(limited, 1.0)  # Should be capped at 1.0

        # Test lower bounds
        limited = apply_convergence_safety_limits(0.2, "emergency")
        self.assertEqual(limited, 0.5)  # Should be floored at 0.5


class TestConfigurationSystem(unittest.TestCase):
    """Test convergence configuration system"""

    def setUp(self):
        """Set up test configuration"""
        # Use a test config file
        self.test_config_file = "test_convergence_config.json"
        self.config = ConvergenceConfig(self.test_config_file)

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)

    def test_default_configuration(self):
        """Test default configuration values"""
        self.assertFalse(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "disabled")
        self.assertEqual(self.config.get_safety_mode(), "conservative")

    def test_monitoring_mode_enablement(self):
        """Test monitoring-only mode enablement"""
        self.config.enable_monitoring_only()

        self.assertTrue(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "monitoring_only")
        self.assertEqual(self.config.get_safety_mode(), "conservative")

        # Features should be disabled for actual application
        self.assertFalse(self.config.is_feature_enabled("convergence_multiplier"))
        self.assertFalse(self.config.is_feature_enabled("technical_strength_scoring"))

    def test_conservative_mode_enablement(self):
        """Test conservative operational mode"""
        self.config.enable_conservative_mode()

        self.assertTrue(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "conservative_enabled")
        self.assertEqual(self.config.get_safety_mode(), "conservative")

        # Basic features should be enabled
        self.assertTrue(self.config.is_feature_enabled("convergence_multiplier"))
        self.assertTrue(self.config.is_feature_enabled("technical_strength_scoring"))

    def test_moderate_mode_enablement(self):
        """Test moderate operational mode"""
        self.config.enable_moderate_mode()

        self.assertTrue(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "moderate_enabled")
        self.assertEqual(self.config.get_safety_mode(), "moderate")

        # All features should be enabled
        self.assertTrue(self.config.is_feature_enabled("convergence_multiplier"))
        self.assertTrue(self.config.is_feature_enabled("technical_strength_scoring"))

    def test_full_mode_enablement(self):
        """Test full operational mode"""
        self.config.enable_full_mode()

        self.assertTrue(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "full_enabled")
        self.assertEqual(self.config.get_safety_mode(), "aggressive")

        # All features should be enabled with maximum limits
        self.assertTrue(self.config.is_feature_enabled("convergence_multiplier"))
        self.assertTrue(self.config.is_feature_enabled("technical_strength_scoring"))

    def test_emergency_disable(self):
        """Test emergency disable functionality"""
        # First enable features
        self.config.enable_full_mode()
        self.assertTrue(self.config.is_enabled())

        # Then emergency disable
        self.config.emergency_disable()
        self.assertFalse(self.config.is_enabled())
        self.assertEqual(self.config.get_rollout_phase(), "emergency_disabled")
        self.assertEqual(self.config.get_safety_mode(), "emergency")

        # All features should be disabled
        self.assertFalse(self.config.is_feature_enabled("convergence_multiplier"))
        self.assertFalse(self.config.is_feature_enabled("technical_strength_scoring"))

    def test_configuration_persistence(self):
        """Test configuration save/load persistence"""
        # Modify configuration
        self.config.enable_conservative_mode()
        self.config.save_config()

        # Create new instance and verify persistence
        new_config = ConvergenceConfig(self.test_config_file)
        self.assertTrue(new_config.is_enabled())
        self.assertEqual(new_config.get_rollout_phase(), "conservative_enabled")


class TestPortfolioManagerSafety(unittest.TestCase):
    """Test PortfolioManager safety integration"""

    def setUp(self):
        """Set up test portfolio manager"""
        self.pm = PortfolioManager(mode='simulated', initial_balance=10000.0)
        self.market_data = {'BTCUSDT': {'close': 50000.0}}

    def test_disabled_convergence_fallback(self):
        """Test fallback to base sizing when convergence is disabled"""
        # Ensure convergence is disabled
        config = get_convergence_config()
        if config.is_enabled():
            config.emergency_disable()

        # Should return base position size
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 1000.0, 0.8, 0.7, self.market_data
        )
        self.assertEqual(result, 1000.0)

    def test_monitoring_mode_logging_only(self):
        """Test monitoring mode logs calculations but doesn't apply them"""
        config = get_convergence_config()
        config.enable_monitoring_only()

        # Should return base position size (not enhanced)
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 1000.0, 0.9, 0.8, self.market_data
        )
        self.assertEqual(result, 1000.0)  # Should be unchanged

    def test_emergency_stop_technical_strength(self):
        """Test emergency stop for extremely weak technical strength"""
        config = get_convergence_config()
        config.enable_full_mode()

        # Extremely weak technical strength should result in zero position
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 1000.0, 0.8, 0.01, self.market_data  # Strength < 0.05 threshold
        )
        self.assertEqual(result, 0.0)  # Should be rejected

    def test_configuration_risk_limits(self):
        """Test configuration-based risk limits"""
        config = get_convergence_config()
        config.enable_moderate_mode()

        # Test minimum position size limit
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 5.0, 0.8, 0.7, self.market_data  # Below $10 minimum
        )
        self.assertEqual(result, 0.0)  # Should be rejected

    def test_error_handling_circuit_breaker(self):
        """Test error handling with circuit breaker"""
        config = get_convergence_config()
        config.enable_full_mode()

        # Force an error condition
        with patch('core.technical_indicators.validate_technical_strength_for_position_size') as mock_validate:
            mock_validate.side_effect = Exception("Test error")

            result = self.pm.calculate_convergence_technical_position_size(
                'BTCUSDT', 1000.0, 0.8, 0.7, self.market_data
            )
            # Should return 0 due to circuit breaker on error
            self.assertEqual(result, 0.0)


class TestGradualRollout(unittest.TestCase):
    """Test gradual rollout functionality"""

    def setUp(self):
        """Set up test configuration"""
        self.test_config_file = "test_rollout_config.json"
        self.config = ConvergenceConfig(self.test_config_file)

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)

    def test_rollout_phase_progression(self):
        """Test proper rollout phase progression"""
        # Start disabled
        self.assertEqual(self.config.get_rollout_phase(), "disabled")

        # Phase 1: Monitoring
        self.config.enable_monitoring_only()
        self.assertEqual(self.config.get_rollout_phase(), "monitoring_only")

        # Phase 2: Conservative
        self.config.enable_conservative_mode()
        self.assertEqual(self.config.get_rollout_phase(), "conservative_enabled")

        # Phase 3: Moderate
        self.config.enable_moderate_mode()
        self.assertEqual(self.config.get_rollout_phase(), "moderate_enabled")

        # Phase 4: Full
        self.config.enable_full_mode()
        self.assertEqual(self.config.get_rollout_phase(), "full_enabled")

    def test_conservative_mode_limits(self):
        """Test conservative mode has appropriate limits"""
        self.config.enable_conservative_mode()

        risk_limits = self.config.get_risk_limits()
        self.assertEqual(risk_limits["max_portfolio_allocation"], 0.6)
        self.assertEqual(risk_limits["max_position_size_usd"], 25000.0)

        conv_config = self.config.get_feature_config("convergence_multiplier")
        self.assertEqual(conv_config["max_multiplier"], 1.3)

    def test_moderate_mode_limits(self):
        """Test moderate mode has appropriate limits"""
        self.config.enable_moderate_mode()

        risk_limits = self.config.get_risk_limits()
        self.assertEqual(risk_limits["max_portfolio_allocation"], 0.7)
        self.assertEqual(risk_limits["max_position_size_usd"], 35000.0)

        conv_config = self.config.get_feature_config("convergence_multiplier")
        self.assertEqual(conv_config["max_multiplier"], 1.8)

    def test_full_mode_limits(self):
        """Test full mode has maximum limits"""
        self.config.enable_full_mode()

        risk_limits = self.config.get_risk_limits()
        self.assertEqual(risk_limits["max_portfolio_allocation"], 0.8)
        self.assertEqual(risk_limits["max_position_size_usd"], 50000.0)

        conv_config = self.config.get_feature_config("convergence_multiplier")
        self.assertEqual(conv_config["max_multiplier"], 2.0)


class TestIntegrationSafety(unittest.TestCase):
    """Test complete integration with safety features"""

    def setUp(self):
        """Set up test environment"""
        self.pm = PortfolioManager(mode='simulated', initial_balance=10000.0)
        self.market_data = {'BTCUSDT': {'close': 50000.0}}

    def test_end_to_end_safe_operation(self):
        """Test complete safe operation flow"""
        # Enable conservative mode
        config = get_convergence_config()
        config.enable_conservative_mode()

        # Test normal operation
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', 1000.0, 0.8, 0.7, self.market_data
        )

        # Should return enhanced size within safe limits
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 25000.0)  # Conservative limit

    def test_emergency_fallback_chain(self):
        """Test emergency fallback chain"""
        # Start with full features enabled
        config = get_convergence_config()
        config.enable_full_mode()

        # Test with invalid inputs
        result = self.pm.calculate_convergence_technical_position_size(
            'BTCUSDT', -1000.0, 1.5, -0.1, self.market_data
        )

        # Should safely return 0 due to validation
        self.assertEqual(result, 0.0)

    def test_configuration_status_reporting(self):
        """Test configuration status reporting"""
        config = get_convergence_config()
        config.enable_moderate_mode()

        status = config.get_status_summary()

        required_keys = ["enabled", "rollout_phase", "safety_mode", "features_status", "risk_limits"]
        for key in required_keys:
            self.assertIn(key, status)

        self.assertTrue(status["enabled"])
        self.assertEqual(status["rollout_phase"], "moderate_enabled")


def run_safety_tests():
    """Run all safety tests with detailed output"""
    print("🛡️ COMPREHENSIVE SAFETY FEATURES TEST SUITE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCircuitBreakers,
        TestSafetyLimits,
        TestConfigurationSystem,
        TestPortfolioManagerSafety,
        TestGradualRollout,
        TestIntegrationSafety
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("SAFETY FEATURES TEST SUMMARY")
    print("=" * 60)
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

    # Overall safety assessment
    print("\n" + "=" * 60)
    print("SAFETY ASSESSMENT")
    print("=" * 60)

    if result.wasSuccessful():
        print("✅ ALL SAFETY TESTS PASSED")
        print("🛡️ System is safe for production deployment")
        print("🔄 Gradual rollout controls are functional")
        print("🚨 Circuit breakers are operational")
    else:
        print("❌ SAFETY TESTS FAILED")
        print("🚨 DO NOT DEPLOY - Safety features compromised")
        print("🔍 Review failures before proceeding")

    return result.wasSuccessful()


def demonstrate_safety_features():
    """Demonstrate safety features in action"""
    print("\n🎯 SAFETY FEATURES DEMONSTRATION")
    print("=" * 40)

    # Test configuration system
    print("1. Configuration System:")
    config = get_convergence_config()

    print("   Default state:")
    status = config.get_status_summary()
    print(f"     Enabled: {status['enabled']}, Phase: {status['rollout_phase']}")

    print("   Enabling monitoring mode:")
    config.enable_monitoring_only()
    status = config.get_status_summary()
    print(f"     Enabled: {status['enabled']}, Phase: {status['rollout_phase']}")

    print("   Enabling conservative mode:")
    config.enable_conservative_mode()
    status = config.get_status_summary()
    print(f"     Enabled: {status['enabled']}, Phase: {status['rollout_phase']}")

    # Test circuit breakers
    print("\n2. Circuit Breakers:")
    print("   Testing invalid inputs:")

    # Invalid strength score
    result = validate_technical_strength_for_position_size(-0.5, 1000, 'BTCUSDT')
    print(f"     Invalid strength (-0.5): {'REJECTED' if not result else 'ALLOWED'}")

    # Invalid position size
    result = validate_technical_strength_for_position_size(0.7, -1000, 'BTCUSDT')
    print(f"     Invalid position (-$1000): {'REJECTED' if not result else 'ALLOWED'}")

    # Emergency conditions
    result = validate_technical_strength_for_position_size(0.02, 1000, 'BTCUSDT')
    print(f"     Emergency weak strength (0.02): {'REJECTED' if not result else 'ALLOWED'}")

    result = validate_technical_strength_for_position_size(0.8, 150000, 'BTCUSDT')
    print(f"     Emergency large position ($150K): {'REJECTED' if not result else 'ALLOWED'}")

    # Test safety limits
    print("\n3. Safety Limits:")
    test_multipliers = [0.2, 1.0, 2.5, 4.0]

    for mult in test_multipliers:
        conservative = apply_convergence_safety_limits(mult, "conservative")
        moderate = apply_convergence_safety_limits(mult, "moderate")
        emergency = apply_convergence_safety_limits(mult, "emergency")
        print(f"     {mult:.1f}x → Conservative: {conservative:.1f}x, Moderate: {moderate:.1f}x, Emergency: {emergency:.1f}x")

    print("\n✅ Safety features demonstration completed")


if __name__ == '__main__':
    # Run safety tests
    success = run_safety_tests()

    # Demonstrate features
    demonstrate_safety_features()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
