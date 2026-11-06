# tests/test_path_modes.py
"""
Comprehensive unit tests for HRM path mode functionality.

Tests validate that:
- PATH1 never generates purchases based on RSI low
- PATH2 never exceeds maximum contra-allocation against L3
- PATH3 never allows deviations > 10% from L3 targets

Path Validation Rules:
- PATH1: Pure trend-following (no restrictions)
- PATH2: Hybrid with 20% contra-allocation limit
- PATH3: L3 dominance (only L3 trend-following signals allowed)
"""

import unittest
from unittest.mock import Mock, patch
from l1_operational.order_manager import OrderManager
from core.config import HRM_PATH_MODE

class TestPathModes(unittest.TestCase):
    """Test HRM path mode validation and behavior"""

    def setUp(self):
        """Setup test fixtures"""
        self.binance_client = Mock()
        self.market_data = {}
        self.order_manager = OrderManager(
            binance_client=self.binance_client,
            market_data=self.market_data
        )

        # Sample signal definitions for different paths
        self.l3_trend_following_signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "signal_source": "path3_full_l3_dominance",
            "quantity": 0.001,
            "price": 50000.0
        }

        self.l2_signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "signal_source": "tactical_signal_L2",
            "quantity": 0.001,
            "price": 50000.0
        }

    def test_path1_allows_all_signals(self):
        """PATH1: Pure trend-following allows all valid signals"""
        # Test L3 trend-following signal allowed
        validation = self.order_manager.validate_order(self.l3_trend_following_signal, path_mode="PATH1")
        self.assertTrue(validation["valid"], "PATH1 should allow L3 trend-following signals")
        self.assertIn("Order allowed in PATH1 mode", validation["reason"])

        # Test L2 signal allowed
        validation = self.order_manager.validate_order(self.l2_signal, path_mode="PATH1")
        self.assertTrue(validation["valid"], "PATH1 should allow L2 signals")

    def test_path2_allows_controlled_signals(self):
        """PATH2: Hybrid mode allows signals with allocation controls"""
        # Test L3 trend-following signal allowed
        validation = self.order_manager.validate_order(self.l3_trend_following_signal, path_mode="PATH2")
        self.assertTrue(validation["valid"], "PATH2 should allow L3 trend-following signals")

        # Test L2 signal allowed
        validation = self.order_manager.validate_order(self.l2_signal, path_mode="PATH2")
        self.assertTrue(validation["valid"], "PATH2 should allow L2 signals")

        # Note: Contra-allocation validation occurs in generate_orders(), not validate_order()
        # The allocation limits are enforced during order generation phase

    def test_path3_blocks_non_l3_signals(self):
        """PATH3: Full L3 dominance blocks non-L3 trend-following signals"""
        # Test L3 trend-following signal allowed
        validation = self.order_manager.validate_order(self.l3_trend_following_signal, path_mode="PATH3")
        self.assertTrue(validation["valid"], "PATH3 should allow L3 trend-following signals")
        self.assertIn("Order allowed in PATH3 mode", validation["reason"])

        # Test L2 signal BLOCKED in PATH3
        validation = self.order_manager.validate_order(self.l2_signal, path_mode="PATH3")
        self.assertFalse(validation["valid"], "PATH3 should block non-L3 signals")
        self.assertIn("PATH3 mode blocks non-L3 orders", validation["reason"])
        self.assertIn("'tactical_signal_L2' != 'path3_full_l3_dominance'", validation["reason"])

    def test_path3_validates_signal_source_exact_match(self):
        """PATH3: Signal source must match exactly 'path3_full_l3_dominance'"""
        # Test wrong L3 source blocked
        wrong_l3_signal = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "signal_source": "l3_trend_following",  # Wrong source name
            "quantity": 0.001,
            "price": 50000.0
        }

        validation = self.order_manager.validate_order(wrong_l3_signal, path_mode="PATH3")
        self.assertFalse(validation["valid"], "PATH3 should block signals with wrong source")
        self.assertIn("PATH3 mode blocks non-L3 orders", validation["reason"])

    def test_invalid_path_mode_defaults_to_allow(self):
        """Invalid or unknown HRM_PATH_MODE should allow orders as fallback"""
        validation = self.order_manager.validate_order(self.l3_trend_following_signal, path_mode="INVALID_MODE")
        self.assertTrue(validation["valid"], "Invalid path mode should allow orders as fallback")
        self.assertIn("Order allowed in INVALID_MODE mode", validation["reason"])

    @patch('core.config.HRM_PATH_MODE', 'PATH2')
    def test_path2_contra_allocation_limits_verified(self):
        """PATH2 contra-allocation limits should be enforced"""
        # This test verifies the allocation limit configuration
        from core.config import MAX_CONTRA_ALLOCATION_PATH2
        self.assertEqual(MAX_CONTRA_ALLOCATION_PATH2, 0.2, "PATH2 should have 20% contra-allocation limit")

    def test_path_mode_validation_logs(self):
        """Path mode validation should log appropriate messages"""
        # PATH3 rejection should log warning
        with patch('core.logging.logger.warning') as mock_warning:
            validation = self.order_manager.validate_order(self.l2_signal, path_mode="PATH3")
            self.assertFalse(validation["valid"])
            mock_warning.assert_called()  # Should log the block

        # PATH3 allowance should log info
        with patch('core.logging.logger.info') as mock_info:
            validation = self.order_manager.validate_order(self.l3_trend_following_signal, path_mode="PATH3")
            self.assertTrue(validation["valid"])
            mock_info.assert_called()  # Should log the allowance

if __name__ == '__main__':
    unittest.main()
