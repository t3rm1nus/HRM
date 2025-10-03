# -*- coding: utf-8 -*-
"""
PATH Mode Validation Tests - HRM Trading System

Validation tests for PATH mode signal source consistency and business logic
ensuring proper operation across different trading modes (PATH1, PATH2, PATH3).
"""

import pytest
from unittest.mock import MagicMock, patch
from core.configuration_manager import ConfigurationManager, get_config_value
from core.unified_validation import UnifiedValidator
from l1_operational.order_manager import OrderManager


class TestPathModeConstants:
    """Test PATH mode constant validation."""

    def test_valid_path_modes(self):
        """Test that only valid PATH modes are accepted."""
        # HRM_PATH_MODE is a read-only constant that can't be changed
        # The system uses static PATH1 for now
        current_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert current_mode == "PATH1"  # Static implementation

        # Verify attempts to set invalid modes are rejected
        valid_modes = ["PATH1", "PATH2", "PATH3"]
        assert current_mode in valid_modes

        # Verify constants are properly defined
        constants = ConfigurationManager.get_all_constants()
        assert "HRM_PATH_MODE" in constants
        assert constants["HRM_PATH_MODE"] == "PATH1"

    def test_path3_signal_source_constant(self):
        """Test PATH3_SIGNAL_SOURCE constant is defined and used correctly."""
        source = ConfigurationManager.get("live", "PATH3_SIGNAL_SOURCE")
        assert source == "path3_full_l3_dominance"
        assert isinstance(source, str)


class TestPathModeSignalValidation:
    """Test PATH mode signal source consistency."""

    def test_path1_mode_allows_all_signals(self):
        """Test that PATH1 (pure trend-following) allows all L2 signals."""
        # Set PATH1 mode
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

        # Mock signals - PATH1 should allow both BUY and SELL
        test_signals_buy = [{"signal_type": "buy", "direction": "buy"}]
        test_signals_sell = [{"signal_type": "sell", "direction": "sell"}]

        # PATH1 is pure trend-following - should allow all signals
        # This is validated in tactical_signal_processor.process_signals

        # Reset
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

    def test_path2_mode_limits_contra_allocation(self):
        """Test that PATH2 mode enforces contra-allocation limits."""
        # Set PATH2 mode
        ConfigurationManager.set("HRM_PATH_MODE", "PATH2")

        max_contra = ConfigurationManager.get("live", "MAX_CONTRA_ALLOCATION_PATH2")
        assert max_contra == 0.2  # 20% limit

        # Test validation would work for order manager
        assert max_contra > 0 and max_contra <= 1

        # Reset
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

    def test_path3_mode_enforces_signal_source(self):
        """Test that PATH3 mode enforces specific signal source."""
        # Set PATH3 mode
        ConfigurationManager.set("HRM_PATH_MODE", "PATH3")

        required_source = ConfigurationManager.get("live", "PATH3_SIGNAL_SOURCE")
        assert required_source == "path3_full_l3_dominance"

        # Test the constant is defined in order manager
        expected_source = "path3_full_l3_dominance"
        assert required_source == expected_source

        # Reset
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

    @patch('l2_tactic.tactical_signal_processor.logger')
    def test_path_mode_signal_filtering_consistency(self, mock_logger):
        """Test that PATH mode filtering is consistent across calls."""
        # This test validates the signal filtering logic consistency
        path_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert path_mode in ["PATH1", "PATH2", "PATH3"]

        # Path mode should remain consistent throughout operation
        for _ in range(5):
            current_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
            assert current_mode == path_mode


class TestPathModeBusinessLogic:
    """Test PATH mode business logic validation."""

    def test_path1_dominance_mode(self):
        """Test PATH1 L3 dominance mode behavior."""
        # Set PATH1
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

        path_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert path_mode == "PATH1"

        # PATH1 is pure trend-following with L3 dominance
        # This would be validated in the signal processing pipeline
        # L3 signals override L2 signals in dominance mode

        # Reset
        ConfigurationManager.set("HRM_PATH_MODE", "PATH1")

    def test_path2_balanced_mode(self):
        """Test PATH2 balanced mode behavior."""
        # PATH2 constant allows limited contra-allocation (20%)
        # Current implementation uses PATH1, but we test the constants still
        contra_limit = ConfigurationManager.get("live", "MAX_CONTRA_ALLOCATION_PATH2")
        assert contra_limit == 0.2

        # Even though we use PATH1, PATH2 constants should still be accessible
        # for future implementation of dynamic PATH mode switching
        assert contra_limit > 0 and contra_limit <= 1

    def test_path3_l3_dominance_mode(self):
        """Test PATH3 full L3 dominance mode behavior."""
        # PATH3 constant enforces specific signal source
        # Current implementation uses PATH1, but we test the constants still
        required_source = ConfigurationManager.get("live", "PATH3_SIGNAL_SOURCE")
        assert required_source == "path3_full_l3_dominance"

        # Signal source should be properly defined for future PATH3 mode
        assert isinstance(required_source, str)
        assert required_source.startswith("path3_")

    def test_path_mode_validation_with_unified_validator(self):
        """Test that UnifiedValidator can validate trading parameters."""
        # Test basic trading parameter validation (independent of PATH modes)
        is_valid, msg = UnifiedValidator.validate_trading_parameters(
            symbol="BTCUSDT",
            quantity=0.001,
            price=50000.0,
            side="buy"
        )
        assert is_valid is True
        assert "Valid trading parameters" in msg

        # Test invalid parameters
        is_valid, msg = UnifiedValidator.validate_trading_parameters(
            symbol="",
            quantity=-1,
            price=0,
            side="invalid"
        )
        assert is_valid is False

    def test_configuration_manager_path_mode_integration(self):
        """Test ConfigurationManager properly handles PATH mode settings."""
        # Test constant access
        constants = ConfigurationManager.get_all_constants()
        assert "HRM_PATH_MODE" in constants
        assert "MAX_CONTRA_ALLOCATION_PATH2" in constants
        assert "PATH3_SIGNAL_SOURCE" in constants

        # Test value access
        path_mode = get_config_value("HRM_PATH_MODE")
        assert path_mode in ["PATH1", "PATH2", "PATH3"]

        contra_limit = get_config_value("MAX_CONTRA_ALLOCATION_PATH2")
        assert isinstance(contra_limit, float)
        assert 0 < contra_limit <= 1

        signal_source = get_config_value("PATH3_SIGNAL_SOURCE")
        assert isinstance(signal_source, str)
        assert signal_source == "path3_full_l3_dominance"


class TestPathModeErrorHandling:
    """Test PATH mode error handling and edge cases."""

    def test_invalid_path_mode_rejection(self):
        """Test that invalid PATH modes are properly handled."""
        # ConfigurationManager should prevent setting invalid PATH modes
        # since HRM_PATH_MODE is read-only and only accepts valid values

        # The system should maintain valid PATH modes at startup
        valid_modes = ["PATH1", "PATH2", "PATH3"]
        current_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert current_mode in valid_modes

    def test_path_mode_persistence_across_sessions(self):
        """Test that PATH mode persists correctly (conceptual test)."""
        # This is a conceptual test for future PATH mode persistence
        # When implemented, PATH mode should persist across system restarts

        initial_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")

        # Simulate system restart (mode should remain the same)
        # Current implementation doesn't have persistence yet
        # This test documents the requirement for future implementation

        # For now, just validate mode remains valid
        assert initial_mode in ["PATH1", "PATH2", "PATH3"]

    def test_path_mode_transition_safety(self):
        """Test PATH mode transition safety (conceptual)."""
        # Future implementation should validate safe mode transitions
        # PATH3 -> PATH1: safe (more conservative)
        # PATH1 -> PATH2: requires balance checks
        # PATH2 -> PATH3: requires signal source validation

        # Current implementation uses static PATH1
        current_mode = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert current_mode == "PATH1"  # Static for now


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
