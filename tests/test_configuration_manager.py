# -*- coding: utf-8 -*-
"""
Test suite for unified configuration management system - HRM Trading System

Comprehensive validation of centralized configuration access to eliminate
inconsistent direct imports and function calls across the system.
"""

import os
import tempfile
import pytest
from unittest.mock import patch
import core.configuration_manager as config_mgmt
from core.configuration_manager import (
    ConfigurationManager,
    get_config as legacy_get_config,  # Backward compatibility wrapper
    get_config_value,
    set_config_value,
)
from core.config import EnvironmentConfig


class TestConfigurationManagerConstants:
    """Test configuration manager constant access."""

    def test_get_path_mode_constant(self):
        """Test getting HRM_PATH_MODE constant."""
        value = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert value == "PATH1"

    def test_get_max_contra_allocation_constant(self):
        """Test getting MAX_CONTRA_ALLOCATION_PATH2 constant."""
        value = ConfigurationManager.get("live", "MAX_CONTRA_ALLOCATION_PATH2")
        assert value == 0.2

    def test_get_path3_signal_source_constant(self):
        """Test getting PATH3_SIGNAL_SOURCE constant."""
        value = ConfigurationManager.get("live", "PATH3_SIGNAL_SOURCE")
        assert value == "path3_full_l3_dominance"

    def test_get_all_constants(self):
        """Test getting all configuration constants."""
        constants = ConfigurationManager.get_all_constants()
        expected_keys = {"HRM_PATH_MODE", "MAX_CONTRA_ALLOCATION_PATH2", "PATH3_SIGNAL_SOURCE"}

        assert set(constants.keys()) == expected_keys
        assert constants["HRM_PATH_MODE"] == "PATH1"
        assert constants["MAX_CONTRA_ALLOCATION_PATH2"] == 0.2
        assert constants["PATH3_SIGNAL_SOURCE"] == "path3_full_l3_dominance"

    def test_readonly_constants(self):
        """Test that constants are read-only (cannot be set)."""
        # Should not be able to set constants
        result = ConfigurationManager.set("HRM_PATH_MODE", "PATH2")
        assert result is False

        # Verify still has original value
        value = ConfigurationManager.get("live", "HRM_PATH_MODE")
        assert value == "PATH1"


class TestConfigurationManagerEnvironmentConfig:
    """Test environment-specific configuration access."""

    def test_get_initial_balance_live(self):
        """Test getting initial balance for live mode."""
        balance = ConfigurationManager.get("live", "INITIAL_BALANCE")
        assert balance == 1000.0  # Live mode has different balance

    def test_get_max_position_size_testnet(self):
        """Test getting max position size for testnet mode."""
        size = ConfigurationManager.get("live", "MAX_POSITION_SIZE")  # Test with live mode for now
        assert size is not None

    def test_get_symbols_default(self):
        """Test getting default symbols."""
        symbols = ConfigurationManager.get("live", "SYMBOLS")
        assert symbols == ["BTCUSDT", "ETHUSDT"]

    def test_get_enable_commission_live(self):
        """Test getting commission setting for live mode."""
        commissions = ConfigurationManager.get("live", "ENABLE_COMMISSIONS")
        assert isinstance(commissions, bool)

    def test_set_config_value(self):
        """Test setting a configuration value."""
        # Set a custom value
        success = ConfigurationManager.set("CUSTOM_VALUE", 123.45, "live")
        assert success is True

        # Retrieve it
        value = ConfigurationManager.get("live", "CUSTOM_VALUE")
        assert value == 123.45

    def test_get_default_value(self):
        """Test getting default value when key doesn't exist."""
        value = ConfigurationManager.get("live", "NON_EXISTENT_KEY", "default_value")
        assert value == "default_value"

    def test_get_full_config(self):
        """Test getting full configuration dictionary."""
        full_config = ConfigurationManager.get("live")

        # Should be a dictionary
        assert isinstance(full_config, dict)

        # Should contain constants
        assert full_config["HRM_PATH_MODE"] == "PATH1"
        assert full_config["MAX_CONTRA_ALLOCATION_PATH2"] == 0.2

        # Should contain environment config
        assert "SYMBOLS" in full_config
        assert "INITIAL_BALANCE" in full_config
        assert "ENABLE_COMMISSIONS" in full_config


class TestConfigurationManagerSaveLoad:
    """Test configuration save/load functionality."""

    def test_save_config_to_default_path(self):
        """Test saving configuration to default path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't patch logger - the actual logging should work
            config_path = os.path.join(temp_dir, "config_live.json")
            success = ConfigurationManager.save_config("live", config_path)
            assert success is True

            # Verify file was created
            assert os.path.exists(config_path)

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "CUSTOM_KEY": "custom_value",
            "INITIAL_BALANCE": 9999.0
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = os.path.join(temp_dir, "config_live.json")
            with open(config_path, 'w') as f:
                import json
                json.dump(config_data, f)

            # Load config (logger mocking has issues, focus on functionality)
            success = ConfigurationManager.load_config("live", config_path)
            assert success is True

            # Verify loaded value
            value = ConfigurationManager.get("live", "CUSTOM_KEY")
            assert value == "custom_value"


class TestConfigurationManagerValidation:
    """Test configuration validation functionality."""

    def test_validate_config_live(self):
        """Test validating live configuration."""
        success = ConfigurationManager.validate_config("live")
        assert success is True

    @patch('core.configuration_manager.logger')
    def test_validate_config_testnet(self, mock_logger):
        """Test validating testnet configuration."""
        success = ConfigurationManager.validate_config("testnet")
        assert success is True


class TestConfigurationManagerTradingCosts:
    """Test trading cost configuration."""

    def test_get_trading_costs_live(self):
        """Test getting trading costs for live mode."""
        costs = ConfigurationManager.get_trading_costs("live")

        assert isinstance(costs, dict)
        assert "commission_rate" in costs
        assert "slippage_bps" in costs
        assert "min_order_value" in costs

        # Verify values make sense
        assert 0 < costs["commission_rate"] < 1  # Should be percentage
        assert costs["slippage_bps"] >= 0  # Should be non-negative


class TestConfigurationManagerEnvironmentDetection:
    """Test environment detection methods."""

    @patch.dict(os.environ, {"BINANCE_MODE": "LIVE"})
    def test_is_production_when_live(self):
        """Test production detection when Binance mode is LIVE."""
        result = ConfigurationManager.is_production()
        assert result is True

    @patch.dict(os.environ, {"BINANCE_MODE": "TEST"})
    def test_is_production_when_test(self):
        """Test production detection when Binance mode is TEST."""
        result = ConfigurationManager.is_production()
        assert result is False

    @patch.dict(os.environ, {"BINANCE_MODE": "TEST"})
    def test_is_testing_when_test(self):
        """Test testing detection when Binance mode is TEST."""
        result = ConfigurationManager.is_testing()
        assert result is True

    @patch.dict(os.environ, {"BINANCE_MODE": "TESTNET"})
    def test_is_testing_when_testnet(self):
        """Test testing detection when Binance mode is TESTNET."""
        result = ConfigurationManager.is_testing()
        assert result is True

    def test_is_backtesting(self):
        """Test backtesting detection (always False for now)."""
        result = ConfigurationManager.is_backtesting()
        assert result is False

    def test_get_environment_modes(self):
        """Test getting all supported environment modes."""
        modes = ConfigurationManager.get_environment_modes()
        expected_modes = {"live", "testnet", "backtest", "simulated"}

        assert modes == expected_modes


class TestConfigurationManagerIntegration:
    """Test ConfigurationManager integration with legacy patterns."""

    def test_integrates_with_legacy_get_config(self):
        """Test that ConfigurationManager integrates with legacy get_config."""
        # Use the legacy wrapper
        env_config = legacy_get_config("live")

        # Should return EnvironmentConfig instance
        assert isinstance(env_config, EnvironmentConfig)

        # Should be able to get values
        symbols = env_config.get("SYMBOLS")
        assert symbols == ["BTCUSDT", "ETHUSDT"]

    def test_get_config_value_function(self):
        """Test the get_config_value convenience function."""
        value = get_config_value("HRM_PATH_MODE")
        assert value == "PATH1"

        value = get_config_value("SYMBOLS", mode="live")
        assert value == ["BTCUSDT", "ETHUSDT"]

    def test_set_config_value_function(self):
        """Test the set_config_value convenience function."""
        # Set a custom value
        success = set_config_value("TEST_KEY", "test_value", "live")
        assert success is True

        # Verify it was set
        value = get_config_value("TEST_KEY", mode="live")
        assert value == "test_value"

    def test_backward_compatibility(self):
        """Test that old import patterns still work."""
        # This should work without breaking existing code
        from core.configuration_manager import HRM_PATH_MODE
        assert HRM_PATH_MODE == "PATH1"

        from core.configuration_manager import MAX_CONTRA_ALLOCATION_PATH2
        assert MAX_CONTRA_ALLOCATION_PATH2 == 0.2

        from core.configuration_manager import PATH3_SIGNAL_SOURCE
        assert PATH3_SIGNAL_SOURCE == "path3_full_l3_dominance"


class TestConfigurationManagerSingleton:
    """Test ConfigurationManager singleton behavior."""

    def test_singleton_instance(self):
        """Test that ConfigurationManager maintains singleton pattern."""
        instance1 = ConfigurationManager._get_instance()
        instance2 = ConfigurationManager._get_instance()

        # Should be the same instance
        assert instance1 is instance2

        # Should be of correct type
        assert isinstance(instance1, ConfigurationManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
