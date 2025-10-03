# -*- coding: utf-8 -*-
"""
Unified Configuration Management System - HRM Trading System

Centralizes all configuration access patterns across the system to eliminate
inconsistent direct imports and function calls. Provides a single, consistent
interface for configuration management.
"""

import os
import json
from typing import Dict, Any, Optional, Union, Set
from core.logging import logger
from core.config import EnvironmentConfig, get_config as legacy_get_config


class ConfigurationManager:
    """
    Unified configuration management system for HRM trading system.
    Provides consistent access to all configuration values and settings.
    """

    # HRM Path Mode Configuration
    HRM_PATH_MODE = "PATH2"  # opciones: PATH1, PATH2, PATH3 (PATH2 = HYBRID INTELLIGENT - BALANCED MULTI-SIGNAL)
    MAX_CONTRA_ALLOCATION_PATH2 = 0.2  # 20% limit for contra-allocation in PATH2

    # Signal source constants for PATH mode validation
    PATH3_SIGNAL_SOURCE = "path3_full_l3_dominance"  # Required signal source for PATH3 orders

    def __init__(self):
        """Initialize configuration manager with all environment configs."""
        self._env_configs: Dict[str, EnvironmentConfig] = {}
        self._cache: Dict[str, Any] = {}
        self._common_defaults = {
            # Trading parameters
            "SYMBOLS": ["BTCUSDT", "ETHUSDT"],
            "TIMEFRAME": "1m",
            "MAX_POSITION_SIZE": 0.05,
            "MIN_ORDER_VALUE": 1.0,
            "RISK_PER_TRADE": 0.02,

            # Feature flags
            "ENABLE_COMMISSIONS": True,
            "ENABLE_SLIPPAGE": True,
            "ENABLE_PERSISTENCE": True,
            "ENABLE_LOGGING": True,

            # Trading costs
            "COMMISSION_RATE": 0.001,  # 0.1%
            "SLIPPAGE_BPS": 2,  # 2 basis points

            # File paths
            "LOG_FILE": "logs/hrm.log",
            "STATE_FILE": "portfolio_state.json",
        }

    @staticmethod
    def get(mode: str = "live", key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value(s) with unified interface.

        Args:
            mode: Environment mode ("live", "testnet", "backtest", "simulated")
            key: Specific configuration key (None returns full config dict)
            default: Default value if key not found

        Returns:
            Configuration value or full config dict if key is None
        """
        manager = ConfigurationManager._get_instance()

        # Handle specific config keys first (constants and common values)
        if key is not None:
            config_value = manager._get_special_config(key, mode, default)
            if config_value is not None:
                return config_value

        # Use legacy config system for environment-specific config
        env_config = manager._get_env_config(mode)

        if key is None:
            # Return full configuration dict
            full_config = {
                # Add constants
                "HRM_PATH_MODE": manager.HRM_PATH_MODE,
                "MAX_CONTRA_ALLOCATION_PATH2": manager.MAX_CONTRA_ALLOCATION_PATH2,
                "PATH3_SIGNAL_SOURCE": manager.PATH3_SIGNAL_SOURCE,
            }
            # Add environment config
            full_config.update(env_config.config)
            return full_config
        else:
            return env_config.get(key, default)

    @staticmethod
    def set(key: str, value: Any, mode: str = "live") -> bool:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
            mode: Environment mode

        Returns:
            Success status
        """
        manager = ConfigurationManager._get_instance()

        # Handle special config keys that are constants
        if key in ["HRM_PATH_MODE", "MAX_CONTRA_ALLOCATION_PATH2", "PATH3_SIGNAL_SOURCE"]:
            return False  # These are read-only constants

        # Use environment config for dynamic values
        try:
            env_config = manager._get_env_config(mode)
            env_config.set(key, value)
            manager._cache_clear()
            return True
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False

    @staticmethod
    def save_config(mode: str = "live", filepath: Optional[str] = None) -> bool:
        """
        Save configuration to file.

        Args:
            mode: Environment mode
            filepath: Optional custom filepath

        Returns:
            Success status
        """
        try:
            manager = ConfigurationManager._get_instance()
            env_config = manager._get_env_config(mode)

            if filepath is None:
                filepath = f"config_{mode}.json"

            env_config.save_to_file(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save config for mode {mode}: {e}")
            return False

    @staticmethod
    def load_config(mode: str = "live", filepath: Optional[str] = None) -> bool:
        """
        Load configuration from file.

        Args:
            mode: Environment mode
            filepath: Optional custom filepath

        Returns:
            Success status
        """
        try:
            manager = ConfigurationManager._get_instance()
            env_config = manager._get_env_config(mode)

            if filepath is None:
                filepath = f"config_{mode}.json"

            env_config.load_from_file(filepath)
            manager._cache_clear()
            return True
        except Exception as e:
            logger.error(f"Failed to load config for mode {mode}: {e}")
            return False

    @staticmethod
    def validate_config(mode: str = "live") -> bool:
        """
        Validate configuration for specified mode.

        Args:
            mode: Environment mode

        Returns:
            Validation status
        """
        try:
            manager = ConfigurationManager._get_instance()
            env_config = manager._get_env_config(mode)
            return env_config.validate()
        except Exception as e:
            logger.error(f"Config validation failed for mode {mode}: {e}")
            return False

    @staticmethod
    def get_trading_costs(mode: str = "live") -> Dict[str, float]:
        """
        Get trading cost configuration.

        Args:
            mode: Environment mode

        Returns:
            Trading costs dict
        """
        manager = ConfigurationManager._get_instance()
        env_config = manager._get_env_config(mode)
        return env_config.get_trading_costs()

    @staticmethod
    def is_production() -> bool:
        """Check if currently in production mode."""
        config = ConfigurationManager.get("live", "ENABLE_LOGGING", True)
        return os.getenv("BINANCE_MODE", "TEST").upper() == "LIVE"

    @staticmethod
    def is_testing() -> bool:
        """Check if currently in testing mode."""
        config = ConfigurationManager.get("live", "ENABLE_LOGGING", True)
        mode = os.getenv("BINANCE_MODE", "TEST").upper()
        return mode in ["TEST", "TESTNET"]

    @staticmethod
    def is_backtesting() -> bool:
        """Check if currently in backtesting mode."""
        return False  # Would need more context to determine

    @staticmethod
    def get_all_constants() -> Dict[str, Any]:
        """Get all configuration constants."""
        manager = ConfigurationManager._get_instance()
        return {
            "HRM_PATH_MODE": manager.HRM_PATH_MODE,
            "MAX_CONTRA_ALLOCATION_PATH2": manager.MAX_CONTRA_ALLOCATION_PATH2,
            "PATH3_SIGNAL_SOURCE": manager.PATH3_SIGNAL_SOURCE,
        }

    @staticmethod
    def get_environment_modes() -> Set[str]:
        """Get all supported environment modes."""
        return {"live", "testnet", "backtest", "simulated"}

    def _get_special_config(self, key: str, mode: str, default: Any) -> Any:
        """Get special configuration keys (constants)."""
        constants = {
            "HRM_PATH_MODE": self.HRM_PATH_MODE,
            "MAX_CONTRA_ALLOCATION_PATH2": self.MAX_CONTRA_ALLOCATION_PATH2,
            "PATH3_SIGNAL_SOURCE": self.PATH3_SIGNAL_SOURCE,
        }

        if key in constants:
            return constants[key]

        return None

    def _get_env_config(self, mode: str) -> EnvironmentConfig:
        """Get or create environment configuration instance."""
        if mode not in self._env_configs:
            self._env_configs[mode] = legacy_get_config(mode)
        return self._env_configs[mode]

    def _cache_clear(self):
        """Clear configuration cache."""
        self._cache.clear()

    @staticmethod
    def _get_instance() -> 'ConfigurationManager':
        """Get singleton instance of ConfigurationManager."""
        if not hasattr(ConfigurationManager, '_instance'):
            ConfigurationManager._instance = ConfigurationManager()
        return ConfigurationManager._instance


# Backward compatibility aliases and wrapper functions
def get_config(mode: str = "live") -> EnvironmentConfig:
    """Backward compatibility wrapper for legacy get_config calls."""
    return legacy_get_config(mode)

# Global configuration access - use ConfigurationManager
def get_config_value(key: str, default=None, mode: str = "live"):
    """Get configuration value with unified interface."""
    return ConfigurationManager.get(mode, key, default)

def set_config_value(key: str, value: Any, mode: str = "live"):
    """Set configuration value with unified interface."""
    return ConfigurationManager.set(key, value, mode)

# Expose constants for direct import compatibility (while discouraging it)
__all__ = [
    'ConfigurationManager',
    'get_config',          # Legacy wrapper
    'get_config_value',    # Unified interface
    'set_config_value',    # Unified interface
]

# Direct constants for backward compatibility (discouraged but functional)
HRM_PATH_MODE = ConfigurationManager.HRM_PATH_MODE
MAX_CONTRA_ALLOCATION_PATH2 = ConfigurationManager.MAX_CONTRA_ALLOCATION_PATH2
PATH3_SIGNAL_SOURCE = ConfigurationManager.PATH3_SIGNAL_SOURCE
