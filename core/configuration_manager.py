"""
HRM Configuration Manager - Single Source of Truth for All Parameters

Implements complete configuration centralization with environment-specific settings,
parameter validation, and dynamic loading.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from core.logging import logger

# Path to configurations
CONFIG_DIR = Path(__file__).parent.parent / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

@dataclass
class TradingConfig:
    """Core trading configuration"""
    symbols: list = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: list = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Balance & Position Management
    initial_balance: float = 500.0
    max_portfolio_exposure_btc: float = 0.40
    max_portfolio_exposure_eth: float = 0.40
    max_position_size_usdt: float = 1200.0
    min_usdt_reserve: float = 500.0

    # Risk Management
    risk_per_trade_percent: float = 0.02
    max_drawdown_limit: float = 0.12
    stop_loss_default_percent: float = 0.03
    take_profit_default_percent: float = 0.05

    # Trading Costs & Fees
    commission_rate: float = 0.0005  # 0.05% (reduced for testing)
    slippage_bps: float = 0.2  # 0.2 basis points (reduced for testing)
    min_order_value_usdt: float = 5.0

    # Trading Logic
    hrm_path_mode: str = "PATH2"
    max_contra_allocation_path2: float = 0.20
    signal_hold_limit: int = 30
    cycle_duration_seconds: int = 10

    # Auto-Learning
    auto_learning_enabled: bool = True
    retrain_threshold_winrate: float = 0.52
    max_models_in_ensemble: int = 10

    # HARDCORE Safety
    hardcore_mode: bool = True
    enable_stop_loss_real: bool = True
    sync_with_exchange: bool = True
    health_check_interval: int = 30

    # Sentiment Analysis
    sentiment_enabled: bool = True
    sentiment_update_interval_cycles: int = 50
    sentiment_cache_hours: int = 6

    # Model Parameters
    l1_models_threshold: float = 0.6
    l2_confidence_min: float = 0.3
    l3_regime_sensitivity: float = 0.001

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    mode: str = "simulated"
    persistence_enabled: bool = True
    commissions_enabled: bool = True
    slippage_enabled: bool = True
    websocket_enabled: bool = True

    # Environment Overrides
    balance_multiplier: float = 1.0
    risk_multiplier: float = 1.0
    fee_multiplier: float = 1.0

    # File Paths
    portfolio_state_file: str = ""
    log_file: str = ""
    cache_dir: str = "cache"

    # API & Exchange
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

@dataclass
class HRMConfig:
    """Complete HRM configuration container"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self):
        """Apply environment-specific adjustments"""
        self._apply_environment_adjustments()

    def _apply_environment_adjustments(self):
        """Apply multipliers and adjustments based on environment"""
        env = self.environment

        if env.mode != "simulated":
            self.trading.initial_balance *= env.balance_multiplier
            self.trading.risk_per_trade_percent *= env.risk_multiplier
            self.trading.commission_rate *= env.fee_multiplier

        # Set default file paths
        if not env.portfolio_state_file:
            env.portfolio_state_file = f"portfolio_state_{env.mode}.json"

        if not env.log_file:
            env.log_file = f"logs/hrm_{env.mode}.log"

class HRMConfigurationManager:
    """
    HRM Configuration Manager - Single Source of Truth

    Features:
    - Environment-specific configurations (dev/prod/live/backtest)
    - Dynamic parameter loading from YAML/JSON
    - Parameter validation and type safety
    - Environment variable overrides
    - Configuration persistence
    """

    def __init__(self, environment: str = "simulated"):
        self.environment = environment
        self.config = None
        self._load_configuration()

    def _load_configuration(self):
        """Load and merge configuration"""
        self.config = HRMConfig()

        # Apply environment settings first
        self._apply_environment_profile()

        # Load from file if exists
        self._load_from_file()

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Validate configuration
        self._validate_config()

        logger.info(f"✅ HRM Configuration loaded for environment: {self.environment}")
        logger.info(f"   Trading symbols: {self.config.trading.symbols}")
        logger.info(f"   Initial balance: {self.config.trading.initial_balance} USDT")
        logger.info(f"   Risk per trade: {self.config.trading.risk_per_trade_percent:.1%}")

    def _apply_environment_profile(self):
        """Apply environment-specific profiles"""
        env = self.config.environment

        if self.environment == "live":
            env.persistence_enabled = True
            env.commissions_enabled = True
            env.slippage_enabled = True
            env.binance_testnet = False
            env.balance_multiplier = 1.0/3.0  # Conservative
            env.risk_multiplier = 0.8  # Conservative

        elif self.environment == "testnet":
            env.persistence_enabled = True
            env.commissions_enabled = True
            env.slippage_enabled = True
            env.binance_testnet = True
            env.balance_multiplier = 1.0/3.0
            env.fee_multiplier = 1.5  # Higher testnet fees

        elif self.environment == "dev":
            env.persistence_enabled = False
            env.binance_testnet = True
            env.risk_multiplier = 0.1  # Ultra conservative
            env.mode = "dev"

        elif self.environment == "backtest":
            env.persistence_enabled = False
            env.commissions_enabled = True
            env.slippage_enabled = True
            env.binance_testnet = True
            env.risk_multiplier = 2.0  # Can be more aggressive
            env.fee_multiplier = 1.2  # Historical fees
            env.mode = "backtest"

        # Simulated is the default, already set

    def _load_from_file(self):
        """Load configuration from YAML/JSON files"""
        config_file = CONFIG_DIR / f"config_{self.environment}.yaml"
        json_config_file = CONFIG_DIR / f"config_{self.environment}.json"

        config_data = None

        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        elif json_config_file.exists():
            with open(json_config_file, 'r') as f:
                config_data = json.load(f)

        if config_data:
            self._apply_config_data(config_data)
            logger.info(f"📂 Configuration loaded from {config_file if config_file.exists() else json_config_file}")

    def _apply_config_data(self, data: Dict[str, Any]):
        """Apply configuration data to config object"""
        if "trading" in data:
            for key, value in data["trading"].items():
                if hasattr(self.config.trading, key):
                    setattr(self.config.trading, key, value)

        if "environment" in data:
            for key, value in data["environment"].items():
                if hasattr(self.config.environment, key):
                    setattr(self.config.environment, key, value)

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Trading config overrides
        overrides = {
            'INITIAL_BALANCE': ('trading.initial_balance', float),
            'RISK_PER_TRADE': ('trading.risk_per_trade_percent', float),
            'COMMISSION_RATE': ('trading.commission_rate', float),
            'SLIPPAGE_BPS': ('trading.slippage_bps', int),
            'HRM_PATH_MODE': ('trading.hrm_path_mode', str),
            'AUTO_LEARNING_ENABLED': ('trading.auto_learning_enabled', lambda x: x.lower() in ('true', '1', 'yes')),
            'HARDCORE_MODE': ('trading.hardcore_mode', lambda x: x.lower() in ('true', '1', 'yes')),
            'SENTIMENT_ENABLED': ('trading.sentiment_enabled', lambda x: x.lower() in ('true', '1', 'yes')),
        }

        for env_var, (config_path, converter) in overrides.items():
            if value := os.getenv(env_var):
                try:
                    attr_path = config_path.split('.')
                    obj = self.config
                    for attr in attr_path[:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, attr_path[-1], converter(value))
                    logger.info(f"🔄 Override applied: {env_var} = {value}")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid override {env_var}={value}: {e}")

        # API credentials
        if api_key := os.getenv("BINANCE_API_KEY"):
            self.config.environment.binance_api_key = api_key
        if api_secret := os.getenv("BINANCE_API_SECRET"):
            self.config.environment.binance_api_secret = api_secret

    def _validate_config(self) -> bool:
        """Validate configuration parameters"""
        errors = []

        # Trading config validation
        t = self.config.trading

        if t.initial_balance <= 0:
            errors.append("Initial balance must be positive")

        if not (0 < t.risk_per_trade_percent <= 0.1):
            errors.append("Risk per trade must be between 0% and 10%")

        if not (0 < t.commission_rate <= 0.01):
            errors.append("Commission rate must be reasonable (0-1%)")

        if t.hrm_path_mode not in ["PATH1", "PATH2", "PATH3"]:
            errors.append("HRM path mode must be PATH1, PATH2, or PATH3")

        if len(t.symbols) == 0:
            errors.append("At least one trading symbol required")

        if errors:
            for error in errors:
                logger.error(f"❌ Configuration error: {error}")
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")

        logger.info("✅ Configuration validation passed")
        return True

    def get(self, key: str, default=None):
        """Get configuration value by dotted path (e.g., 'trading.initial_balance')"""
        keys = key.split('.')
        obj = self.config
        try:
            for k in keys:
                obj = getattr(obj, k)
            return obj
        except AttributeError:
            return default

    def set(self, key: str, value: Any):
        """Set configuration value by dotted path"""
        keys = key.split('.')
        obj = self.config
        for k in keys[:-1]:
            if not hasattr(obj, k):
                setattr(obj, k, type('TempObj', (), {})())
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)
        logger.info(f"🔧 Config set: {key} = {value}")

    def save_config(self, filepath: Optional[str] = None):
        """Save current configuration to file"""
        if not filepath:
            filepath = CONFIG_DIR / f"config_{self.environment}.yaml"

        config_dict = asdict(self.config)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"💾 Configuration saved to {filepath}")

    def __repr__(self) -> str:
        return f"HRMConfigurationManager(environment='{self.environment}')"

# Global configuration instance
_config_manager = None

def get_config_manager(env: str = "simulated") -> HRMConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None or _config_manager.environment != env:
        _config_manager = HRMConfigurationManager(env)
    return _config_manager

def get_config_value(key: str, default=None, env: str = "simulated"):
    """Get configuration value by key"""
    return get_config_manager(env).get(key, default)

def set_config_value(key: str, value: Any, env: str = "simulated"):
    """Set configuration value by key"""
    get_config_manager(env).set(key, value)

# Convenience constants
DEFAULT_CONFIG = HRMConfig()

# Example usage:
#
# # Get trading config
# from core.configuration_manager import get_config_manager
# config = get_config_manager('live')
# balance = config.get('trading.initial_balance')
#
# # Get specific value
# symbols = get_config_value('trading.symbols', ['BTCUSDT'])
#
# # Set value
# set_config_value('trading.risk_per_trade_percent', 0.025)
