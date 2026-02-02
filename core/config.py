# core/config.py - Environment Configuration Management
import os
import json
from typing import Dict, Any
from core.logging import logger

# HRM Path Mode Configuration
HRM_PATH_MODE = "PATH2"  # opciones: PATH1, PATH2, PATH3 (PATH2 = HYBRID INTELLIGENT - BALANCED MULTI-SIGNAL)
MAX_CONTRA_ALLOCATION_PATH2 = 0.2  # 20% limit for contra-allocation in PATH2

# Signal source constants for PATH mode validation
PATH3_SIGNAL_SOURCE = "path3_full_l3_dominance"  # Required signal source for PATH3 orders

class EnvironmentConfig:
    """
    Configuration management for different HRM environments:
    - live: Production trading with real money
    - testnet: Testing with Binance testnet
    - backtest: Historical backtesting
    - simulated: Simulation with commissions and slippage
    """

    def __init__(self, mode: str = "live"):
        self.mode = mode
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration based on environment mode"""
        base_config = {
            "SYMBOLS": ["BTCUSDT", "ETHUSDT"],
            "TIMEFRAME": "1m",
            "INITIAL_BALANCE": 3000.0,
            "ENABLE_COMMISSIONS": True,
            "ENABLE_SLIPPAGE": True,
            "ENABLE_PERSISTENCE": True,
            "ENABLE_LOGGING": True,
            "MAX_POSITION_SIZE": 0.05,  # 5% max position per symbol
            "MIN_ORDER_VALUE": 1.0,     # Minimum order value in USDT
            "RISK_PER_TRADE": 0.02,    # 2% risk per trade
        }

        # Environment-specific overrides
        if self.mode == "live":
            env_config = {
                "INITIAL_BALANCE": 1000.0,  # Conservative starting balance
                "ENABLE_COMMISSIONS": True,
                "ENABLE_SLIPPAGE": True,
                "ENABLE_PERSISTENCE": True,
                "STATE_FILE": "portfolio_state_live.json",
                "LOG_FILE": "logs/hrm_live.log",
                "MAX_POSITION_SIZE": 0.03,  # More conservative in live
                "COMMISSION_RATE": 0.001,  # 0.1% maker/taker
                "SLIPPAGE_BPS": 2,         # 2 basis points
            }
        elif self.mode == "testnet":
            env_config = {
                "INITIAL_BALANCE": 1000.0,
                "ENABLE_COMMISSIONS": True,
                "ENABLE_SLIPPAGE": True,
                "ENABLE_PERSISTENCE": True,
                "STATE_FILE": "portfolio_state_testnet.json",
                "LOG_FILE": "logs/hrm_testnet.log",
                "MAX_POSITION_SIZE": 0.05,
                "COMMISSION_RATE": 0.0015,  # Higher fees for testing
                "SLIPPAGE_BPS": 5,
            }
        elif self.mode == "backtest":
            env_config = {
                "INITIAL_BALANCE": 10000.0,  # Higher for backtesting
                "ENABLE_COMMISSIONS": True,
                "ENABLE_SLIPPAGE": True,
                "ENABLE_PERSISTENCE": False,  # No persistence in backtest
                "STATE_FILE": "portfolio_state_backtest.json",
                "LOG_FILE": "logs/hrm_backtest.log",
                "MAX_POSITION_SIZE": 0.10,  # Higher limits for backtest
                "COMMISSION_RATE": 0.0012,  # Historical average
                "SLIPPAGE_BPS": 3,
            }
        elif self.mode == "simulated":
            env_config = {
                "INITIAL_BALANCE": 10000.0,
                "ENABLE_COMMISSIONS": True,   # Enable for realistic simulation
                "ENABLE_SLIPPAGE": True,     # Enable for realistic simulation
                "ENABLE_PERSISTENCE": False, # No persistence
                "STATE_FILE": "portfolio_state_simulated.json",
                "LOG_FILE": "logs/hrm_simulated.log",
                "MAX_POSITION_SIZE": 0.08,
                "COMMISSION_RATE": 0.002,    # Higher for conservative testing
                "SLIPPAGE_BPS": 10,          # Higher slippage for testing
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Merge configurations
        config = {**base_config, **env_config}

        # Load from environment variables if available
        config.update(self._load_env_vars())

        logger.info(f"✅ Environment config loaded for mode: {self.mode}")
        logger.info(f"   Balance: {config['INITIAL_BALANCE']} USDT")
        logger.info(f"   Commissions: {'Enabled' if config['ENABLE_COMMISSIONS'] else 'Disabled'}")
        logger.info(f"   Slippage: {'Enabled' if config['ENABLE_SLIPPAGE'] else 'Disabled'}")
        logger.info(f"   Persistence: {'Enabled' if config['ENABLE_PERSISTENCE'] else 'Disabled'}")

        return config

    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}

        # Balance override
        if balance := os.getenv("HRM_INITIAL_BALANCE"):
            try:
                env_config["INITIAL_BALANCE"] = float(balance)
            except ValueError:
                logger.warning(f"Invalid HRM_INITIAL_BALANCE: {balance}")

        # Commission override
        if commissions := os.getenv("HRM_ENABLE_COMMISSIONS"):
            env_config["ENABLE_COMMISSIONS"] = commissions.lower() in ("true", "1", "yes")

        # Slippage override
        if slippage := os.getenv("HRM_ENABLE_SLIPPAGE"):
            env_config["ENABLE_SLIPPAGE"] = slippage.lower() in ("true", "1", "yes")

        return env_config

    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

    def save_to_file(self, filepath: str = None):
        """Save current configuration to file"""
        if filepath is None:
            filepath = f"config_{self.mode}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"💾 Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")

    def load_from_file(self, filepath: str = None):
        """Load configuration from file"""
        if filepath is None:
            filepath = f"config_{self.mode}.json"

        if not os.path.exists(filepath):
            logger.warning(f"Config file not found: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                file_config = json.load(f)
            self.config.update(file_config)
            logger.info(f"📂 Configuration loaded from {filepath}")
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")

    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = ["SYMBOLS", "INITIAL_BALANCE", "MAX_POSITION_SIZE"]
        missing = [key for key in required_keys if key not in self.config]

        if missing:
            logger.error(f"❌ Missing required config keys: {missing}")
            return False

        # Validate balance
        if self.config["INITIAL_BALANCE"] <= 0:
            logger.error("❌ Initial balance must be positive")
            return False

        # Validate position size
        if not 0 < self.config["MAX_POSITION_SIZE"] <= 1:
            logger.error("❌ Max position size must be between 0 and 1")
            return False

        logger.info("✅ Configuration validation passed")
        return True

    def get_trading_costs(self) -> Dict[str, float]:
        """Get trading cost configuration"""
        return {
            "commission_rate": self.config.get("COMMISSION_RATE", 0.001),
            "slippage_bps": self.config.get("SLIPPAGE_BPS", 2),
            "min_order_value": self.config.get("MIN_ORDER_VALUE", 1.0),
        }

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.mode == "live"

    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.mode in ["testnet", "simulated"]

    def is_backtesting(self) -> bool:
        """Check if running in backtesting mode"""
        return self.mode == "backtest"

# ============================================================================
# HRM SIGNAL CONFIDENCE THRESHOLDS - REDUCED FOR MORE TRADING OPPORTUNITIES
# ============================================================================

# ❌ BEFORE (too restrictive - blocking signals)
MIN_SIGNAL_CONFIDENCE_OLD = 0.50  # 50% minimum
MIN_L2_CONFIDENCE_OLD = 0.60      # 60% for L2-only signals

# ✅ AFTER (more permissive - allowing legitimate signals)
MIN_SIGNAL_CONFIDENCE = 0.35      # 35% minimum (allows more L1/L2 signals)
MIN_L2_CONFIDENCE = 0.40          # 40% for L2 signals (more flexible)

# ============================================================================
# SIGNAL VERIFICATION CONFIG - LESS RESTRICTIVE
# ============================================================================
SIGNAL_VERIFICATION_CONFIG = {
    'min_confidence': 0.35,         # ✅ Reduced from 0.50
    'min_l3_confidence': 0.45,      # ✅ Reduced from 0.55
    'min_convergence': 0.30,        # ✅ Reduced from 0.40
    'allow_hold_signals': True,     # ✅ Allow HOLD when L3 says BUY
    'l3_override_threshold': 0.55   # ✅ L3 can force signals if > 55%
}

# ============================================================================
# SIGNAL COMPOSER CONFIG - BALANCED WEIGHTS
# ============================================================================
SIGNAL_COMPOSER_CONFIG = {
    'hold_confidence_boost': 0.15,  # ✅ Amplify HOLD confidence +15%
    'l3_signal_weight': 0.60,       # ✅ Higher L3 weight (60%)
    'l2_technical_weight': 0.40,    # ✅ Lower technical weight (40%)
    'require_l1_l2_agreement': False # ✅ No L1+L2 agreement required
}

# Logging for confidence adjustments
logger.info(f"✅ Confidence thresholds adjusted: min={MIN_SIGNAL_CONFIDENCE}, L2={MIN_L2_CONFIDENCE}")

# Global configuration instance
_config_instance = None

def get_config(mode: str = "live") -> EnvironmentConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None or _config_instance.mode != mode:
        _config_instance = EnvironmentConfig(mode)
    return _config_instance

# Convenience functions
def get_config_value(key: str, default=None, mode: str = "live"):
    """Get configuration value"""
    return get_config(mode).get(key, default)

def set_config_value(key: str, value: Any, mode: str = "live"):
    """Set configuration value"""
    get_config(mode).set(key, value)
