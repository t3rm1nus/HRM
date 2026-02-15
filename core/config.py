# core/config.py - Environment Configuration Management
"""
HRM Configuration - Strongly Typed Configuration Object

✅ USA config.paper_mode EN LUGAR DE config['paper_mode']
✅ Type safety con dataclasses
✅ Autocompletado en IDE
"""

# =============================================================================
# CRITICAL SYSTEM MODE CONFIGURATION - FORZAR MODO PAPER
# =============================================================================
# ⚠️ IMPORTANTE: Este sistema SOLO opera en modo paper/simulado
# NO modificar a 'live' - Solo trading simulado permitido
SYSTEM_MODE = 'paper'  # NO usar 'live' - Solo 'paper' o 'simulated'
PAPER_MODE = True      # Siempre True para forzar modo paper

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from core.logging import logger

# =============================================================================
# HRM PATH MODE CONFIGURATION
# =============================================================================
HRM_PATH_MODE = "PATH2"  # opciones: PATH1, PATH2, PATH3
MAX_CONTRA_ALLOCATION_PATH2 = 0.2  # 20% limit for contra-allocation in PATH2
PATH3_SIGNAL_SOURCE = "path3_full_l3_dominance"  # Required signal source for PATH3 orders

# =============================================================================
# HRM CONFIG - ESTRUCTURA FUERTEMENTE TIPADA
# =============================================================================

@dataclass
class BinanceConfig:
    """Configuración de Binance"""
    API_KEY: str = ""
    API_SECRET: str = ""
    MODE: str = "PAPER"  # PAPER, LIVE
    USE_TESTNET: bool = True

@dataclass
class TradingConfig:
    """Configuración de trading"""
    SYMBOLS: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    TIMEFRAME: str = "1m"
    INITIAL_BALANCE: float = 3000.0
    MAX_POSITION_SIZE: float = 0.05  # 5% max position per symbol
    MIN_ORDER_VALUE: float = 1.0
    RISK_PER_TRADE: float = 0.02
    PAPER_MODE: bool = True
    
    # Costs
    ENABLE_COMMISSIONS: bool = True
    ENABLE_SLIPPAGE: bool = True
    COMMISSION_RATE: float = 0.001
    SLIPPAGE_BPS: float = 2
    
    # Persistence
    ENABLE_PERSISTENCE: bool = True
    STATE_FILE: str = "portfolio_state.json"
    LOG_FILE: str = "logs/hrm.log"

@dataclass
class ConvergenceConfig:
    """Configuración de convergence"""
    enabled: bool = True
    rollout_phase: str = "monitoring_only"
    safety_mode: str = "conservative"
    confidence_min: float = 0.35
    confidence_l2: float = 0.40

@dataclass
class HRMConfig:
    """
    Configuración principal de HRM - FUERTEMENTE TIPADA
    
    ✅ USA: config.paper_mode
    ❌ NO USES: config['paper_mode']
    """
    trading: TradingConfig = field(default_factory=TradingConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    
    # Mode
    mode: str = "simulated"
    
    def get(self, key: str, default=None) -> Any:
        """Get config value by dotted path (legacy compatibility)"""
        keys = key.split('.')
        obj = self
        try:
            for k in keys:
                obj = getattr(obj, k)
            return obj
        except AttributeError:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Legacy dict-like access for compatibility"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Legacy dict-like set for compatibility"""
        if '.' in key:
            parts = key.split('.')
            obj = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
        else:
            setattr(self, key, value)

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
            "PAPER_MODE": True,        # Paper mode enabled by default
            "BOOTSTRAP_ENABLED": True,  # Bootstrap functionality enabled by default
            "BOOTSTRAP_MIN_EXPOSURE": 0.10,  # Minimum 10% exposure for initial bootstrap
            "BOOTSTRAP_MAX_EXPOSURE": 0.30,  # Maximum 30% exposure for initial bootstrap
            "BOOTSTRAP_MIN_ORDER_VALUE": 10.0,  # Minimum $10 per bootstrap order
            "SIMULATED_INITIAL_BALANCES": {
                "BTC": 0.01549,
                "ETH": 0.385,
                "USDT": 3000.0
            }
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

        # Paper mode override
        if paper_mode := os.getenv("HRM_PAPER_MODE"):
            env_config["PAPER_MODE"] = paper_mode.lower() in ("true", "1", "yes")

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
# TEMPORARY AGGRESSIVE MODE CONFIG
# ============================================================================
TEMPORARY_AGGRESSIVE_MODE = False  # Default: False (conservative mode)
TEMPORARY_AGGRESSIVE_MODE_START_TIME = None
TEMPORARY_AGGRESSIVE_MODE_DURATION = 300  # 5 minutes in seconds
TEMPORARY_AGGRESSIVE_MODE_CYCLES = None
TEMPORARY_AGGRESSIVE_MODE_MAX_CYCLES = 100  # Maximum 100 cycles

def enable_temporary_aggressive_mode(duration_seconds: int = 300, max_cycles: int = 100):
    """Enable temporary aggressive mode with duration and cycle limits."""
    import sys
    module = sys.modules[__name__]
    module.TEMPORARY_AGGRESSIVE_MODE = True
    module.TEMPORARY_AGGRESSIVE_MODE_START_TIME = datetime.now()
    module.TEMPORARY_AGGRESSIVE_MODE_CYCLES = 0
    module.TEMPORARY_AGGRESSIVE_MODE_DURATION = duration_seconds
    module.TEMPORARY_AGGRESSIVE_MODE_MAX_CYCLES = max_cycles
    logger.warning(f"🔥 TEMPORARY AGGRESSIVE MODE ENABLED (duration: {duration_seconds}s, max cycles: {max_cycles})")

def disable_temporary_aggressive_mode():
    """Disable temporary aggressive mode."""
    import sys
    module = sys.modules[__name__]
    module.TEMPORARY_AGGRESSIVE_MODE = False
    module.TEMPORARY_AGGRESSIVE_MODE_START_TIME = None
    module.TEMPORARY_AGGRESSIVE_MODE_CYCLES = None
    logger.info("🧯 TEMPORARY AGGRESSIVE MODE DISABLED")

def check_temporary_aggressive_mode():
    """Check if temporary aggressive mode should be disabled (time or cycle limit reached)."""
    import sys
    module = sys.modules[__name__]
    
    if not module.TEMPORARY_AGGRESSIVE_MODE:
        return False
        
    # Check time limit
    if module.TEMPORARY_AGGRESSIVE_MODE_START_TIME:
        elapsed = (datetime.now() - module.TEMPORARY_AGGRESSIVE_MODE_START_TIME).total_seconds()
        if elapsed >= module.TEMPORARY_AGGRESSIVE_MODE_DURATION:
            logger.warning(f"⏰ TEMPORARY AGGRESSIVE MODE EXPIRED (time limit reached: {elapsed:.0f}s)")
            disable_temporary_aggressive_mode()
            return False
    
    # Check cycle limit
    if module.TEMPORARY_AGGRESSIVE_MODE_CYCLES is not None:
        if module.TEMPORARY_AGGRESSIVE_MODE_CYCLES >= module.TEMPORARY_AGGRESSIVE_MODE_MAX_CYCLES:
            logger.warning(f"🔄 TEMPORARY AGGRESSIVE MODE EXPIRED (cycle limit reached: {module.TEMPORARY_AGGRESSIVE_MODE_CYCLES})")
            disable_temporary_aggressive_mode()
            return False
        module.TEMPORARY_AGGRESSIVE_MODE_CYCLES += 1
    
    return True

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

# Log temporary aggressive mode status initially
if TEMPORARY_AGGRESSIVE_MODE:
    logger.warning("🔥 TEMPORARY AGGRESSIVE MODE ENABLED")
else:
    logger.debug("🧯 TEMPORARY AGGRESSIVE MODE DISABLED")

# Global configuration instance
_config_instance = None
# Force mode override - set by bootstrap to force specific mode
_forced_mode: str = None

def set_forced_mode(mode: str):
    """Force a specific mode - used by bootstrap only"""
    global _forced_mode
    _forced_mode = mode
    logger.info(f"🎯 MODE FORCED by bootstrap to: {mode}")

def get_config(mode: str = None) -> EnvironmentConfig:
    """
    Get global configuration instance.
    
    IMPORTANT: This function should ONLY be called from bootstrap.
    All other components should receive 'mode' via constructor injection.
    
    The mode is determined by:
    1. _forced_mode (set by bootstrap via set_forced_mode)
    2. mode parameter passed to this function
    3. Default to "paper" if neither is set
    """
    global _config_instance, _forced_mode
    
    # Priority 1: Use forced mode from bootstrap if set
    if _forced_mode is not None:
        effective_mode = _forced_mode
    elif mode is not None:
        effective_mode = mode
    else:
        effective_mode = "paper"  # Default to paper
    
    if _config_instance is None or _config_instance.mode != effective_mode:
        _config_instance = EnvironmentConfig(effective_mode)
    return _config_instance

# Convenience functions
def get_config_value(key: str, default=None, mode: str = "live"):
    """Get configuration value"""
    return get_config(mode).get(key, default)

def set_config_value(key: str, value: Any, mode: str = "live"):
    """Set configuration value"""
    get_config(mode).set(key, value)
