"""
Weight Calculator Configuration

Centralized configuration for all weight calculator parameters,
thresholds, and safety controls.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from core.logging import logger


@dataclass
class WeightCalculatorConfig:
    """Configuration for weight calculator parameters"""

    # Core weighting parameters
    default_strategy: str = "equal_weight"
    risk_appetite: str = "moderate"

    # Risk appetite multipliers
    risk_multipliers: Dict[str, float] = None

    # Strategy-specific parameters
    volatility_target: float = 0.15  # 15% target volatility
    max_sharpe_lookback: int = 252  # 1 year lookback for Sharpe

    # Constraints
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_concentration: float = 0.3  # 30% max per asset
    min_assets: int = 1
    max_assets: int = 10

    # Risk parameters
    risk_free_rate: float = 0.02
    confidence_level_var: float = 0.95
    confidence_level_es: float = 0.95

    # Circuit breakers
    circuit_breakers: Dict[str, Any] = None

    # Gradual rollout settings
    gradual_rollout: Dict[str, Any] = None

    # Dynamic calibration
    calibration: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.risk_multipliers is None:
            self.risk_multipliers = {
                "low": 0.7,
                "moderate": 1.0,
                "high": 1.3,
                "aggressive": 1.6
            }

        if self.circuit_breakers is None:
            self.circuit_breakers = {
                "enabled": True,
                "max_volatility": 0.50,  # 50% max volatility
                "max_correlation": 0.95,  # 95% max correlation
                "min_diversification": 0.1,  # Minimum diversification ratio
                "max_concentration": 0.5,  # 50% max concentration
                "emergency_stop": {
                    "enabled": True,
                    "volatility_threshold": 0.75,  # 75% emergency stop
                    "correlation_threshold": 0.98,  # 98% emergency stop
                    "concentration_threshold": 0.7  # 70% emergency stop
                }
            }

        if self.gradual_rollout is None:
            self.gradual_rollout = {
                "enabled": True,
                "initial_operations": 10,
                "conservative_multipliers": {
                    "volatility_target": 0.85,  # More conservative initially
                    "max_concentration": 0.8,   # More conservative initially
                    "risk_multiplier": 0.9      # More conservative initially
                }
            }

        if self.calibration is None:
            self.calibration = {
                "enabled": True,
                "performance_window": 100,
                "adjustment_step": 0.05,
                "min_threshold": 0.50,
                "max_threshold": 0.95,
                "auto_adjust": True
            }


@dataclass
class CorrelationSizerConfig:
    """Configuration for correlation position sizer"""

    # Core parameters
    max_correlation_threshold: float = 0.8
    correlation_penalty_factor: float = 0.7

    # Risk thresholds
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.6

    # Circuit breakers
    circuit_breakers: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values"""
        if self.circuit_breakers is None:
            self.circuit_breakers = {
                "enabled": True,
                "correlation_emergency": 0.95,
                "volatility_emergency": 0.60,
                "diversification_emergency": 0.05
            }


@dataclass
class PortfolioRebalancerConfig:
    """Configuration for portfolio rebalancer"""

    # Rebalancing parameters
    drift_threshold: float = 0.15  # 15% drift threshold (increased from 5%)
    transaction_costs: float = 0.001  # 0.1% transaction costs
    min_trade_value: float = 10.0  # $10 minimum trade

    # Trigger settings
    calendar_rebalance_days: int = 30
    volatility_change_threshold: float = 0.2  # 20% volatility change
    correlation_change_threshold: float = 0.1  # 10% correlation change

    # Circuit breakers
    circuit_breakers: Dict[str, Any] = None

    # Minimum position size protection
    minimum_position_size: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values"""
        if self.circuit_breakers is None:
            self.circuit_breakers = {
                "enabled": True,
                "max_trade_ratio": 0.25,  # Max 25% of portfolio in one trade
                "max_daily_trades": 5,
                "cost_threshold": 0.005  # 0.5% max cost ratio
            }

        if self.minimum_position_size is None:
            self.minimum_position_size = {
                "enabled": True,
                "min_portfolio_percentage": 0.10,  # Minimum 10% of portfolio
                "exemption_list": []  # Assets exempt from minimum size rule
            }


class WeightCalculatorConfigManager:
    """Manager for weight calculator configuration"""

    def __init__(self, config_file: str = "core/weight_calculator_config.json"):
        """
        Initialize configuration manager

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.weight_config = WeightCalculatorConfig()
        self.correlation_config = CorrelationSizerConfig()
        self.rebalancer_config = PortfolioRebalancerConfig()

        # Load configuration from file
        self.load_config()

        logger.info("⚙️ Weight Calculator Configuration Manager initialized")

    def load_config(self) -> bool:
        """
        Load configuration from file

        Returns:
            Success status
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load weight calculator config
                if 'weight_calculator' in data:
                    wc_data = data['weight_calculator']
                    self.weight_config = WeightCalculatorConfig(**wc_data)

                # Load correlation sizer config
                if 'correlation_sizer' in data:
                    cs_data = data['correlation_sizer']
                    self.correlation_config = CorrelationSizerConfig(**cs_data)

                # Load rebalancer config
                if 'portfolio_rebalancer' in data:
                    pr_data = data['portfolio_rebalancer']
                    self.rebalancer_config = PortfolioRebalancerConfig(**pr_data)

                logger.info(f"📂 Configuration loaded from {self.config_file}")
                return True
            else:
                logger.info(f"📄 Configuration file {self.config_file} not found, using defaults")
                # Save default configuration
                self.save_config()
                return True

        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            return False

    def save_config(self) -> bool:
        """
        Save current configuration to file

        Returns:
            Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            config_data = {
                'weight_calculator': asdict(self.weight_config),
                'correlation_sizer': asdict(self.correlation_config),
                'portfolio_rebalancer': asdict(self.rebalancer_config),
                'metadata': {
                    'version': '1.0',
                    'timestamp': str(pd.Timestamp.now()),
                    'description': 'Weight Calculator Configuration'
                }
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)

            logger.info(f"💾 Configuration saved to {self.config_file}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
            return False

    def get_weight_calculator_config(self) -> WeightCalculatorConfig:
        """Get weight calculator configuration"""
        return self.weight_config

    def get_correlation_sizer_config(self) -> CorrelationSizerConfig:
        """Get correlation sizer configuration"""
        return self.correlation_config

    def get_rebalancer_config(self) -> PortfolioRebalancerConfig:
        """Get portfolio rebalancer configuration"""
        return self.rebalancer_config

    def update_weight_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update weight calculator configuration

        Args:
            updates: Configuration updates

        Returns:
            Success status
        """
        try:
            for key, value in updates.items():
                if hasattr(self.weight_config, key):
                    setattr(self.weight_config, key, value)
                else:
                    logger.warning(f"⚠️ Unknown weight config parameter: {key}")

            # Save updated configuration
            self.save_config()
            logger.info("🔄 Weight calculator configuration updated")
            return True

        except Exception as e:
            logger.error(f"❌ Error updating weight configuration: {e}")
            return False

    def check_circuit_breakers(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all circuit breakers against current system state

        Args:
            system_state: Current system state metrics

        Returns:
            Circuit breaker status
        """
        try:
            status = {
                'all_clear': True,
                'triggered_breakers': [],
                'warnings': [],
                'emergency_stop': False
            }

            # Check weight calculator circuit breakers
            wc_breakers = self._check_weight_calculator_breakers(system_state)
            if wc_breakers['triggered']:
                status['triggered_breakers'].extend(wc_breakers['triggered_breakers'])
                status['all_clear'] = False

            # Check correlation sizer circuit breakers
            cs_breakers = self._check_correlation_sizer_breakers(system_state)
            if cs_breakers['triggered']:
                status['triggered_breakers'].extend(cs_breakers['triggered_breakers'])
                status['all_clear'] = False

            # Check rebalancer circuit breakers
            pr_breakers = self._check_rebalancer_breakers(system_state)
            if pr_breakers['triggered']:
                status['triggered_breakers'].extend(pr_breakers['triggered_breakers'])
                status['all_clear'] = False

            # Check emergency stops
            emergency = self._check_emergency_stops(system_state)
            if emergency['triggered']:
                status['emergency_stop'] = True
                status['triggered_breakers'].extend(emergency['triggered_breakers'])
                status['all_clear'] = False

            if status['triggered_breakers']:
                logger.warning(f"🚨 CIRCUIT BREAKERS TRIGGERED: {', '.join(status['triggered_breakers'])}")

            return status

        except Exception as e:
            logger.error(f"❌ Error checking circuit breakers: {e}")
            return {'all_clear': False, 'error': str(e)}

    def _check_weight_calculator_breakers(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check weight calculator specific circuit breakers"""
        breakers = self.weight_config.circuit_breakers
        triggered = []

        # Check volatility
        portfolio_vol = system_state.get('portfolio_volatility', 0.0)
        if portfolio_vol > breakers['max_volatility']:
            triggered.append(f"Portfolio volatility {portfolio_vol:.1%} > {breakers['max_volatility']:.1%}")

        # Check concentration
        max_weight = system_state.get('max_weight', 0.0)
        if max_weight > breakers['max_concentration']:
            triggered.append(f"Max weight {max_weight:.1%} > {breakers['max_concentration']:.1%}")

        return {
            'triggered': len(triggered) > 0,
            'triggered_breakers': triggered
        }

    def _check_correlation_sizer_breakers(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check correlation sizer specific circuit breakers"""
        breakers = self.correlation_config.circuit_breakers
        triggered = []

        # Check correlation emergency
        avg_correlation = system_state.get('average_correlation', 0.0)
        if avg_correlation > breakers['correlation_emergency']:
            triggered.append(f"Average correlation {avg_correlation:.1%} > {breakers['correlation_emergency']:.1%}")

        # Check diversification emergency
        diversification = system_state.get('diversification_ratio', 1.0)
        if diversification < breakers['diversification_emergency']:
            triggered.append(f"Diversification ratio {diversification:.2f} < {breakers['diversification_emergency']:.2f}")

        return {
            'triggered': len(triggered) > 0,
            'triggered_breakers': triggered
        }

    def _check_rebalancer_breakers(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check rebalancer specific circuit breakers"""
        breakers = self.rebalancer_config.circuit_breakers
        triggered = []

        # Check trade ratio
        max_trade_ratio = system_state.get('max_trade_ratio', 0.0)
        if max_trade_ratio > breakers['max_trade_ratio']:
            triggered.append(f"Max trade ratio {max_trade_ratio:.1%} > {breakers['max_trade_ratio']:.1%}")

        # Check cost threshold
        cost_ratio = system_state.get('cost_ratio', 0.0)
        if cost_ratio > breakers['cost_threshold']:
            triggered.append(f"Cost ratio {cost_ratio:.1%} > {breakers['cost_threshold']:.1%}")

        return {
            'triggered': len(triggered) > 0,
            'triggered_breakers': triggered
        }

    def _check_emergency_stops(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check emergency stop conditions"""
        emergency = self.weight_config.circuit_breakers['emergency_stop']
        triggered = []

        if emergency['enabled']:
            # Check volatility emergency
            portfolio_vol = system_state.get('portfolio_volatility', 0.0)
            if portfolio_vol > emergency['volatility_threshold']:
                triggered.append(f"EMERGENCY: Portfolio volatility {portfolio_vol:.1%} > {emergency['volatility_threshold']:.1%}")

            # Check correlation emergency
            avg_correlation = system_state.get('average_correlation', 0.0)
            if avg_correlation > emergency['correlation_threshold']:
                triggered.append(f"EMERGENCY: Average correlation {avg_correlation:.1%} > {emergency['correlation_threshold']:.1%}")

            # Check concentration emergency
            max_weight = system_state.get('max_weight', 0.0)
            if max_weight > emergency['concentration_threshold']:
                triggered.append(f"EMERGENCY: Max weight {max_weight:.1%} > {emergency['concentration_threshold']:.1%}")

        return {
            'triggered': len(triggered) > 0,
            'triggered_breakers': triggered
        }

    def get_safety_limits(self) -> Dict[str, Any]:
        """
        Get all safety limits and thresholds

        Returns:
            Safety limits dictionary
        """
        return {
            'weight_calculator': {
                'max_volatility': self.weight_config.circuit_breakers['max_volatility'],
                'max_concentration': self.weight_config.circuit_breakers['max_concentration'],
                'min_diversification': self.weight_config.circuit_breakers['min_diversification']
            },
            'correlation_sizer': {
                'correlation_emergency': self.correlation_config.circuit_breakers['correlation_emergency'],
                'volatility_emergency': self.correlation_config.circuit_breakers['volatility_emergency']
            },
            'portfolio_rebalancer': {
                'max_trade_ratio': self.rebalancer_config.circuit_breakers['max_trade_ratio'],
                'cost_threshold': self.rebalancer_config.circuit_breakers['cost_threshold']
            }
        }

    def reset_to_defaults(self) -> bool:
        """
        Reset all configurations to default values

        Returns:
            Success status
        """
        try:
            self.weight_config = WeightCalculatorConfig()
            self.correlation_config = CorrelationSizerConfig()
            self.rebalancer_config = PortfolioRebalancerConfig()

            self.save_config()
            logger.info("🔄 Configuration reset to defaults")
            return True

        except Exception as e:
            logger.error(f"❌ Error resetting configuration: {e}")
            return False


# Global configuration manager instance
_config_manager = None

def get_weight_config_manager() -> WeightCalculatorConfigManager:
    """Get global weight calculator configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = WeightCalculatorConfigManager()
    return _config_manager


def create_default_config_file(config_file: str = "core/weight_calculator_config.json") -> bool:
    """
    Create default configuration file

    Args:
        config_file: Path to configuration file

    Returns:
        Success status
    """
    try:
        manager = WeightCalculatorConfigManager(config_file)
        return manager.save_config()
    except Exception as e:
        logger.error(f"❌ Error creating default config file: {e}")
        return False
