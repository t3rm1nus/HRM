"""
Weight Calculator for Portfolio Rebalancing and Position Sizing

This module provides comprehensive portfolio weighting strategies including:
- Equal Weight: Equal allocation across all assets
- Market Cap Weight: Weight based on market capitalization
- Risk Parity: Equal risk contribution from each asset
- Minimum Variance: Minimize portfolio variance
- Maximum Sharpe: Maximize risk-adjusted returns
- Custom constraints and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from core.logging import logger
from l2_tactic.l2_utils import safe_float


class WeightStrategy(Enum):
    """Portfolio weighting strategies"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    VOLATILITY_TARGETED = "volatility_targeted"
    CUSTOM = "custom"


@dataclass
class AssetData:
    """Data structure for asset information"""
    symbol: str
    price: float
    market_cap: Optional[float] = None
    volatility: Optional[float] = None
    expected_return: Optional[float] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    historical_returns: Optional[pd.Series] = None


@dataclass
class WeightConstraints:
    """Constraints for portfolio optimization"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_concentration: float = 0.3  # Maximum weight for any single asset
    min_assets: int = 1
    max_assets: int = 10
    risk_free_rate: float = 0.02
    target_volatility: Optional[float] = None
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None  # (min, max) by sector


class WeightCalculator:
    """
    Main weight calculator class supporting multiple weighting strategies
    """

    def __init__(self, constraints: Optional[WeightConstraints] = None):
        """
        Initialize weight calculator

        Args:
            constraints: Portfolio constraints for optimization
        """
        self.constraints = constraints or WeightConstraints()
        self.assets: Dict[str, AssetData] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.market_data: Dict[str, Any] = {}

        logger.info("🎯 Weight Calculator initialized")

    def add_asset(self, asset_data: AssetData) -> bool:
        """
        Add asset data for weighting calculations

        Args:
            asset_data: Asset information

        Returns:
            Success status
        """
        try:
            self.assets[asset_data.symbol] = asset_data
            logger.debug(f"📊 Added asset {asset_data.symbol} for weighting")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding asset {asset_data.symbol}: {e}")
            return False

    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update market data for all assets"""
        self.market_data = market_data.copy()
        logger.debug("📈 Market data updated for weight calculations")

    def calculate_weights(self, strategy: WeightStrategy,
                         target_volatility: Optional[float] = None,
                         risk_appetite: str = "moderate") -> Dict[str, float]:
        """
        Calculate portfolio weights using specified strategy

        Args:
            strategy: Weighting strategy to use
            target_volatility: Target portfolio volatility (for vol-targeted strategies)
            risk_appetite: Risk appetite level ("low", "moderate", "high", "aggressive")

        Returns:
            Dictionary of asset weights
        """
        try:
            if not self.assets:
                logger.warning("⚠️ No assets available for weighting")
                return {}

            # Apply risk appetite adjustments
            risk_multiplier = self._get_risk_multiplier(risk_appetite)

            # Calculate base weights based on strategy
            if strategy == WeightStrategy.EQUAL_WEIGHT:
                weights = self._calculate_equal_weights()
            elif strategy == WeightStrategy.MARKET_CAP_WEIGHT:
                weights = self._calculate_market_cap_weights()
            elif strategy == WeightStrategy.RISK_PARITY:
                weights = self._calculate_risk_parity_weights()
            elif strategy == WeightStrategy.MINIMUM_VARIANCE:
                weights = self._calculate_minimum_variance_weights()
            elif strategy == WeightStrategy.MAXIMUM_SHARPE:
                weights = self._calculate_maximum_sharpe_weights()
            elif strategy == WeightStrategy.VOLATILITY_TARGETED:
                weights = self._calculate_volatility_targeted_weights(target_volatility or 0.15)
            else:
                logger.warning(f"⚠️ Unknown strategy {strategy}, using equal weight")
                weights = self._calculate_equal_weights()

            # Apply risk adjustments
            weights = self._apply_risk_adjustments(weights, risk_multiplier)

            # Apply constraints
            weights = self._apply_constraints(weights)

            # Normalize weights
            weights = self._normalize_weights(weights)

            logger.info(f"⚖️ Weights calculated using {strategy.value}: {len(weights)} assets")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating weights with {strategy.value}: {e}")
            return {}

    def _calculate_equal_weights(self) -> Dict[str, float]:
        """Calculate equal weights for all assets"""
        try:
            num_assets = len(self.assets)
            if num_assets == 0:
                return {}

            equal_weight = 1.0 / num_assets
            weights = {symbol: equal_weight for symbol in self.assets.keys()}

            logger.debug(f"⚖️ Equal weights calculated: {equal_weight:.4f} per asset")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating equal weights: {e}")
            return {}

    def _calculate_market_cap_weights(self) -> Dict[str, float]:
        """Calculate weights based on market capitalization"""
        try:
            market_caps = {}
            total_market_cap = 0.0

            for symbol, asset in self.assets.items():
                if asset.market_cap and asset.market_cap > 0:
                    market_caps[symbol] = asset.market_cap
                    total_market_cap += asset.market_cap
                else:
                    logger.warning(f"⚠️ No market cap data for {symbol}, skipping")

            if total_market_cap == 0:
                logger.warning("⚠️ No valid market cap data, falling back to equal weights")
                return self._calculate_equal_weights()

            weights = {}
            for symbol, market_cap in market_caps.items():
                weights[symbol] = market_cap / total_market_cap

            logger.debug(f"🏢 Market cap weights calculated: total_cap=${total_market_cap:,.0f}")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating market cap weights: {e}")
            return self._calculate_equal_weights()

    def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calculate risk parity weights (equal risk contribution)"""
        try:
            volatilities = {}
            valid_assets = []

            # Get volatilities for all assets
            for symbol, asset in self.assets.items():
                vol = asset.volatility
                if vol is None and asset.historical_returns is not None:
                    vol = asset.historical_returns.std() * np.sqrt(252)  # Annualized volatility

                if vol and vol > 0:
                    volatilities[symbol] = vol
                    valid_assets.append(symbol)
                else:
                    logger.warning(f"⚠️ No volatility data for {symbol}, skipping")

            if len(valid_assets) < 2:
                logger.warning("⚠️ Need at least 2 assets with volatility data for risk parity")
                return self._calculate_equal_weights()

            # Risk parity: weight = 1/(volatility * num_assets)
            # This gives equal risk contribution from each asset
            weights = {}
            for symbol in valid_assets:
                weights[symbol] = 1.0 / (volatilities[symbol] * len(valid_assets))

            # Normalize to sum to 1
            total_weight = sum(weights.values())
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

            logger.debug(f"🎯 Risk parity weights calculated: {len(weights)} assets")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating risk parity weights: {e}")
            return self._calculate_equal_weights()

    def _calculate_minimum_variance_weights(self) -> Dict[str, float]:
        """Calculate minimum variance portfolio weights"""
        try:
            if not self._has_correlation_data():
                logger.warning("⚠️ No correlation data for minimum variance optimization")
                return self._calculate_equal_weights()

            # Get covariance matrix
            cov_matrix = self._calculate_covariance_matrix()

            if cov_matrix is None or cov_matrix.empty:
                logger.warning("⚠️ Could not calculate covariance matrix")
                return self._calculate_equal_weights()

            # Minimum variance portfolio: weights that minimize portfolio variance
            # For simplicity, use equal weights if optimization fails
            try:
                # Calculate minimum variance weights using matrix algebra
                ones = np.ones(len(cov_matrix))
                inv_cov = np.linalg.inv(cov_matrix.values)
                weights_array = inv_cov @ ones
                weights_array = weights_array / (ones @ weights_array)

                # Convert to dictionary
                weights = {}
                for i, symbol in enumerate(cov_matrix.index):
                    weights[symbol] = max(0, weights_array[i])  # Ensure non-negative

                # Re-normalize
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

                logger.debug(f"📉 Minimum variance weights calculated: {len(weights)} assets")
                return weights

            except Exception as e:
                logger.warning(f"⚠️ Minimum variance optimization failed: {e}, using equal weights")
                return self._calculate_equal_weights()

        except Exception as e:
            logger.error(f"❌ Error calculating minimum variance weights: {e}")
            return self._calculate_equal_weights()

    def _calculate_maximum_sharpe_weights(self) -> Dict[str, float]:
        """Calculate maximum Sharpe ratio portfolio weights"""
        try:
            if not self._has_return_data():
                logger.warning("⚠️ No return data for maximum Sharpe optimization")
                return self._calculate_equal_weights()

            expected_returns = {}
            volatilities = {}

            # Get expected returns and volatilities
            for symbol, asset in self.assets.items():
                ret = asset.expected_return
                if ret is None and asset.historical_returns is not None:
                    ret = asset.historical_returns.mean() * 252  # Annualized return

                vol = asset.volatility
                if vol is None and asset.historical_returns is not None:
                    vol = asset.historical_returns.std() * np.sqrt(252)

                if ret is not None and vol is not None and vol > 0:
                    expected_returns[symbol] = ret
                    volatilities[symbol] = vol

            if len(expected_returns) < 2:
                logger.warning("⚠️ Need at least 2 assets with return/volatility data")
                return self._calculate_equal_weights()

            # Maximum Sharpe ratio portfolio
            # For simplicity, use risk-adjusted equal weights
            weights = {}
            total_score = 0

            for symbol in expected_returns.keys():
                # Sharpe ratio = (expected_return - risk_free) / volatility
                sharpe = (expected_returns[symbol] - self.constraints.risk_free_rate) / volatilities[symbol]
                weights[symbol] = max(0, sharpe)  # Only positive Sharpe ratios
                total_score += weights[symbol]

            if total_score > 0:
                weights = {symbol: weight / total_score for symbol, weight in weights.items()}

            logger.debug(f"📈 Maximum Sharpe weights calculated: {len(weights)} assets")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating maximum Sharpe weights: {e}")
            return self._calculate_equal_weights()

    def _calculate_volatility_targeted_weights(self, target_volatility: float) -> Dict[str, float]:
        """Calculate weights to achieve target portfolio volatility"""
        try:
            # Start with risk parity weights as base
            base_weights = self._calculate_risk_parity_weights()

            if not base_weights:
                return self._calculate_equal_weights()

            # Calculate current portfolio volatility
            current_vol = self._calculate_portfolio_volatility(base_weights)

            if current_vol <= 0:
                logger.warning("⚠️ Could not calculate current portfolio volatility")
                return base_weights

            # Scale weights to achieve target volatility
            scale_factor = target_volatility / current_vol

            # Apply scaling but maintain relative weights
            weights = {symbol: weight * scale_factor for symbol, weight in base_weights.items()}

            # Re-normalize
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {symbol: weight / total_weight for symbol, weight in weights.items()}

            logger.debug(f"🎯 Volatility-targeted weights: target={target_volatility:.1%}, current={current_vol:.1%}")
            return weights

        except Exception as e:
            logger.error(f"❌ Error calculating volatility-targeted weights: {e}")
            return self._calculate_equal_weights()

    def _get_risk_multiplier(self, risk_appetite: str) -> float:
        """Get risk multiplier based on risk appetite"""
        risk_multipliers = {
            "low": 0.7,        # Conservative: reduce weights
            "moderate": 1.0,   # Balanced: standard weights
            "high": 1.3,       # Aggressive: increase weights
            "aggressive": 1.6  # Very aggressive: significantly increase weights
        }
        return risk_multipliers.get(risk_appetite.lower(), 1.0)

    def _apply_risk_adjustments(self, weights: Dict[str, float], risk_multiplier: float) -> Dict[str, float]:
        """Apply risk-based adjustments to weights"""
        try:
            adjusted_weights = {}

            for symbol, weight in weights.items():
                asset = self.assets.get(symbol)
                if asset and asset.volatility:
                    # Reduce weight for high volatility assets
                    vol_adjustment = min(1.0, 0.5 / asset.volatility) if asset.volatility > 0.5 else 1.0
                    adjusted_weights[symbol] = weight * risk_multiplier * vol_adjustment
                else:
                    adjusted_weights[symbol] = weight * risk_multiplier

            return adjusted_weights

        except Exception as e:
            logger.error(f"❌ Error applying risk adjustments: {e}")
            return weights

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply portfolio constraints to weights"""
        try:
            constrained_weights = weights.copy()

            # Apply minimum and maximum weight constraints
            for symbol in constrained_weights:
                weight = constrained_weights[symbol]
                constrained_weights[symbol] = max(self.constraints.min_weight,
                                                min(self.constraints.max_weight, weight))

            # Apply concentration limits
            for symbol in constrained_weights:
                weight = constrained_weights[symbol]
                if weight > self.constraints.max_concentration:
                    constrained_weights[symbol] = self.constraints.max_concentration

            # Apply sector constraints if available
            if self.constraints.sector_constraints:
                constrained_weights = self._apply_sector_constraints(constrained_weights)

            return constrained_weights

        except Exception as e:
            logger.error(f"❌ Error applying constraints: {e}")
            return weights

    def _apply_sector_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply sector-based constraints (placeholder for future implementation)"""
        # This would require sector classification data
        # For now, return weights unchanged
        return weights

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0"""
        try:
            total_weight = sum(weights.values())
            if total_weight > 0:
                return {symbol: weight / total_weight for symbol, weight in weights.items()}
            else:
                logger.warning("⚠️ Total weight is zero, cannot normalize")
                return {}

        except Exception as e:
            logger.error(f"❌ Error normalizing weights: {e}")
            return weights

    def _has_correlation_data(self) -> bool:
        """Check if correlation data is available"""
        return self.correlation_matrix is not None and not self.correlation_matrix.empty

    def _has_return_data(self) -> bool:
        """Check if return data is available for assets"""
        return any(asset.expected_return is not None or asset.historical_returns is not None
                  for asset in self.assets.values())

    def _calculate_covariance_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate covariance matrix from available data"""
        try:
            if not self._has_correlation_data():
                return None

            # Get volatilities
            volatilities = {}
            for symbol in self.correlation_matrix.index:
                asset = self.assets.get(symbol)
                if asset and asset.volatility:
                    volatilities[symbol] = asset.volatility
                else:
                    return None  # Need volatility for all assets

            # Calculate covariance matrix: cov(i,j) = corr(i,j) * vol(i) * vol(j)
            cov_matrix = pd.DataFrame(index=self.correlation_matrix.index,
                                    columns=self.correlation_matrix.columns)

            for i in self.correlation_matrix.index:
                for j in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[i, j]
                    cov = corr * volatilities[i] * volatilities[j]
                    cov_matrix.loc[i, j] = cov

            return cov_matrix

        except Exception as e:
            logger.error(f"❌ Error calculating covariance matrix: {e}")
            return None

    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility given weights"""
        try:
            cov_matrix = self._calculate_covariance_matrix()
            if cov_matrix is None:
                return 0.0

            # Calculate portfolio variance: w^T * Σ * w
            weight_vector = np.array([weights.get(symbol, 0) for symbol in cov_matrix.index])
            portfolio_variance = weight_vector.T @ cov_matrix.values @ weight_vector
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))

            return portfolio_volatility

        except Exception as e:
            logger.error(f"❌ Error calculating portfolio volatility: {e}")
            return 0.0

    def get_portfolio_risk_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for the portfolio"""
        try:
            metrics = {}

            # Portfolio volatility
            metrics['volatility'] = self._calculate_portfolio_volatility(weights)

            # Value at Risk (VaR) - assuming normal distribution
            if metrics['volatility'] > 0:
                confidence_levels = [0.95, 0.99]
                for conf in confidence_levels:
                    z_score = {0.95: 1.645, 0.99: 2.326}[conf]
                    metrics[f'var_{int(conf*100)}'] = metrics['volatility'] * z_score

            # Expected Shortfall (ES) - assuming normal distribution
            if metrics['volatility'] > 0:
                metrics['es_95'] = metrics['volatility'] * (1 / (1 - 0.95)) * 1.645  # Approximation

            # Concentration metrics
            weights_list = list(weights.values())
            if weights_list:
                metrics['max_weight'] = max(weights_list)
                metrics['weight_concentration'] = sum(w**2 for w in weights_list)  # Herfindahl index

                # Number of assets with significant weights (>1%)
                metrics['effective_assets'] = sum(1 for w in weights_list if w > 0.01)

            # Risk-adjusted return metrics (if return data available)
            expected_return = self._calculate_portfolio_expected_return(weights)
            if expected_return is not None and metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = (expected_return - self.constraints.risk_free_rate) / metrics['volatility']

            logger.debug(f"📊 Portfolio risk metrics calculated: vol={metrics.get('volatility', 0):.1%}")
            return metrics

        except Exception as e:
            logger.error(f"❌ Error calculating portfolio risk metrics: {e}")
            return {}

    def _calculate_portfolio_expected_return(self, weights: Dict[str, float]) -> Optional[float]:
        """Calculate expected portfolio return"""
        try:
            weighted_returns = []

            for symbol, weight in weights.items():
                asset = self.assets.get(symbol)
                if asset:
                    ret = asset.expected_return
                    if ret is None and asset.historical_returns is not None:
                        ret = asset.historical_returns.mean() * 252  # Annualized

                    if ret is not None:
                        weighted_returns.append(weight * ret)

            return sum(weighted_returns) if weighted_returns else None

        except Exception as e:
            logger.error(f"❌ Error calculating portfolio expected return: {e}")
            return None

    def optimize_portfolio(self, strategy: WeightStrategy,
                          constraints: Optional[WeightConstraints] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Advanced portfolio optimization with constraints

        Args:
            strategy: Optimization strategy
            constraints: Additional constraints

        Returns:
            Tuple of (optimal_weights, optimization_metrics)
        """
        try:
            # For now, return the basic calculation
            # In a full implementation, this would use scipy.optimize or similar
            weights = self.calculate_weights(strategy)

            # Calculate metrics
            metrics = self.get_portfolio_risk_metrics(weights)
            metrics['optimization_strategy'] = strategy.value

            return weights, metrics

        except Exception as e:
            logger.error(f"❌ Error in portfolio optimization: {e}")
            return {}, {'error': str(e)}

    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """Update the correlation matrix for all assets"""
        try:
            self.correlation_matrix = correlation_matrix.copy()
            logger.info("📈 Correlation matrix updated for weight calculations")
        except Exception as e:
            logger.error(f"❌ Error updating correlation matrix: {e}")


# Utility functions for weight calculations

def calculate_equal_weights(assets: List[str]) -> Dict[str, float]:
    """Utility function to calculate equal weights"""
    if not assets:
        return {}
    weight = 1.0 / len(assets)
    return {asset: weight for asset in assets}


def calculate_market_cap_weights(market_caps: Dict[str, float]) -> Dict[str, float]:
    """Utility function to calculate market cap weights"""
    total_cap = sum(market_caps.values())
    if total_cap == 0:
        return calculate_equal_weights(list(market_caps.keys()))

    return {symbol: cap / total_cap for symbol, cap in market_caps.items()}


def validate_weights(weights: Dict[str, float], constraints: Optional[WeightConstraints] = None) -> Dict[str, Any]:
    """Validate weight calculations against constraints"""
    try:
        constraints = constraints or WeightConstraints()
        validation = {
            'valid': True,
            'issues': [],
            'total_weight': sum(weights.values()),
            'num_assets': len(weights)
        }

        # Check total weight
        if abs(validation['total_weight'] - 1.0) > 0.001:
            validation['issues'].append(f"Total weight {validation['total_weight']:.4f} != 1.0")
            validation['valid'] = False

        # Check individual weight constraints
        for symbol, weight in weights.items():
            if weight < constraints.min_weight:
                validation['issues'].append(f"{symbol} weight {weight:.4f} < min {constraints.min_weight}")
                validation['valid'] = False
            if weight > constraints.max_weight:
                validation['issues'].append(f"{symbol} weight {weight:.4f} > max {constraints.max_weight}")
                validation['valid'] = False

        # Check concentration
        max_weight = max(weights.values()) if weights else 0
        if max_weight > constraints.max_concentration:
            validation['issues'].append(f"Max weight {max_weight:.4f} > concentration limit {constraints.max_concentration}")
            validation['valid'] = False

        # Check number of assets
        if validation['num_assets'] < constraints.min_assets:
            validation['issues'].append(f"Too few assets: {validation['num_assets']} < {constraints.min_assets}")
            validation['valid'] = False
        if validation['num_assets'] > constraints.max_assets:
            validation['issues'].append(f"Too many assets: {validation['num_assets']} > {constraints.max_assets}")
            validation['valid'] = False

        return validation

    except Exception as e:
        logger.error(f"❌ Error validating weights: {e}")
        return {'valid': False, 'issues': [str(e)]}
