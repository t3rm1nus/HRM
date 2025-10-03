"""
Correlation-Based Position Sizer

This module provides position sizing adjustments based on asset correlations,
helping to manage portfolio risk by reducing exposure to highly correlated assets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from core.logging import logger
from l2_tactic.utils import safe_float


@dataclass
class CorrelationData:
    """Data structure for correlation-based position sizing"""
    symbol: str
    correlations: Dict[str, float]  # Correlation with other assets
    volatility: float
    current_weight: float = 0.0


class CorrelationPositionSizer:
    """
    Handles correlation-based position sizing to manage portfolio risk
    """

    def __init__(self, max_correlation_threshold: float = 0.8,
                 correlation_penalty_factor: float = 0.7):
        """
        Initialize correlation position sizer

        Args:
            max_correlation_threshold: Maximum allowed correlation before applying penalties
            correlation_penalty_factor: Factor to reduce position size for highly correlated assets
        """
        self.max_correlation_threshold = max_correlation_threshold
        self.correlation_penalty_factor = correlation_penalty_factor
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.asset_data: Dict[str, CorrelationData] = {}

        logger.info("🔗 Correlation Position Sizer initialized")

    def add_asset_correlation_data(self, correlation_data: CorrelationData) -> bool:
        """
        Add correlation data for an asset

        Args:
            correlation_data: Correlation information for the asset

        Returns:
            Success status
        """
        try:
            self.asset_data[correlation_data.symbol] = correlation_data
            logger.debug(f"🔗 Added correlation data for {correlation_data.symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding correlation data for {correlation_data.symbol}: {e}")
            return False

    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """
        Update the correlation matrix for all assets

        Args:
            correlation_matrix: Correlation matrix with assets as index/columns
        """
        try:
            self.correlation_matrix = correlation_matrix.copy()
            logger.info("📈 Correlation matrix updated for position sizing")
        except Exception as e:
            logger.error(f"❌ Error updating correlation matrix: {e}")

    def calculate_correlation_adjusted_size(self, symbol: str, base_position_size: float,
                                          current_portfolio: Dict[str, float],
                                          market_data: Dict[str, Any]) -> float:
        """
        Calculate position size adjusted for correlations

        Args:
            symbol: Asset symbol
            base_position_size: Base position size before correlation adjustment
            current_portfolio: Current portfolio weights
            market_data: Current market data

        Returns:
            Adjusted position size
        """
        try:
            # Input validation and safety checks
            if not isinstance(base_position_size, (int, float)) or base_position_size < 0:
                logger.warning(f"⚠️ Invalid base_position_size: {base_position_size}, using 0.0")
                return 0.0

            if not isinstance(symbol, str) or not symbol:
                logger.warning(f"⚠️ Invalid symbol: {symbol}, using base size")
                return base_position_size

            if symbol not in self.asset_data:
                logger.warning(f"⚠️ No correlation data for {symbol}, using base size")
                return base_position_size

            asset_data = self.asset_data[symbol]

            # Calculate correlation risk score with safety check
            correlation_risk = self._calculate_correlation_risk_score(symbol, current_portfolio)

            # Validate correlation_risk is a number
            if not isinstance(correlation_risk, (int, float)) or np.isnan(correlation_risk) or not np.isfinite(correlation_risk):
                logger.warning(f"⚠️ Invalid correlation risk score for {symbol}: {correlation_risk}, using base size")
                return base_position_size

            # Apply correlation-based adjustment
            if correlation_risk > self.max_correlation_threshold:
                # High correlation - reduce position size
                correlation_factor = self.correlation_penalty_factor

                # Ensure correlation_factor is valid
                if not isinstance(correlation_factor, (int, float)) or correlation_factor < 0:
                    correlation_factor = 1.0
                    logger.warning(f"⚠️ Invalid correlation_factor: {correlation_factor}, using 1.0")

                adjusted_size = base_position_size * correlation_factor

                # Ensure adjusted_size is valid
                if not isinstance(adjusted_size, (int, float)) or np.isnan(adjusted_size) or not np.isfinite(adjusted_size):
                    logger.warning(f"⚠️ Invalid adjusted_size for {symbol}: {adjusted_size}, using base size")
                    return base_position_size

                logger.info(f"🔗 CORRELATION ADJUSTMENT for {symbol}:")
                logger.info(f"   Base size: ${base_position_size:.2f}")
                logger.info(f"   Correlation risk: {correlation_risk:.3f}")
                logger.info(f"   Adjustment factor: {correlation_factor:.3f}")
                logger.info(f"   Adjusted size: ${adjusted_size:.2f}")

                return adjusted_size
            else:
                # Low correlation - keep base size
                return base_position_size

        except Exception as e:
            logger.error(f"❌ Error calculating correlation-adjusted size for {symbol}: {e}")
            # Ensure we return a valid float on error
            if isinstance(base_position_size, (int, float)) and not np.isnan(base_position_size):
                return base_position_size
            return 0.0  # Return 0.0 as final fallback

    def _calculate_correlation_risk_score(self, symbol: str,
                                        current_portfolio: Dict[str, float]) -> float:
        """
        Calculate correlation risk score for an asset

        Args:
            symbol: Asset symbol
            current_portfolio: Current portfolio weights

        Returns:
            Correlation risk score (0.0 to 1.0, higher = more correlated)
        """
        try:
            if symbol not in self.asset_data:
                return 0.0

            asset_data = self.asset_data[symbol]
            correlations = asset_data.correlations

            # Calculate weighted average correlation with current portfolio
            total_correlation = 0.0
            total_weight = 0.0

            for other_symbol, weight in current_portfolio.items():
                if other_symbol != symbol and other_symbol in correlations and weight > 0:
                    correlation = abs(correlations.get(other_symbol, 0.0))
                    total_correlation += correlation * weight
                    total_weight += weight

            if total_weight == 0:
                return 0.0

            # Average correlation weighted by portfolio weights
            avg_correlation = total_correlation / total_weight

            # Also consider the asset's volatility correlation with portfolio
            portfolio_volatility = self._calculate_portfolio_volatility_correlation(symbol)

            # Combine correlation and volatility factors
            correlation_risk = (avg_correlation * 0.7 + portfolio_volatility * 0.3)

            return min(1.0, correlation_risk)

        except Exception as e:
            logger.error(f"❌ Error calculating correlation risk score for {symbol}: {e}")
            return 0.0

    def _calculate_portfolio_volatility_correlation(self, symbol: str) -> float:
        """
        Calculate how the asset's volatility correlates with portfolio volatility

        Args:
            symbol: Asset symbol

        Returns:
            Volatility correlation score (0.0 to 1.0)
        """
        try:
            if not self.correlation_matrix or symbol not in self.correlation_matrix.index:
                return 0.5  # Neutral score

            # Get correlations with other assets
            correlations = self.correlation_matrix.loc[symbol]

            # Calculate average absolute correlation (volatility clustering)
            avg_abs_correlation = correlations.abs().mean()

            return avg_abs_correlation

        except Exception as e:
            logger.error(f"❌ Error calculating portfolio volatility correlation for {symbol}: {e}")
            return 0.5

    def get_correlation_report(self, current_portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive correlation report for the portfolio

        Args:
            current_portfolio: Current portfolio weights

        Returns:
            Correlation analysis report
        """
        try:
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'portfolio_correlation_analysis': {},
                'correlation_risk_metrics': {},
                'recommendations': []
            }

            # Analyze each asset's correlation profile
            for symbol in current_portfolio.keys():
                if symbol in self.asset_data:
                    asset_data = self.asset_data[symbol]
                    correlation_risk = self._calculate_correlation_risk_score(symbol, current_portfolio)

                    report['portfolio_correlation_analysis'][symbol] = {
                        'current_weight': current_portfolio.get(symbol, 0.0),
                        'correlation_risk_score': correlation_risk,
                        'correlations': asset_data.correlations,
                        'volatility': asset_data.volatility,
                        'risk_level': self._classify_risk_level(correlation_risk)
                    }

            # Calculate portfolio-level correlation metrics
            portfolio_correlations = self._calculate_portfolio_correlation_metrics(current_portfolio)
            report['correlation_risk_metrics'] = portfolio_correlations

            # Generate recommendations
            report['recommendations'] = self._generate_correlation_recommendations(
                report['portfolio_correlation_analysis'],
                portfolio_correlations
            )

            logger.info("📊 Correlation report generated")
            return report

        except Exception as e:
            logger.error(f"❌ Error generating correlation report: {e}")
            return {'error': str(e)}

    def _calculate_portfolio_correlation_metrics(self, current_portfolio: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio-level correlation metrics

        Args:
            current_portfolio: Current portfolio weights

        Returns:
            Portfolio correlation metrics
        """
        try:
            metrics = {}

            if not self.correlation_matrix:
                return {'error': 'No correlation matrix available'}

            # Get assets in current portfolio
            portfolio_assets = [symbol for symbol in current_portfolio.keys()
                              if symbol in self.correlation_matrix.index and current_portfolio[symbol] > 0]

            if len(portfolio_assets) < 2:
                return {'error': 'Need at least 2 assets for correlation analysis'}

            # Extract correlation submatrix
            portfolio_corr = self.correlation_matrix.loc[portfolio_assets, portfolio_assets]

            # Average correlation
            metrics['average_correlation'] = portfolio_corr.values[np.triu_indices_from(portfolio_corr.values, k=1)].mean()

            # Maximum correlation
            metrics['max_correlation'] = portfolio_corr.values[np.triu_indices_from(portfolio_corr.values, k=1)].max()

            # Correlation diversity score (lower is better diversified)
            metrics['correlation_diversity_score'] = np.sqrt(np.sum(portfolio_corr ** 2).mean())

            # Effective number of uncorrelated assets
            if metrics['average_correlation'] < 1.0:
                metrics['effective_num_assets'] = 1 / (1 - metrics['average_correlation'])
            else:
                metrics['effective_num_assets'] = 1.0

            # Risk concentration from correlations
            weights = np.array([current_portfolio[symbol] for symbol in portfolio_assets])
            corr_matrix = portfolio_corr.values

            # Calculate correlation-adjusted volatility
            portfolio_variance = weights.T @ corr_matrix @ weights
            metrics['correlation_adjusted_volatility'] = np.sqrt(max(0, portfolio_variance))

            return metrics

        except Exception as e:
            logger.error(f"❌ Error calculating portfolio correlation metrics: {e}")
            return {'error': str(e)}

    def _classify_risk_level(self, correlation_risk: float) -> str:
        """Classify correlation risk level"""
        if correlation_risk >= 0.8:
            return "HIGH"
        elif correlation_risk >= 0.6:
            return "MEDIUM_HIGH"
        elif correlation_risk >= 0.4:
            return "MEDIUM"
        elif correlation_risk >= 0.2:
            return "MEDIUM_LOW"
        else:
            return "LOW"

    def _generate_correlation_recommendations(self, asset_analysis: Dict[str, Any],
                                            portfolio_metrics: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on correlation analysis

        Args:
            asset_analysis: Individual asset correlation analysis
            portfolio_metrics: Portfolio-level correlation metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        try:
            # Check for high correlation assets
            high_risk_assets = [
                symbol for symbol, data in asset_analysis.items()
                if data.get('risk_level') == 'HIGH'
            ]

            if high_risk_assets:
                recommendations.append(
                    f"Reduce position sizes for highly correlated assets: {', '.join(high_risk_assets)}"
                )

            # Check portfolio diversification
            avg_correlation = portfolio_metrics.get('average_correlation', 0.5)
            if avg_correlation > 0.7:
                recommendations.append(
                    f"Portfolio is highly correlated (avg: {avg_correlation:.2f}). Consider adding uncorrelated assets."
                )
            elif avg_correlation < 0.3:
                recommendations.append(
                    f"Portfolio has good diversification (avg correlation: {avg_correlation:.2f})."
                )

            # Check effective number of assets
            effective_assets = portfolio_metrics.get('effective_num_assets', 1.0)
            actual_assets = len(asset_analysis)

            if effective_assets < actual_assets * 0.7:
                recommendations.append(
                    f"Low diversification benefit: {effective_assets:.1f} effective assets vs {actual_assets} actual assets."
                )

            # Correlation-adjusted volatility warning
            corr_vol = portfolio_metrics.get('correlation_adjusted_volatility', 0.0)
            if corr_vol > 0.25:  # 25% correlation-adjusted volatility
                recommendations.append(
                    f"High correlation-adjusted volatility: {corr_vol:.1%}. Consider reducing correlated positions."
                )

        except Exception as e:
            logger.error(f"❌ Error generating correlation recommendations: {e}")
            recommendations.append("Error generating recommendations")

        return recommendations

    def get_correlation_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data for correlation heatmap visualization

        Returns:
            Heatmap data or None if not available
        """
        try:
            if not self.correlation_matrix:
                return None

            # Convert correlation matrix to list format for visualization
            symbols = self.correlation_matrix.index.tolist()
            correlation_values = self.correlation_matrix.values.tolist()

            heatmap_data = {
                'symbols': symbols,
                'correlation_matrix': correlation_values,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            return heatmap_data

        except Exception as e:
            logger.error(f"❌ Error generating correlation heatmap data: {e}")
            return None

    def calculate_optimal_correlation_adjustment(self, symbol: str,
                                               target_correlation_limit: float = 0.7) -> float:
        """
        Calculate optimal position size adjustment to meet correlation limits

        Args:
            symbol: Asset symbol
            target_correlation_limit: Target maximum correlation

        Returns:
            Adjustment factor (multiplier for position size)
        """
        try:
            if symbol not in self.asset_data:
                return 1.0

            asset_data = self.asset_data[symbol]
            current_correlations = asset_data.correlations

            # Find assets with correlation above target
            high_corr_assets = [
                other_symbol for other_symbol, corr in current_correlations.items()
                if abs(corr) > target_correlation_limit and other_symbol != symbol
            ]

            if not high_corr_assets:
                return 1.0  # No adjustment needed

            # Calculate average correlation with high-correlation assets
            high_corrs = [abs(current_correlations[asset]) for asset in high_corr_assets]
            avg_high_corr = sum(high_corrs) / len(high_corrs)

            # Calculate adjustment factor
            # More correlated = smaller position
            adjustment_factor = max(0.3, 1.0 - (avg_high_corr - target_correlation_limit) * 2)

            logger.info(f"🎯 Optimal correlation adjustment for {symbol}:")
            logger.info(f"   Target limit: {target_correlation_limit:.2f}")
            logger.info(f"   Average high correlation: {avg_high_corr:.2f}")
            logger.info(f"   Adjustment factor: {adjustment_factor:.2f}")

            return adjustment_factor

        except Exception as e:
            logger.error(f"❌ Error calculating optimal correlation adjustment for {symbol}: {e}")
            return 1.0


# Utility functions for correlation analysis

def calculate_asset_correlations(price_data: pd.DataFrame,
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlations between assets from price data

    Args:
        price_data: DataFrame with asset prices (columns = assets, index = dates)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix
    """
    try:
        # Calculate returns
        returns = price_data.pct_change().dropna()

        # Calculate correlations
        correlation_matrix = returns.corr(method=method)

        logger.info(f"📊 Asset correlations calculated using {method} method")
        return correlation_matrix

    except Exception as e:
        logger.error(f"❌ Error calculating asset correlations: {e}")
        return pd.DataFrame()


def detect_correlation_clusters(correlation_matrix: pd.DataFrame,
                              threshold: float = 0.7) -> List[List[str]]:
    """
    Detect clusters of highly correlated assets

    Args:
        correlation_matrix: Asset correlation matrix
        threshold: Correlation threshold for clustering

    Returns:
        List of asset clusters
    """
    try:
        clusters = []
        visited = set()

        for asset in correlation_matrix.index:
            if asset in visited:
                continue

            # Find highly correlated assets
            correlated_assets = correlation_matrix.index[
                correlation_matrix.loc[asset].abs() > threshold
            ].tolist()

            # Remove already visited assets
            cluster = [asset for asset in correlated_assets if asset not in visited]

            if len(cluster) > 1:
                clusters.append(cluster)
                visited.update(cluster)

        logger.info(f"🔗 Detected {len(clusters)} correlation clusters")
        return clusters

    except Exception as e:
        logger.error(f"❌ Error detecting correlation clusters: {e}")
        return []


def calculate_correlation_diversification_ratio(correlation_matrix: pd.DataFrame) -> float:
    """
    Calculate diversification ratio based on correlations

    Args:
        correlation_matrix: Asset correlation matrix

    Returns:
        Diversification ratio (higher = better diversification)
    """
    try:
        n_assets = len(correlation_matrix)

        if n_assets < 2:
            return 1.0

        # Average correlation
        avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

        # Diversification ratio = sqrt(1 / (1 - avg_corr)) for n assets
        if avg_corr >= 1.0:
            return 1.0

        diversification_ratio = np.sqrt(n_assets / (1 + (n_assets - 1) * avg_corr))

        return diversification_ratio

    except Exception as e:
        logger.error(f"❌ Error calculating correlation diversification ratio: {e}")
        return 1.0
