# position_sizer.py - L2 Tactical Position Sizing

"""
position_sizer.py - L2 Tactical Position Sizing

Implementa algoritmos avanzados de position sizing para L2:
- Kelly Criterion fraccionado
- Volatility targeting
- Risk parity optimization
- Correlation adjustments
- Portfolio heat management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .models import TacticalSignal, PositionSize, MarketFeatures, RiskMetrics
from .config import L2Config


logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Estado actual del portfolio para sizing decisions"""
    total_capital: float
    available_capital: float
    current_positions: Dict[str, float]  # symbol -> position_size
    unrealized_pnl: Dict[str, float]    # symbol -> unrealized P&L
    daily_pnl: float
    max_daily_loss: float
    portfolio_heat: float               # 0.0 - 1.0
    correlation_matrix: Optional[pd.DataFrame] = None


@dataclass  
class SizingContext:
    """Contexto para decisiones de sizing"""
    signal: TacticalSignal
    portfolio_state: PortfolioState
    market_features: MarketFeatures
    risk_metrics: RiskMetrics
    regime: str                         # trending/ranging/volatile
    volatility_forecast: float          # expected volatility


class BaseSizer(ABC):
    """Base class para position sizers"""
    
    def __init__(self, config: L2Config):
        self.config = config
        
    @abstractmethod
    def calculate_position_size(self, context: SizingContext) -> PositionSize:
        """Calcular tamaño de posición basado en contexto"""
        pass


class KellyCriterionSizer(BaseSizer):
    """Kelly Criterion con ajustes prácticos"""
    
    def __init__(self, config: L2Config, fraction: float = 0.25):
        super().__init__(config)
        self.kelly_fraction = fraction
        self.min_win_rate = 0.51
        self.lookback_trades = 50
        
    def calculate_position_size(self, context: SizingContext) -> PositionSize:
        """
        Kelly Criterion: f* = (bp - q) / b
        Donde:
        - b = odds received (avg_win / avg_loss)
        - p = win probability
        - q = loss probability (1-p)
        """
        logger.info(f"Calculating Kelly position size for {context.signal.symbol}")
        signal = context.signal
        portfolio = context.portfolio_state
        
        # Calcular estadísticas históricas (simulado por ahora)
        win_rate = max(signal.confidence, self.min_win_rate)
        avg_win_pct = self._estimate_avg_win(signal, context.market_features)
        avg_loss_pct = self._estimate_avg_loss(signal, context.risk_metrics)
        
        # Kelly calculation
        b = avg_win_pct / abs(avg_loss_pct) if avg_loss_pct != 0 else 2.0
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, min(kelly_fraction, 1.0))  # Clamp [0,1]
        
        # Aplicar fracción conservadora
        adjusted_fraction = kelly_fraction * self.kelly_fraction
        
        # Calcular tamaño base
        risk_amount = portfolio.available_capital * adjusted_fraction
        
        # Considerar stop loss para calcular shares
        stop_distance = abs(signal.price - (signal.stop_loss or signal.price * 0.98))
        stop_distance_pct = stop_distance / signal.price
        
        # Position size = risk_amount / stop_distance_pct
        raw_position_size = risk_amount / (stop_distance_pct * signal.price) if stop_distance_pct > 0 else 0
        
        logger.info(
            f"Kelly sizing for {signal.symbol}: "
            f"win_rate={win_rate:.2f}, b={b:.2f}, "
            f"kelly_f={kelly_fraction:.3f}, adj_f={adjusted_fraction:.3f}, "
            f"size={raw_position_size:.4f}"
        )
        
        return PositionSize(
            symbol=signal.symbol,
            size=raw_position_size,
            notional=raw_position_size * signal.price,
            sizing_method="kelly_criterion",
            confidence=signal.confidence * 0.8,  # Discount for Kelly uncertainty
            risk_amount=risk_amount,
            metadata={
                "kelly_fraction": kelly_fraction,
                "adjusted_fraction": adjusted_fraction,
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct
            }
        )
    
    def _estimate_avg_win(self, signal: TacticalSignal, features: MarketFeatures) -> float:
        """Estimar average win percentage basado en señal y features"""
        base_win = 0.02  # 2% base win
        
        # Ajustar por strength y confidence
        strength_adj = signal.strength * 0.01
        confidence_adj = signal.confidence * 0.01
        
        # Ajustar por volatilidad (mayor vol -> mayor potential win)
        vol_adj = features.volatility * 0.5 if features.volatility else 0
        
        return min(base_win + strength_adj + confidence_adj + vol_adj, 0.08)  # Cap at 8%
    
    def _estimate_avg_loss(self, signal: TacticalSignal, risk_metrics: RiskMetrics) -> float:
        """Estimar average loss percentage"""
        base_loss = -0.015  # 1.5% base loss
        
        # Ajustar por VaR si disponible
        if risk_metrics.var_95:
            var_adj = risk_metrics.var_95 * 0.5
            return max(base_loss + var_adj, -0.05)  # Cap at 5% loss
        
        return base_loss


class VolTargetingSizer(BaseSizer):
    """Volatility targeting position sizer"""
    
    def __init__(self, config: L2Config, target_vol: float = 0.15):
        super().__init__(config)
        self.target_vol = target_vol
        self.vol_lookback = 20
        
    def calculate_position_size(self, context: SizingContext) -> PositionSize:
        """Size position to achieve target portfolio volatility"""
        logger.info(f"Calculating Vol Targeting position size for {context.signal.symbol}")
        signal = context.signal
        portfolio = context.portfolio_state
        
        # Usar volatilidad forecasted o estimar
        asset_vol = context.volatility_forecast or context.market_features.volatility or 0.2
        
        # Target vol adjustment
        vol_scalar = self.target_vol / asset_vol if asset_vol > 0 else 0.5
        vol_scalar = min(vol_scalar, 2.0)  # Cap leverage
        
        # Base position size como % del portfolio
        base_position_pct = 0.1 * vol_scalar  # 10% base * vol adjustment
        
        # Ajustar por confidence
        confidence_adj = signal.confidence ** 0.5  # Square root for smoother scaling
        
        # Final position percentage
        position_pct = base_position_pct * confidence_adj
        position_pct = min(position_pct, 0.25)  # Cap at 25%
        
        # Calculate position size
        position_notional = portfolio.available_capital * position_pct
        position_size = position_notional / signal.price
        
        logger.info(
            f"Vol targeting for {signal.symbol}: "
            f"asset_vol={asset_vol:.3f}, target_vol={self.target_vol:.3f}, "
            f"vol_scalar={vol_scalar:.2f}, position_pct={position_pct:.3f}, "
            f"size={position_size:.4f}"
        )
        
        return PositionSize(
            symbol=signal.symbol,
            size=position_size,
            notional=position_notional,
            sizing_method="vol_targeting",
            confidence=signal.confidence,
            risk_amount=position_notional * 0.02,  # Assume 2% risk
            metadata={
                "target_vol": self.target_vol,
                "asset_vol": asset_vol,
                "vol_scalar": vol_scalar,
                "position_pct": position_pct
            }
        )


class RiskParitySizer(BaseSizer):
    """Risk parity position sizing"""
    
    def __init__(self, config: L2Config):
        super().__init__(config)
        self.risk_budget = 0.02  # 2% risk per position
        
    def calculate_position_size(self, context: SizingContext) -> PositionSize:
        """Size position based on equal risk contribution"""
        logger.info(f"Calculating Risk Parity position size for {context.signal.symbol}")
        signal = context.signal
        portfolio = context.portfolio_state
        
        # Calculate risk budget for this position
        available_risk_budget = portfolio.available_capital * self.risk_budget
        
        # Adjust by portfolio heat (reduce size if portfolio is hot)
        heat_adjustment = 1.0 - (portfolio.portfolio_heat * 0.5)
        adjusted_risk_budget = available_risk_budget * heat_adjustment
        
        # Calculate position size based on stop distance
        stop_price = signal.stop_loss or (signal.price * 0.98)  # Default 2% stop
        stop_distance = abs(signal.price - stop_price)
        
        # Position size = risk_budget / stop_distance
        position_size = adjusted_risk_budget / stop_distance if stop_distance > 0 else 0
        
        # Apply confidence scaling
        position_size *= signal.confidence
        
        logger.info(
            f"Risk parity for {signal.symbol}: "
            f"risk_budget={adjusted_risk_budget:.2f}, "
            f"stop_distance={stop_distance:.2f}, "
            f"heat_adj={heat_adjustment:.2f}, "
            f"size={position_size:.4f}"
        )
        
        return PositionSize(
            symbol=signal.symbol,
            size=position_size,
            notional=position_size * signal.price,
            sizing_method="risk_parity",
            confidence=signal.confidence,
            risk_amount=adjusted_risk_budget,
            metadata={
                "risk_budget": self.risk_budget,
                "heat_adjustment": heat_adjustment,
                "stop_distance": stop_distance,
                "portfolio_heat": portfolio.portfolio_heat
            }
        )


class EnsembleSizer(BaseSizer):
    """Ensemble de múltiples sizing methods"""
    
    def __init__(self, config: L2Config):
        super().__init__(config)
        
        # Initialize individual sizers
        self.kelly_sizer = KellyCriterionSizer(config, fraction=0.25)
        self.vol_sizer = VolTargetingSizer(config, target_vol=0.15)
        self.risk_parity_sizer = RiskParitySizer(config)
        
        # Weights for ensemble (can be dynamic)
        self.weights = {
            "kelly": 0.4,
            "vol_targeting": 0.35,
            "risk_parity": 0.25
        }
        
    def calculate_position_size(self, context: SizingContext) -> PositionSize:
        """Combine multiple sizing approaches"""
        logger.info(f"Calculating Ensemble position size for {context.signal.symbol}")
        
        # Get individual size recommendations
        kelly_size = self.kelly_sizer.calculate_position_size(context)
        vol_size = self.vol_sizer.calculate_position_size(context)
        rp_size = self.risk_parity_sizer.calculate_position_size(context)
        
        sizes = [kelly_size.size, vol_size.size, rp_size.size]
        weights = list(self.weights.values())
        
        # Weighted average
        ensemble_size = np.average(sizes, weights=weights)
        
        # Take average of risk amounts
        avg_risk_amount = np.mean([kelly_size.risk_amount, vol_size.risk_amount, rp_size.risk_amount])
        
        # Confidence as weighted average
        confidences = [kelly_size.confidence, vol_size.confidence, rp_size.confidence]
        ensemble_confidence = np.average(confidences, weights=weights)
        
        logger.info(
            f"Ensemble sizing for {context.signal.symbol}: "
            f"kelly={kelly_size.size:.4f}, vol={vol_size.size:.4f}, "
            f"rp={rp_size.size:.4f}, ensemble={ensemble_size:.4f}"
        )
        
        return PositionSize(
            symbol=context.signal.symbol,
            size=ensemble_size,
            notional=ensemble_size * context.signal.price,
            sizing_method="ensemble",
            confidence=ensemble_confidence,
            risk_amount=avg_risk_amount,
            metadata={
                "individual_sizes": {
                    "kelly": kelly_size.size,
                    "vol_targeting": vol_size.size,
                    "risk_parity": rp_size.size
                },
                "weights": self.weights,
                "individual_confidences": confidences
            }
        )


class PositionSizerManager:
    """Manager principal para position sizing en L2"""
    
    def __init__(self, config: L2Config):
        self.config = config
        
        # Initialize sizer based on config
        sizing_method = config.sizing_method.lower()
        
        if sizing_method == "kelly":
            self.sizer = KellyCriterionSizer(config)
        elif sizing_method == "vol_targeting":
            self.sizer = VolTargetingSizer(config)
        elif sizing_method == "risk_parity":
            self.sizer = RiskParitySizer(config)
        else:  # Default to ensemble
            self.sizer = EnsembleSizer(config)
        
        # Position limits and checks
        self.max_position_pct = config.max_position_pct or 0.1
        self.max_total_exposure = config.max_total_exposure or 0.8
        self.min_position_notional = config.min_position_notional or 100.0
        
        logger.info(f"Initialized PositionSizerManager with method: {sizing_method}")
        
    def calculate_position_size(
        self,
        signal: TacticalSignal,
        portfolio_state: PortfolioState,
        market_features: MarketFeatures,
        risk_metrics: RiskMetrics,
        regime: str = "neutral"
    ) -> Optional[PositionSize]:
        """
        Calcular position size con todos los checks y validaciones
        """
        logger.info(f"Calculating final position size for {signal.symbol}")
        
        # Create sizing context
        context = SizingContext(
            signal=signal,
            portfolio_state=portfolio_state,
            market_features=market_features,
            risk_metrics=risk_metrics,
            regime=regime,
            volatility_forecast=market_features.volatility or 0.2
        )
        
        # Get raw position size
        raw_position = self.sizer.calculate_position_size(context)
        
        # Apply position limits and checks
        adjusted_position = self._apply_position_limits(raw_position, portfolio_state)
        
        if adjusted_position is None:
            logger.warning(f"Position rejected for {signal.symbol} after limit checks")
            return None
            
        # Final validation
        if not self._validate_position(adjusted_position, portfolio_state):
            logger.warning(f"Position failed final validation for {signal.symbol}")
            return None
            
        logger.info(
            f"Final position size for {signal.symbol}: "
            f"{adjusted_position.size:.4f} (${adjusted_position.notional:.2f}) "
            f"via {adjusted_position.sizing_method}"
        )
        
        return adjusted_position
    
    def _apply_position_limits(
        self,
        position: PositionSize,
        portfolio_state: PortfolioState
    ) -> Optional[PositionSize]:
        """Aplicar límites de posición"""
        logger.info(f"Applying position limits for {position.symbol}")
        
        # Check minimum notional
        if position.notional < self.min_position_notional:
            logger.info(f"Position too small: ${position.notional:.2f} < ${self.min_position_notional}")
            return None
        
        # Check maximum position percentage
        position_pct = position.notional / portfolio_state.total_capital
        if position_pct > self.max_position_pct:
            # Scale down position
            scale_factor = self.max_position_pct / position_pct
            position.size *= scale_factor
            position.notional *= scale_factor
            position.risk_amount *= scale_factor
            
            logger.info(
                f"Scaled down position for {position.symbol}: "
                f"factor={scale_factor:.2f}, new_size={position.size:.4f}"
            )
        
        # Check total portfolio exposure
        current_exposure = sum(abs(pos) * price for pos, price in 
                             zip(portfolio_state.current_positions.values(), 
                                 [50000] * len(portfolio_state.current_positions)))  # Simplified
        new_exposure = current_exposure + position.notional
        exposure_pct = new_exposure / portfolio_state.total_capital
        
        if exposure_pct > self.max_total_exposure:
            # Reduce position to fit exposure limit
            available_exposure = self.max_total_exposure * portfolio_state.total_capital - current_exposure
            if available_exposure <= self.min_position_notional:
                logger.warning("No available exposure capacity")
                return None
                
            scale_factor = available_exposure / position.notional
            position.size *= scale_factor
            position.notional *= scale_factor
            position.risk_amount *= scale_factor
            
            logger.info(
                f"Reduced position for exposure limit: "
                f"scale_factor={scale_factor:.2f}, new_notional=${position.notional:.2f}"
            )
        
        return position
    
    def _validate_position(
        self,
        position: PositionSize,
        portfolio_state: PortfolioState
    ) -> bool:
        """Validación final de posición"""
        logger.info(f"Validating position for {position.symbol}")
        
        # Check for NaN or invalid values
        if not all([
            np.isfinite(position.size),
            np.isfinite(position.notional),
            position.size > 0,
            position.notional > 0
        ]):
            return False
        
        # Check portfolio heat doesn't exceed limits
        if portfolio_state.portfolio_heat > 0.9:  # 90% heat limit
            logger.warning("Portfolio too hot, rejecting new positions")
            return False
        
        # Check daily loss limits
        if portfolio_state.daily_pnl < portfolio_state.max_daily_loss:
            logger.warning("Daily loss limit exceeded")
            return False
        
        return True
    
    def update_sizer_weights(self, performance_data: Dict[str, float]):
        """Actualizar pesos del ensemble basado en performance"""
        logger.info("Updating sizer weights")
        if hasattr(self.sizer, 'weights'):
            # Simple performance-based weight adjustment
            total_perf = sum(performance_data.values())
            if total_perf > 0:
                for method, perf in performance_data.items():
                    if method in self.sizer.weights:
                        # Gradually adjust weights toward better performers
                        current_weight = self.sizer.weights[method]
                        target_weight = perf / total_perf
                        self.sizer.weights[method] = 0.9 * current_weight + 0.1 * target_weight
                
                # Normalize weights
                total_weight = sum(self.sizer.weights.values())
                for method in self.sizer.weights:
                    self.sizer.weights[method] /= total_weight
                    
                logger.info(f"Updated sizer weights: {self.sizer.weights}")


# Utility functions
def calculate_correlation_adjustment(
    symbol: str,
    positions: Dict[str, float],
    correlation_matrix: Optional[pd.DataFrame] = None
) -> float:
    """Calcular ajuste por correlación entre posiciones"""
    logger.info(f"Calculating correlation adjustment for {symbol}")
    if not correlation_matrix or symbol not in correlation_matrix.index:
        return 1.0
    
    adjustment = 1.0
    for other_symbol, position in positions.items():
        if other_symbol != symbol and other_symbol in correlation_matrix.columns:
            correlation = correlation_matrix.loc[symbol, other_symbol]
            position_weight = abs(position) / 100000  # Normalize by typical position size
            adjustment -= abs(correlation) * position_weight * 0.1  # 10% max adjustment
    
    return max(adjustment, 0.5)  # Minimum 50% of original size


def estimate_portfolio_heat(
    positions: Dict[str, float],
    prices: Dict[str, float],
    total_capital: float
) -> float:
    """Estimar 'heat' del portfolio (0.0 - 1.0)"""
    logger.info("Estimating portfolio heat")
    total_exposure = sum(
        abs(positions.get(symbol, 0)) * prices.get(symbol, 0) 
        for symbol in set(positions.keys()) | set(prices.keys())
    )
    
    heat = total_exposure / total_capital if total_capital > 0 else 0
    return min(heat, 1.0)


# Example usage and testing
if __name__ == "__main__":
    from .config import L2Config
    
    # Demo configuration
    config = L2Config()
    config.sizing_method = "ensemble"
    config.max_position_pct = 0.15
    config.max_total_exposure = 0.8
    
    # Create manager
    sizer_manager = PositionSizerManager(config)
    
    # Demo signal
    demo_signal = TacticalSignal(
        symbol="BTC/USDT",
        side="buy",
        strength=0.8,
        confidence=0.75,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        timestamp=datetime.now(),
        source="ensemble",
        reasoning={"demo": True}
    )
    
    # Demo portfolio state
    portfolio = PortfolioState(
        total_capital=100000.0,
        available_capital=80000.0,
        current_positions={"ETH/USDT": 1.0},
        unrealized_pnl={"ETH/USDT": 200.0},
        daily_pnl=150.0,
        max_daily_loss=-5000.0,
        portfolio_heat=0.3
    )
    
    # Demo market features  
    features = MarketFeatures(
        volatility=0.25,
        volume_ratio=1.2,
        price_momentum=0.1
    )
    
    # Demo risk metrics
    risk = RiskMetrics(
        var_95=-0.025,
        expected_shortfall=-0.035,
        max_drawdown=0.12
    )
    
    # Calculate position size
    position = sizer_manager.calculate_position_size(
        signal=demo_signal,
        portfolio_state=portfolio,
        market_features=features,
        risk_metrics=risk,
        regime="trending"
    )
    
    if position:
        print(f"Calculated position: {position}")
    else:
        print("Position was rejected")