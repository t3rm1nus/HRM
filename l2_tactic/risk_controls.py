"""
risk_controls.py - L2 Tactical Risk Management

Implementa controles de riesgo dinámicos para L2:
- Dynamic stop-loss y take-profit
- Correlation risk management  
- Portfolio heat monitoring
- Drawdown protection
- Position sizing limits
- Real-time risk metrics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .models import TacticalSignal, PositionSize, RiskMetrics, MarketFeatures
from .config import L2Config


logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Niveles de riesgo del sistema"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipos de alertas de riesgo"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    CORRELATION_LIMIT = "correlation_limit"
    PORTFOLIO_HEAT = "portfolio_heat"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class RiskAlert:
    """Alerta de riesgo"""
    alert_type: AlertType
    severity: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (f"[{self.severity.value.upper()}] {self.alert_type.value} "
                f"for {self.symbol}: {self.message}")


@dataclass
class StopLossOrder:
    """Orden de stop loss dinámico"""
    symbol: str
    stop_price: float
    original_price: float
    entry_price: float
    position_size: float
    stop_type: str  # "fixed", "trailing", "atr", "volatility"
    last_updated: datetime
    trail_amount: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RiskPosition:
    """Posición con métricas de riesgo"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: float = 0.0
    time_in_position: timedelta = field(default_factory=lambda: timedelta())
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE


class DynamicStopLoss:
    """Gestor de stop-loss dinámicos"""
    
    def __init__(self, config: L2Config):
        self.config = config
        self.default_stop_pct = config.default_stop_pct or 0.02  # 2%
        self.atr_multiplier = config.atr_multiplier or 2.0
        self.trailing_stop_pct = config.trailing_stop_pct or 0.01  # 1%
        self.breakeven_threshold = config.breakeven_threshold or 1.5  # 1.5R
        
        self.active_stops: Dict[str, StopLossOrder] = {}
        
    def calculate_initial_stop(
        self,
        signal: TacticalSignal,
        market_features: MarketFeatures,
        position: RiskPosition
    ) -> float:
        """Calcular stop loss inicial basado en múltiples factores"""
        
        price = signal.price
        side = signal.side
        
        # Method 1: Fixed percentage stop
        fixed_stop_pct = self._calculate_fixed_stop_pct(signal, market_features)
        if side == "buy":
            fixed_stop = price * (1 - fixed_stop_pct)
        else:  # sell
            fixed_stop = price * (1 + fixed_stop_pct)
            
        # Method 2: ATR-based stop
        atr_stop = self._calculate_atr_stop(price, market_features, side)
        
        # Method 3: Volatility-adjusted stop
        vol_stop = self._calculate_volatility_stop(price, market_features, side)
        
        # Method 4: Support/Resistance based stop
        sr_stop = self._calculate_support_resistance_stop(price, market_features, side)
        
        # Combine methods (weighted average)
        stops = [fixed_stop, atr_stop, vol_stop, sr_stop]
        weights = [0.3, 0.3, 0.2, 0.2]  # Pesos configurables
        
        # Filter out None values and adjust weights
        valid_stops = [(stop, weight) for stop, weight in zip(stops, weights) if stop is not None]
        if not valid_stops:
            final_stop = fixed_stop
        else:
            total_weight = sum(weight for _, weight in valid_stops)
            final_stop = sum(stop * weight for stop, weight in valid_stops) / total_weight
        
        # Ensure stop is reasonable (not too tight or too wide)
        stop_distance_pct = abs(final_stop - price) / price
        stop_distance_pct = max(0.005, min(stop_distance_pct, 0.05))  # 0.5% - 5%
        
        if side == "buy":
            final_stop = price * (1 - stop_distance_pct)
        else:
            final_stop = price * (1 + stop_distance_pct)
        
        logger.info(
            f"Calculated initial stop for {signal.symbol}: "
            f"price={price:.2f}, stop={final_stop:.2f} "
            f"({stop_distance_pct*100:.2f}% distance)"
        )
        
        return final_stop
    
    def _calculate_fixed_stop_pct(
        self,
        signal: TacticalSignal,
        market_features: MarketFeatures
    ) -> float:
        """Calcular stop percentage fijo ajustado por confianza"""
        base_stop = self.default_stop_pct
        
        # Ajustar por confidence (mayor confidence -> stop más ajustado)
        confidence_adj = (1 - signal.confidence) * 0.01  # Max 1% adjustment
        
        # Ajustar por volatilidad
        vol_adj = 0
        if market_features.volatility:
            vol_adj = (market_features.volatility - 0.2) * 0.5  # Adjust for vol above 20%
            vol_adj = max(0, min(vol_adj, 0.02))  # Cap at 2%
        
        return base_stop + confidence_adj + vol_adj
    
    def _calculate_atr_stop(
        self,
        price: float,
        market_features: MarketFeatures,
        side: str
    ) -> Optional[float]:
        """Calcular stop basado en ATR"""
        if not hasattr(market_features, 'atr') or not market_features.atr:
            return None
        
        atr_distance = market_features.atr * self.atr_multiplier
        
        if side == "buy":
            return price - atr_distance
        else:
            return price + atr_distance
    
    def _calculate_volatility_stop(
        self,
        price: float,
        market_features: MarketFeatures,
        side: str
    ) -> Optional[float]:
        """Calcular stop basado en volatilidad realizada"""
        if not market_features.volatility:
            return None
        
        # Convert annual vol to daily
        daily_vol = market_features.volatility / np.sqrt(252)
        vol_distance_pct = daily_vol * 2.0  # 2 standard deviations
        
        if side == "buy":
            return price * (1 - vol_distance_pct)
        else:
            return price * (1 + vol_distance_pct)
    
    def _calculate_support_resistance_stop(
        self,
        price: float,
        market_features: MarketFeatures,
        side: str
    ) -> Optional[float]:
        """Calcular stop basado en soporte/resistencia"""
        if not hasattr(market_features, 'support') or not hasattr(market_features, 'resistance'):
            return None
        
        support = getattr(market_features, 'support', None)
        resistance = getattr(market_features, 'resistance', None)
        
        if side == "buy" and support:
            # Stop slightly below support
            return support * 0.995
        elif side == "sell" and resistance:
            # Stop slightly above resistance
            return resistance * 1.005
        
        return None
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        position: RiskPosition
    ) -> Optional[float]:
        """Actualizar trailing stop si es beneficioso"""
        
        if symbol not in self.active_stops:
            return None
        
        stop_order = self.active_stops[symbol]
        if stop_order.stop_type != "trailing":
            return None
        
        old_stop = stop_order.stop_price
        entry_price = position.entry_price
        trail_pct = self.trailing_stop_pct
        
        # Determinar dirección de la posición
        is_long = position.size > 0
        
        if is_long:
            # Para posiciones largas, subir el stop si precio sube
            if current_price > entry_price:  # En ganancia
                new_stop = current_price * (1 - trail_pct)
                if new_stop > old_stop:
                    stop_order.stop_price = new_stop
                    stop_order.last_updated = datetime.now()
                    logger.info(f"Updated trailing stop for {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    return new_stop
        else:
            # Para posiciones cortas, bajar el stop si precio baja
            if current_price < entry_price:  # En ganancia
                new_stop = current_price * (1 + trail_pct)
                if new_stop < old_stop:
                    stop_order.stop_price = new_stop
                    stop_order.last_updated = datetime.now()
                    logger.info(f"Updated trailing stop for {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    return new_stop
        
        return None
    
    def should_move_to_breakeven(
        self,
        position: RiskPosition,
        current_price: float
    ) -> bool:
        """Determinar si mover stop a breakeven"""
        
        entry_price = position.entry_price
        is_long = position.size > 0
        
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Si estamos en ganancia por encima del threshold, mover a breakeven
        if profit_pct >= (self.breakeven_threshold * self.default_stop_pct):
            return True
            
        return False


class PortfolioRiskManager:
    """Gestor de riesgo a nivel portfolio"""
    
    def __init__(self, config: L2Config):
        self.config = config
        self.max_correlation = config.max_correlation or 0.7
        self.max_portfolio_heat = config.max_portfolio_heat or 0.8
        self.daily_loss_limit = config.daily_loss_limit or 0.05  # 5%
        self.max_drawdown_limit = config.max_drawdown_limit or 0.15  # 15%
        self.max_positions = config.max_positions or 5
        
        self.risk_alerts: List[RiskAlert] = []
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        
    def check_correlation_risk(
        self,
        new_signal: TacticalSignal,
        current_positions: Dict[str, RiskPosition],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, List[RiskAlert]]:
        """Verificar riesgo de correlación antes de nueva posición"""
        
        alerts = []
        
        if not correlation_matrix or new_signal.symbol not in correlation_matrix.index:
            return True, alerts  # No data, allow position
        
        for pos_symbol, position in current_positions.items():
            if pos_symbol in correlation_matrix.columns:
                correlation = abs(correlation_matrix.loc[new_signal.symbol, pos_symbol])
                
                if correlation > self.max_correlation:
                    alert = RiskAlert(
                        alert_type=AlertType.CORRELATION_LIMIT,
                        severity=RiskLevel.HIGH,
                        symbol=new_signal.symbol,
                        message=f"High correlation ({correlation:.2f}) with existing position {pos_symbol}",
                        current_value=correlation,
                        threshold=self.max_correlation,
                        timestamp=datetime.now(),
                        metadata={"correlated_symbol": pos_symbol}
                    )
                    alerts.append(alert)
        
        # Si hay correlaciones altas, rechazar la posición
        high_correlation_alerts = [a for a in alerts if a.severity == RiskLevel.HIGH]
        allow_position = len(high_correlation_alerts) == 0
        
        return allow_position, alerts
    
    def calculate_portfolio_heat(
        self,
        positions: Dict[str, RiskPosition],
        total_capital: float
    ) -> float:
        """Calcular 'heat' total del portfolio"""
        
        total_risk = sum(pos.risk_amount for pos in positions.values())
        heat = total_risk / total_capital if total_capital > 0 else 0
        
        return min(heat, 1.0)
    
    def check_portfolio_limits(
        self,
        positions: Dict[str, RiskPosition],
        total_capital: float,
        daily_pnl: float
    ) -> List[RiskAlert]:
        """Verificar límites del portfolio"""
        
        alerts = []
        
        # Check portfolio heat
        portfolio_heat = self.calculate_portfolio_heat(positions, total_capital)
        if portfolio_heat > self.max_portfolio_heat:
            alert = RiskAlert(
                alert_type=AlertType.PORTFOLIO_HEAT,
                severity=RiskLevel.HIGH if portfolio_heat > 0.9 else RiskLevel.MODERATE,
                symbol="PORTFOLIO",
                message=f"Portfolio heat too high: {portfolio_heat:.2f}",
                current_value=portfolio_heat,
                threshold=self.max_portfolio_heat,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Check daily loss limit
        daily_loss_pct = abs(daily_pnl) / total_capital if daily_pnl < 0 else 0
        if daily_loss_pct > self.daily_loss_limit:
            alert = RiskAlert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                severity=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2%}",
                current_value=daily_loss_pct,
                threshold=self.daily_loss_limit,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Check number of positions
        num_positions = len(positions)
        if num_positions >= self.max_positions:
            alert = RiskAlert(
                alert_type=AlertType.POSITION_SIZE_LIMIT,
                severity=RiskLevel.MODERATE,
                symbol="PORTFOLIO",
                message=f"Maximum positions reached: {num_positions}/{self.max_positions}",
                current_value=num_positions,
                threshold=self.max_positions,
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        return alerts
    
    def check_drawdown_limit(
        self,
        current_portfolio_value: float,
        peak_portfolio_value: float
    ) -> Optional[RiskAlert]:
        """Verificar límite de drawdown"""
        
        drawdown = (peak_portfolio_value - current_portfolio_value) / peak_portfolio_value
        
        if drawdown > self.max_drawdown_limit:
            return RiskAlert(
                alert_type=AlertType.DRAWDOWN_LIMIT,
                severity=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                message=f"Drawdown limit exceeded: {drawdown:.2%}",
                current_value=drawdown,
                threshold=self.max_drawdown_limit,
                timestamp=datetime.now()
            )
        
        return None
    
    def update_daily_pnl(self, pnl: float):
        """Actualizar historial de P&L diario"""
        now = datetime.now()
        self.daily_pnl_history.append((now, pnl))
        
        # Mantener solo últimos 30 días
        cutoff = now - timedelta(days=30)
        self.daily_pnl_history = [
            (date, pnl) for date, pnl in self.daily_pnl_history
            if date > cutoff
        ]
    
    def update_portfolio_value(self, value: float):
        """Actualizar historial de valor del portfolio"""
        now = datetime.now()
        self.portfolio_value_history.append((now, value))
        
        # Mantener solo últimos 90 días
        cutoff = now - timedelta(days=90)
        self.portfolio_value_history = [
            (date, val) for date, val in self.portfolio_value_history
            if date > cutoff
        ]
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Obtener métricas del portfolio"""
        
        if len(self.portfolio_value_history) < 2:
            return {}
        
        values = [val for _, val in self.portfolio_value_history]
        returns = np.diff(values) / values[:-1]
        
        # Peak portfolio value
        peak_value = max(values)
        current_value = values[-1]
        current_drawdown = (peak_value - current_value) / peak_value
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        sharpe = (mean_return * 252) / volatility if volatility > 0 else 0
        
        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": self._calculate_max_drawdown(values),
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "total_return": (current_value - values[0]) / values[0] if values[0] > 0 else 0
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calcular máximo drawdown histórico"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd


class VolatilityMonitor:
    """Monitor de volatilidad y spikes"""
    
    def __init__(self, config: L2Config):
        self.config = config
        self.vol_spike_threshold = config.vol_spike_threshold or 2.0  # 2x normal vol
        self.vol_window = config.vol_window or 20
        
        self.volatility_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def update_volatility(self, symbol: str, volatility: float):
        """Actualizar volatilidad para un símbolo"""
        now = datetime.now()
        
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
        
        self.volatility_history[symbol].append((now, volatility))
        
        # Mantener solo las últimas N observaciones
        self.volatility_history[symbol] = self.volatility_history[symbol][-self.vol_window:]
    
    def check_volatility_spike(self, symbol: str, current_vol: float) -> Optional[RiskAlert]:
        """Verificar si hay un spike de volatilidad"""
        
        if symbol not in self.volatility_history or len(self.volatility_history[symbol]) < 5:
            return None
        
        # Calcular volatilidad promedio histórica
        historical_vols = [vol for _, vol in self.volatility_history[symbol][:-1]]
        avg_vol = np.mean(historical_vols)
        
        # Verificar spike
        if current_vol > avg_vol * self.vol_spike_threshold:
            return RiskAlert(
                alert_type=AlertType.VOLATILITY_SPIKE,
                severity=RiskLevel.HIGH,
                symbol=symbol,
                message=f"Volatility spike detected: {current_vol:.3f} vs avg {avg_vol:.3f}",
                current_value=current_vol,
                threshold=avg_vol * self.vol_spike_threshold,
                timestamp=datetime.now(),
                metadata={
                    "avg_volatility": avg_vol,
                    "spike_ratio": current_vol / avg_vol
                }
            )
        
        return None


class RiskControlManager:
    """Manager principal de controles de riesgo L2"""
    
    def __init__(self, config: L2Config):
        self.config = config
        
        # Initialize components
        self.stop_loss_manager = DynamicStopLoss(config)
        self.portfolio_manager = PortfolioRiskManager(config)
        self.volatility_monitor = VolatilityMonitor(config)
        
        # Risk state
        self.current_positions: Dict[str, RiskPosition] = {}
        self.active_alerts: List[RiskAlert] = []
        
        logger.info("Initialized RiskControlManager")
    
    def evaluate_pre_trade_risk(
        self,
        signal: TacticalSignal,
        position_size: PositionSize,
        market_features: MarketFeatures,
        portfolio_state: Dict,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, List[RiskAlert], Optional[PositionSize]]:
        """
        Evaluación completa de riesgo antes de ejecutar trade
        
        Returns:
            - bool: Permitir trade
            - List[RiskAlert]: Alertas generadas
            - Optional[PositionSize]: Posición ajustada por riesgo
        """
        
        all_alerts = []
        adjusted_position = position_size
        
        # 1. Check correlation risk
        allow_correlation, correlation_alerts = self.portfolio_manager.check_correlation_risk(
            signal, self.current_positions, correlation_matrix
        )
        all_alerts.extend(correlation_alerts)
        
        # 2. Check portfolio limits
        total_capital = portfolio_state.get("total_capital", 100000)
        daily_pnl = portfolio_state.get("daily_pnl", 0)
        
        limit_alerts = self.portfolio_manager.check_portfolio_limits(
            self.current_positions, total_capital, daily_pnl
        )
        all_alerts.extend(limit_alerts)
        
        # 3. Check volatility spike
        vol_alert = self.volatility_monitor.check_volatility_spike(
            signal.symbol, market_features.volatility or 0.2
        )
        if vol_alert:
            all_alerts.append(vol_alert)
        
        # 4. Adjust position size based on risk alerts
        if any(alert.severity == RiskLevel.HIGH for alert in all_alerts):
            # Reduce position size by 50% for high risk
            adjusted_position.size *= 0.5
            adjusted_position.notional *= 0.5
            adjusted_position.risk_amount *= 0.5
            adjusted_position.metadata["risk_adjustment"] = "reduced_50pct_high_risk"
            
            logger.warning(f"Reduced position size for {signal.symbol} due to high risk alerts")
        
        # 5. Final decision - block trade if critical alerts
        critical_alerts = [a for a in all_alerts if a.severity == RiskLevel.CRITICAL]
        allow_trade = len(critical_alerts) == 0 and allow_correlation
        
        # Update alerts
        self.active_alerts.extend(all_alerts)
        
        return allow_trade, all_alerts, adjusted_position if allow_trade else None
    
    def monitor_existing_positions(
        self,
        market_data: Dict[str, float],
        portfolio_value: float
    ) -> List[RiskAlert]:
        """Monitorear posiciones existentes y generar alertas"""
        
        alerts = []
        
        for symbol, position in self.current_positions.items():
            current_price = market_data.get(symbol)
            if not current_price:
                continue
            
            # Update position P&L
            position.current_price = current_price
            if position.size > 0:  # Long position
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:  # Short position
                position.unrealized_pnl = (position.entry_price - current_price) * abs(position.size)
            
            position.unrealized_pnl_pct = position.unrealized_pnl / (position.entry_price * abs(position.size))
            
            # Update MAE/MFE
            if position.unrealized_pnl < 0:
                position.max_adverse_excursion = min(position.max_adverse_excursion, position.unrealized_pnl)
            else:
                position.max_favorable_excursion = max(position.max_favorable_excursion, position.unrealized_pnl)
            
            # Check stop loss trigger
            stop_alert = self._check_stop_loss_trigger(symbol, position, current_price)
            if stop_alert:
                alerts.append(stop_alert)
            
            # Update trailing stops
            self.stop_loss_manager.update_trailing_stop(symbol, current_price, position)
            
            # Check take profit
            if position.take_profit:
                take_profit_triggered = (
                    (position.size > 0 and current_price >= position.take_profit) or
                    (position.size < 0 and current_price <= position.take_profit)
                )
                
                if take_profit_triggered:
                    alert = RiskAlert(
                        alert_type=AlertType.TAKE_PROFIT,
                        severity=RiskLevel.LOW,
                        symbol=symbol,
                        message=f"Take profit triggered at {current_price:.2f}",
                        current_value=current_price,
                        threshold=position.take_profit,
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _check_stop_loss_trigger(
        self,
        symbol: str,
        position: RiskPosition,
        current_price: float
    ) -> Optional[RiskAlert]:
        """Verificar si se disparó el stop loss"""
        
        if not position.stop_loss:
            return None
        
        stop_triggered = False
        
        if position.size > 0:  # Long position
            stop_triggered = current_price <= position.stop_loss
        else:  # Short position
            stop_triggered = current_price >= position.stop_loss
        
        if stop_triggered:
            return RiskAlert(
                alert_type=AlertType.STOP_LOSS,
                severity=RiskLevel.HIGH,
                symbol=symbol,
                message=f"Stop loss triggered at {current_price:.2f}",
                current_value=current_price,
                threshold=position.stop_loss,
                timestamp=datetime.now(),
                metadata={
                    "position_size": position.size,
                    "unrealized_pnl": position.unrealized_pnl
                }
            )
        
        return None
    
    def add_position(
        self,
        signal: TacticalSignal,
        position_size: PositionSize,
        market_features: MarketFeatures
    ):
        """Agregar nueva posición al tracking"""
        
        # Create risk position
        risk_position = RiskPosition(
            symbol=signal.symbol,
            size=position_size.size if signal.side == "buy" else -position_size.size,
            entry_price=signal.price,
            current_price=signal.price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=position_size.risk_amount
        )
        
        self.current_positions[signal.symbol] = risk_position
        
        # Create initial stop loss order
        if signal.stop_loss:
            stop_order = StopLossOrder(
                symbol=signal.symbol,
                stop_price=signal.stop_loss,
                original_price=signal.stop_loss,
                entry_price=signal.price,
                position_size=position_size.size,
                stop_type="fixed",  # Can be updated later
                last_updated=datetime.now()
            )
            self.stop_loss_manager.active_stops[signal.symbol] = stop_order
        
        logger.info(f"Added position to risk tracking: {signal.symbol} @ {signal.price:.2f}")
    
    def remove_position(self, symbol: str):
        """Remover posición del tracking"""
        
        if symbol in self.current_positions:
            del self.current_positions[symbol]
        
        if symbol in self.stop_loss_manager.active_stops:
            del self.stop_loss_manager.active_stops[symbol]
        
        logger.info(f"Removed position from risk tracking: {symbol}")
    
    def get_portfolio_risk_summary(self) -> Dict:
        """Obtener resumen de riesgo del portfolio"""
        
        total_positions = len(self.current_positions)
        total_risk = sum(pos.risk_amount for pos in self.current_positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.current_positions.values())
        
        # Count alerts by severity
        alert_counts = {}
        for severity in RiskLevel:
            alert_counts[severity.value] = sum(
                1 for alert in self.active_alerts 
                if alert.severity == severity
            )
        
        # Portfolio metrics
        portfolio_metrics = self.portfolio_manager.get_portfolio_metrics()
        
        return {
            "positions_count": total_positions,
            "total_risk_amount": total_risk,
            "total_unrealized_pnl": total_unrealized,
            "alert_counts": alert_counts,
            "portfolio_metrics": portfolio_metrics,
            "active_stops": len(self.stop_loss_manager.active_stops),
            "last_update": datetime.now().isoformat()
        }
    
    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Limpiar alertas antiguas"""
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert.timestamp > cutoff
        ]


# Utility functions
def calculate_position_var(
    position_size: float,
    price: float,
    volatility: float,
    confidence_level: float = 0.95,
    holding_period_days: int = 1
) -> float:
    """Calcular VaR de una posición"""
    
    # Convert annual volatility to holding period
    period_vol = volatility * np.sqrt(holding_period_days / 252)
    
    # Z-score for confidence level
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
    z_score = z_scores.get(confidence_level, 1.65)
    
    # Position value
    position_value = abs(position_size) * price
    
    # VaR calculation
    var = position_value * period_vol * z_score
    
    return var


# Example usage
if __name__ == "__main__":
    from .config import L2Config
    
    # Demo configuration
    config = L2Config()
    config.default_stop_pct = 0.02
    config.max_correlation = 0.7
    config.daily_loss_limit = 0.05
    
    # Create risk manager
    risk_manager = RiskControlManager(config)
    
    # Demo signal and position
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
    
    demo_position = PositionSize(
        symbol="BTC/USDT",
        size=0.1,
        notional=5000.0,
        sizing_method="ensemble",
        confidence=0.75,
        risk_amount=100.0
    )
    
    demo_features = MarketFeatures(
        volatility=0.25,
        volume_ratio=1.2,
        price_momentum=0.1
    )
    
    # Test pre-trade risk evaluation
    allow_trade, alerts, adjusted_position = risk_manager.evaluate_pre_trade_risk(
        signal=demo_signal,
        position_size=demo_position,
        market_features=demo_features,
        portfolio_state={"total_capital": 100000, "daily_pnl": -200}
    )
    
    print(f"Allow trade: {allow_trade}")
    print(f"Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert}")
    
    if adjusted_position:
        print(f"Adjusted position: {adjusted_position.size:.4f}")