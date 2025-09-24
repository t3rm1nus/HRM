"""
Trading Metrics Module - Real-time performance monitoring
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from core.logging import logger

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float
    duration: timedelta

@dataclass
class TradingMetrics:
    """Real-time trading performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))

    # Signal metrics
    signal_accuracy: float = 0.0
    signals_generated: int = 0
    signals_executed: int = 0

    # Portfolio metrics
    peak_portfolio_value: float = 0.0
    current_portfolio_value: float = 0.0
    portfolio_returns: List[float] = field(default_factory=list)

    # Trade history
    completed_trades: List[TradeRecord] = field(default_factory=list)
    active_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __init__(self):
        # Initialize all fields explicitly to ensure they exist
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.avg_trade_duration = timedelta(0)
        self.signal_accuracy = 0.0
        self.signals_generated = 0
        self.signals_executed = 0
        self.peak_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.portfolio_returns = []
        self.completed_trades = []
        self.active_positions = {}

        self.reset_time = datetime.now()
        logger.info("📊 TradingMetrics initialized for real-time monitoring")

    def update_from_orders(self, executed_orders: List[Dict[str, Any]], portfolio_value: float):
        """Update metrics from executed orders"""
        try:
            # Update portfolio value tracking
            self.current_portfolio_value = portfolio_value
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value

            # Calculate current drawdown
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

            # Process executed orders
            for order in executed_orders:
                if order.get('status') == 'filled':
                    self._process_filled_order(order)

            # Update derived metrics
            self._update_derived_metrics()

        except Exception as e:
            logger.error(f"❌ Error updating trading metrics: {e}")

    def _process_filled_order(self, order: Dict[str, Any]):
        """Process a filled order"""
        try:
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            price = order['filled_price']

            # Check if this completes a trade (opposite side exists)
            if symbol in self.active_positions:
                active_pos = self.active_positions[symbol]
                if active_pos['side'] != side:
                    # This closes the position - record trade
                    self._record_completed_trade(active_pos, order)
                    del self.active_positions[symbol]
                else:
                    # This adds to position
                    self._update_position(symbol, order)
            else:
                # This opens a new position
                self.active_positions[symbol] = {
                    'side': side,
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': datetime.fromisoformat(order['timestamp']),
                    'commission': order.get('commission', 0)
                }

        except Exception as e:
            logger.error(f"❌ Error processing filled order: {e}")

    def _update_position(self, symbol: str, order: Dict[str, Any]):
        """Update existing position with new order"""
        try:
            position = self.active_positions[symbol]
            new_quantity = position['quantity'] + order['quantity']

            if new_quantity == 0:
                # Position closed
                del self.active_positions[symbol]
            else:
                # Update average price
                total_value = position['quantity'] * position['entry_price'] + order['quantity'] * order['filled_price']
                position['quantity'] = new_quantity
                position['entry_price'] = total_value / abs(new_quantity)

        except Exception as e:
            logger.error(f"❌ Error updating position for {symbol}: {e}")

    def _record_completed_trade(self, entry_position: Dict[str, Any], exit_order: Dict[str, Any]):
        """Record a completed trade"""
        try:
            symbol = exit_order['symbol']
            exit_price = exit_order['filled_price']
            exit_time = datetime.fromisoformat(exit_order['timestamp'])

            # Calculate P&L
            quantity = abs(entry_position['quantity'])
            entry_price = entry_position['entry_price']
            pnl = (exit_price - entry_price) * quantity if entry_position['side'] == 'buy' else (entry_price - exit_price) * quantity
            pnl_pct = pnl / (entry_price * quantity)

            trade = TradeRecord(
                symbol=symbol,
                side=entry_position['side'],
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                entry_time=entry_position['entry_time'],
                exit_time=exit_time,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=entry_position.get('commission', 0) + exit_order.get('commission', 0),
                duration=exit_time - entry_position['entry_time']
            )

            self.completed_trades.append(trade)
            self.total_trades += 1
            self.total_pnl += pnl
            self.total_pnl_pct += pnl_pct

            if pnl > 0:
                self.winning_trades += 1
                self.avg_win = (self.avg_win * (self.winning_trades - 1) + pnl) / self.winning_trades
            else:
                self.losing_trades += 1
                self.avg_loss = (self.avg_loss * (self.losing_trades - 1) + pnl) / self.losing_trades

        except Exception as e:
            logger.error(f"❌ Error recording completed trade: {e}")

    def _update_derived_metrics(self):
        """Update calculated metrics"""
        try:
            # Win rate
            if self.total_trades > 0:
                self.win_rate = self.winning_trades / self.total_trades

            # Profit factor
            total_wins = self.winning_trades * abs(self.avg_win) if self.winning_trades > 0 else 0
            total_losses = self.losing_trades * abs(self.avg_loss) if self.losing_trades > 0 else 0
            if total_losses > 0:
                self.profit_factor = total_wins / total_losses

            # Sharpe ratio calculation removed for simplicity
            # TODO: Implement proper portfolio returns tracking for Sharpe ratio
            self.sharpe_ratio = 0.0  # Placeholder

            # Average trade duration
            if self.completed_trades:
                total_duration = sum((t.duration for t in self.completed_trades), timedelta())
                self.avg_trade_duration = total_duration / len(self.completed_trades)

        except Exception as e:
            logger.error(f"❌ Error updating derived metrics: {e}")

    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive trading summary"""
        return {
            'performance': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'total_pnl': self.total_pnl,
                'total_pnl_pct': self.total_pnl_pct,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
                'avg_trade_duration': str(self.avg_trade_duration)
            },
            'risk': {
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': self.sharpe_ratio,
                'active_positions': len(self.active_positions)
            },
            'signals': {
                'signals_generated': self.signals_generated,
                'signals_executed': self.signals_executed,
                'signal_accuracy': self.signal_accuracy
            },
            'portfolio': {
                'current_value': self.current_portfolio_value,
                'peak_value': self.peak_portfolio_value,
                'unrealized_pnl': self._calculate_unrealized_pnl()
            }
        }

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L from active positions"""
        # This would need current market prices - simplified for now
        return 0.0

    def log_periodic_report(self):
        """Log periodic performance report"""
        if self.total_trades > 0:
            logger.info("📊 TRADING PERFORMANCE REPORT:")
            logger.info(f"   Trades: {self.total_trades} | Win Rate: {self.win_rate:.1%}")
            logger.info(f"   Total P&L: ${self.total_pnl:.2f} ({self.total_pnl_pct:.2f}%)")
            logger.info(f"   Avg Win/Loss: ${self.avg_win:.2f} / ${self.avg_loss:.2f}")
            logger.info(f"   Profit Factor: {self.profit_factor:.2f}")
            logger.info(f"   Max Drawdown: {self.max_drawdown:.1%}")
            logger.info(f"   Active Positions: {len(self.active_positions)}")

# Global metrics instance
_trading_metrics = None

def get_trading_metrics() -> TradingMetrics:
    """Get global trading metrics instance"""
    global _trading_metrics
    if _trading_metrics is None:
        _trading_metrics = TradingMetrics()
    return _trading_metrics
