"""
Cycle PnL Metrics Module - Per-cycle performance tracking
Enhanced with aggressive mode metrics for L3 optimization.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from core.logging import logger

# Metrics storage directory
METRICS_DIR = "data/metrics"
CYCLE_METRICS_FILE = os.path.join(METRICS_DIR, "cycle_pnl_metrics.json")

# Ensure metrics directory exists
os.makedirs(METRICS_DIR, exist_ok=True)


class CycleMetrics:
    """
    Track P&L metrics per trading cycle.
    Supports aggressive mode configurations.
    """
    
    def __init__(self):
        self.cycle_count = 0
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.cycle_pnl_history = []
        self.aggressive_mode_active = True  # Flag for aggressive mode
        
        # Track cycle timing
        self.last_cycle_time = None
        self.cycle_timing_history = []
        
        # WEAK_BULL specific tracking
        self.weak_bull_cycles = 0
        self.weak_bull_signals = 0
        self.weak_bull_pnl = 0.0
        
        # ETH sync tracking
        self.eth_sync_cycles = 0
        self.eth_sync_enabled = True
        
        # Load historical data if exists
        self._load_history()
        
        logger.info("📊 CycleMetrics initialized for per-cycle P&L tracking")
    
    def _load_history(self):
        """Load historical metrics from file"""
        if os.path.exists(CYCLE_METRICS_FILE):
            try:
                with open(CYCLE_METRICS_FILE, 'r') as f:
                    data = json.load(f)
                    self.cycle_count = data.get('cycle_count', 0)
                    self.total_pnl = data.get('total_pnl', 0.0)
                    self.total_pnl_pct = data.get('total_pnl_pct', 0.0)
                    self.cycle_pnl_history = data.get('cycle_pnl_history', [])
                logger.info(f"📊 Loaded {self.cycle_count} historical cycles")
            except Exception as e:
                logger.warning(f"⚠️ Could not load cycle metrics history: {e}")
    
    def _save_history(self):
        """Save current metrics to file"""
        try:
            data = {
                'cycle_count': self.cycle_count,
                'total_pnl': self.total_pnl,
                'total_pnl_pct': self.total_pnl_pct,
                'cycle_pnl_history': self.cycle_pnl_history[-100:],  # Keep last 100 cycles
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(CYCLE_METRICS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Could not save cycle metrics: {e}")
    
    def record_cycle(self, cycle_id: int, pnl: float, pnl_pct: float, 
                    regime: str = None, subtype: str = None,
                    signals_generated: int = 0, orders_executed: int = 0,
                    portfolio_value: float = 0.0, market_data: Dict = None) -> Dict[str, Any]:
        """
        Record metrics for a single trading cycle.
        
        Args:
            cycle_id: Current cycle number
            pnl: P&L for this cycle (USD)
            pnl_pct: P&L percentage for this cycle
            regime: Market regime detected
            subtype: Regime subtype (e.g., WEAK_BULL)
            signals_generated: Number of signals generated
            orders_executed: Number of orders executed
            portfolio_value: Current portfolio value
            
        Returns:
            Dict with current metrics summary
        """
        self.cycle_count = cycle_id
        
        # Update totals
        self.total_pnl += pnl
        self.total_pnl_pct += pnl_pct
        
        # Record cycle data
        cycle_data = {
            'cycle_id': cycle_id,
            'timestamp': datetime.utcnow().isoformat(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'cumulative_pnl': self.total_pnl,
            'cumulative_pnl_pct': self.total_pnl_pct,
            'regime': regime,
            'subtype': subtype,
            'signals_generated': signals_generated,
            'orders_executed': orders_executed,
            'portfolio_value': portfolio_value,
            'aggressive_mode': self.aggressive_mode_active,
            'weak_bull_detected': subtype == 'WEAK_BULL' if subtype else False,
            'eth_sync_active': self.eth_sync_enabled
        }
        
        self.cycle_pnl_history.append(cycle_data)
        
        # Track WEAK_BULL specific metrics
        if subtype == 'WEAK_BULL':
            self.weak_bull_cycles += 1
            self.weak_bull_signals += signals_generated
            self.weak_bull_pnl += pnl
            cycle_data['weak_bull_stats'] = {
                'cycles': self.weak_bull_cycles,
                'signals': self.weak_bull_signals,
                'pnl': self.weak_bull_pnl
            }
        
        # Track ETH sync cycles
        if self.eth_sync_enabled:
            self.eth_sync_cycles += 1
            cycle_data['eth_sync_cycles'] = self.eth_sync_cycles
        
        # Keep only last 100 cycles in memory
        if len(self.cycle_pnl_history) > 100:
            self.cycle_pnl_history = self.cycle_pnl_history[-100:]
        
        # Save to file periodically (every 10 cycles)
        if cycle_id % 10 == 0:
            self._save_history()
        
        return cycle_data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        recent_cycles = self.cycle_pnl_history[-10:] if self.cycle_pnl_history else []
        recent_pnl = sum(c['pnl'] for c in recent_cycles)
        
        # Calculate win rate from recent cycles
        winning_cycles = sum(1 for c in recent_cycles if c['pnl'] > 0)
        total_cycles = len(recent_cycles)
        recent_win_rate = winning_cycles / total_cycles if total_cycles > 0 else 0.0
        
        return {
            'cycle_count': self.cycle_count,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'recent_pnl_10': recent_pnl,
            'recent_win_rate': recent_win_rate,
            'aggressive_mode': self.aggressive_mode_active,
            'weak_bull_stats': {
                'cycles': self.weak_bull_cycles,
                'signals': self.weak_bull_signals,
                'pnl': self.weak_bull_pnl
            },
            'eth_sync_stats': {
                'total_cycles': self.eth_sync_cycles,
                'enabled': self.eth_sync_enabled
            },
            'last_cycle': self.cycle_pnl_history[-1] if self.cycle_pnl_history else None
        }
    
    def log_cycle_report(self, cycle_data: Dict[str, Any]):
        """Log detailed cycle report with aggressive mode metrics"""
        cycle_id = cycle_data['cycle_id']
        pnl = cycle_data['pnl']
        pnl_pct = cycle_data['pnl_pct']
        regime = cycle_data.get('regime', 'unknown')
        subtype = cycle_data.get('subtype', None)
        
        # Color based on P&L
        if pnl > 0:
            pnl_str = f"🟢 +${pnl:.2f} (+{pnl_pct:.2f}%)"
        elif pnl < 0:
            pnl_str = f"🔴 ${pnl:.2f} ({pnl_pct:.2f}%)"
        else:
            pnl_str = f"⚪ ${pnl:.2f} (0.00%)"
        
        logger.info(f"📊 CYCLE {cycle_id} PnL REPORT:")
        logger.info(f"   Regime: {regime}" + (f" ({subtype})" if subtype else ""))
        logger.info(f"   P&L: {pnl_str}")
        logger.info(f"   Cumulative: ${self.total_pnl:.2f} ({self.total_pnl_pct:.2f}%)")
        logger.info(f"   Signals: {cycle_data.get('signals_generated', 0)} | Orders: {cycle_data.get('orders_executed', 0)}")
        
        # Aggressive mode specific logging
        if self.aggressive_mode_active:
            if subtype == 'WEAK_BULL':
                logger.info(f"   🐂 WEAK_BULL CYCLE #{self.weak_bull_cycles} - Aggressive BUY active")
            if self.eth_sync_enabled:
                logger.info(f"   🔄 ETH Sync: Active (Cycle #{self.eth_sync_cycles})")
        
        # Show cumulative stats every 5 cycles
        if cycle_id % 5 == 0:
            summary = self.get_summary()
            logger.info("="*60)
            logger.info(f"📈 CUMULATIVE STATS (Cycles 1-{cycle_id}):")
            logger.info(f"   Total P&L: ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']:.2f}%)")
            logger.info(f"   Recent 10 Cycles P&L: ${summary['recent_pnl_10']:.2f}")
            logger.info(f"   Recent Win Rate: {summary['recent_win_rate']:.1%}")
            logger.info("="*60)
    
    def reset(self):
        """Reset all metrics"""
        self.cycle_count = 0
        self.total_pnl = 0.0
        self.total_pnl_pct = 0.0
        self.cycle_pnl_history = []
        self.weak_bull_cycles = 0
        self.weak_bull_signals = 0
        self.weak_bull_pnl = 0.0
        self.eth_sync_cycles = 0
        
        # Delete history file
        if os.path.exists(CYCLE_METRICS_FILE):
            os.remove(CYCLE_METRICS_FILE)
        
        logger.info("📊 CycleMetrics reset complete")


# Global instance
_cycle_metrics = None

def get_cycle_metrics() -> CycleMetrics:
    """Get global cycle metrics instance"""
    global _cycle_metrics
    if _cycle_metrics is None:
        _cycle_metrics = CycleMetrics()
    return _cycle_metrics


def record_aggressive_mode_cycle(
    cycle_id: int,
    pnl: float,
    regime: str,
    subtype: str = None,
    signals: int = 0,
    orders: int = 0,
    portfolio_value: float = 0.0
) -> Dict[str, Any]:
    """
    Convenience function to record a cycle with aggressive mode settings.
    """
    metrics = get_cycle_metrics()
    
    # Calculate PnL percentage
    pnl_pct = (pnl / portfolio_value * 100) if portfolio_value > 0 else 0.0
    
    cycle_data = metrics.record_cycle(
        cycle_id=cycle_id,
        pnl=pnl,
        pnl_pct=pnl_pct,
        regime=regime,
        subtype=subtype,
        signals_generated=signals,
        orders_executed=orders,
        portfolio_value=portfolio_value
    )
    
    # Log the report
    metrics.log_cycle_report(cycle_data)
    
    return cycle_data
