"""
Test for portfolio rebalancer
"""

import pytest
import numpy as np
import pandas as pd
from core.portfolio_rebalancer import PortfolioRebalancer, RebalanceTrigger


class MockWeightCalculator:
    """Mock weight calculator for testing"""
    def __init__(self, target_weights):
        self.target_weights = target_weights
    
    def get_portfolio_risk_metrics(self, current_weights):
        """Mock risk metrics"""
        return {'volatility': 0.15}
    
    def get_correlation_report(self, current_weights):
        """Mock correlation report"""
        return {'correlation_risk_metrics': {'average_correlation': 0.5}}


def test_portfolio_rebalancer_initialization():
    """Test portfolio rebalancer initialization"""
    weight_calc = MockWeightCalculator({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    rebalancer = PortfolioRebalancer(weight_calc, partial_rebalance_factor=0.5)
    
    assert rebalancer.rebalance_enabled == True
    assert rebalancer.partial_rebalance_factor == 0.5
    assert rebalancer.drift_threshold == 0.10


def test_portfolio_rebalancer_set_target_weights():
    """Test setting target weights"""
    weight_calc = MockWeightCalculator({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    rebalancer = PortfolioRebalancer(weight_calc)
    
    target_weights = {'BTC': 0.5, 'ETH': 0.25, 'USDT': 0.25}
    rebalancer.set_target_weights(target_weights)
    
    assert rebalancer.target_weights == target_weights


def test_portfolio_rebalancer_should_rebalance_threshold():
    """Test threshold-based rebalance trigger"""
    weight_calc = MockWeightCalculator({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    rebalancer = PortfolioRebalancer(weight_calc)
    
    rebalancer.set_target_weights({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    
    # Current weights within threshold
    current_weights = {'BTC': 0.42, 'ETH': 0.28, 'USDT': 0.30}
    should_rebalance, reason = rebalancer.should_rebalance(current_weights, RebalanceTrigger.THRESHOLD_BASED)
    assert should_rebalance == False
    
    # Current weights exceed threshold
    current_weights = {'BTC': 0.55, 'ETH': 0.20, 'USDT': 0.25}
    should_rebalance, reason = rebalancer.should_rebalance(current_weights, RebalanceTrigger.THRESHOLD_BASED)
    assert should_rebalance == True


@pytest.mark.asyncio
async def test_portfolio_rebalancer_execute_partial_rebalance():
    """Test executing partial rebalance"""
    weight_calc = MockWeightCalculator({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    rebalancer = PortfolioRebalancer(weight_calc, partial_rebalance_factor=0.5)
    
    rebalancer.set_target_weights({'BTC': 0.4, 'ETH': 0.3, 'USDT': 0.3})
    
    # Mock current portfolio and market data
    current_weights = {'BTC': 0.5, 'ETH': 0.25, 'USDT': 0.25}
    portfolio_value = 10000  # $10,000 portfolio
    
    market_data = {
        'BTC': pd.DataFrame({'close': [50000]}),
        'ETH': pd.DataFrame({'close': [3000]}),
        'USDT': 1.0  # USDT price is always 1
    }
    
    result = await rebalancer.execute_rebalance(current_weights, portfolio_value, market_data, 
                                               RebalanceTrigger.THRESHOLD_BASED, partial=True)
    
    assert result.success == True
    assert len(result.trades_required) > 0
    assert result.metadata['partial'] == True
    assert result.metadata['partial_factor'] == 0.5
    
    # Verify total trade value is less than full rebalance
    total_trade_value = sum(trade.estimated_value for trade in result.trades_required)
    assert total_trade_value < 10000 * 0.1  # Less than 10% of portfolio


if __name__ == '__main__':
    pytest.main([__file__, '-v'])