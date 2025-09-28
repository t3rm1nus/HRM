#!/usr/bin/env python3
"""
Test script to verify profitability fixes
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from l1_operational.order_manager import OrderManager
from l2_tactic.models import TacticalSignal
from core.portfolio_manager import PortfolioManager
import pandas as pd

async def test_order_sizing():
    """Test that order sizing works with new parameters"""
    print("🧪 Testing Order Sizing Fixes...")

    # Create test portfolio with $3000 balance
    portfolio_manager = PortfolioManager(
        mode="simulated",
        initial_balance=3000.0,
        symbols=["BTCUSDT", "ETHUSDT"]
    )

    # Create order manager
    order_manager = OrderManager()

    # Create test market data
    market_data = {
        "BTCUSDT": pd.DataFrame({
            "close": [109494.74] * 200,
            "open": [109400.0] * 200,
            "high": [109600.0] * 200,
            "low": [109300.0] * 200,
            "volume": [100.0] * 200
        }),
        "ETHUSDT": pd.DataFrame({
            "close": [4016.81] * 200,
            "open": [4000.0] * 200,
            "high": [4050.0] * 200,
            "low": [3980.0] * 200,
            "volume": [1000.0] * 200
        })
    }

    # Create test signals
    signals = [
        TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.8,
            confidence=0.75,
            signal_type="test",
            source="test"
        ),
        TacticalSignal(
            symbol="ETHUSDT",
            side="buy",
            strength=0.7,
            confidence=0.70,
            signal_type="test",
            source="test"
        )
    ]

    # Create test state
    state = {
        "portfolio": portfolio_manager.get_portfolio_state(),
        "market_data": market_data,
        "l3_output": {
            "volatility_forecast": {"BTCUSDT": 0.02, "ETHUSDT": 0.025},
            "risk_appetite": 0.6
        }
    }

    # Generate orders
    orders = await order_manager.generate_orders(state, signals)

    print(f"📊 Generated {len(orders)} orders")

    for order in orders:
        if order.get("type") == "MARKET":
            symbol = order["symbol"]
            quantity = order["quantity"]
            price = order["price"]
            value = abs(quantity) * price
            print(f"✅ {symbol} {order['side']} {quantity:.4f} (${value:.2f})")

    # Execute orders
    executed_orders = await order_manager.execute_orders(orders)
    print(f"📊 Executed {len([o for o in executed_orders if o.get('status') == 'filled'])} orders successfully")

    # Update portfolio
    await portfolio_manager.update_from_orders_async(executed_orders, market_data)

    final_value = portfolio_manager.get_total_value()
    print(f"💰 Final portfolio value: ${final_value:.2f}")
    return final_value > 3000.0

if __name__ == "__main__":
    success = asyncio.run(test_order_sizing())
    if success:
        print("✅ Profitability fixes working correctly!")
    else:
        print("❌ Issues detected with profitability fixes")
