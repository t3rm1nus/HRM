#!/usr/bin/env python3
"""
Test script for Paper Trade Logger
"""

from storage.paper_trade_logger import get_paper_logger
from datetime import datetime

def test_paper_trade_logger():
    """Test the paper trade logging functionality"""
    print("🧪 Testing Paper Trade Logger...")

    # Get logger instance with clearing (simulates main.py behavior)
    logger = get_paper_logger(clear_on_init=True)

    # Create some test orders
    test_orders = [
        {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 0.001,
            'filled_price': 50000.0,
            'commission': 0.05,
            'status': 'filled',
            'order_id': 'test_1'
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'quantity': 0.01,
            'filled_price': 3000.0,
            'commission': 0.03,
            'status': 'filled',
            'order_id': 'test_2'
        },
        {
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'quantity': 0.001,
            'filled_price': 51000.0,
            'commission': 0.051,
            'status': 'filled',
            'order_id': 'test_3'
        }
    ]

    # Market data for context
    market_data = {
        'BTCUSDT': {'close': 50000.0},
        'ETHUSDT': {'close': 3000.0}
    }

    # Log trades
    for i, order in enumerate(test_orders):
        logger.log_paper_trade(
            order=order,
            market_data=market_data,
            cycle_id=100 + i,
            strategy="test_strategy"
        )
        print(f"✅ Logged trade {i+1}: {order['symbol']} {order['side']} {order['quantity']}")

    # Get session summary
    summary = logger.get_session_summary()
    print("\n📊 Session Summary:")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Total Fees: ${summary['total_fees']:.2f}")
    print(f"   Win Rate: {summary['win_rate']:.1f}%")

    # Get recent trades
    recent = logger.get_recent_trades(5)
    print(f"\n📝 Recent Trades: {len(recent)}")

    # Export for analysis
    export_file = logger.export_for_analysis()
    print(f"\n📊 Exported trades to: {export_file}")

    # Show stats report
    print("\n📈 Full Report:")
    print(logger.get_stats_report())

    print("\n✅ Paper Trade Logger test completed!")

if __name__ == "__main__":
    test_paper_trade_logger()
