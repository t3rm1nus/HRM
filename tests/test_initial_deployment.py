#!/usr/bin/env python3
"""
Test script for the initial deployment functionality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.position_rotator import generate_initial_deployment

def test_initial_deployment():
    """Test the generate_initial_deployment function"""
    print("Testing initial deployment function...")

    # Test with default parameters
    orders = generate_initial_deployment(3000.0, 0.4, 0.3)

    print(f"\nGenerated {len(orders)} orders:")

    total_deployed = 0
    for order in orders:
        symbol = order["symbol"]
        side = order["side"]
        quantity = order["quantity"]
        price = order["price"]
        allocation_pct = order.get("allocation_pct", 0)

        value = quantity * price
        total_deployed += value

        print(f"  {symbol} {side} {quantity:.4f} @ ${price:.2f} = ${value:.2f} ({allocation_pct*100:.1f}%)")

    expected_deployed = 3000.0 * (0.4 + 0.3)  # 70% of 3000 = 2100
    usdt_reserve = 3000.0 - total_deployed

    print(f"\nSummary:")
    print(f"  Total capital: $3000.00")
    print(f"  Total deployed: ${total_deployed:.2f}")
    print(f"  USDT reserve: ${usdt_reserve:.2f} ({usdt_reserve/3000.0*100:.1f}%)")
    print(f"  Orders generated: {len(orders)}")

    # Validate results
    assert len(orders) == 2, f"Expected 2 orders, got {len(orders)}"
    assert abs(total_deployed - 2100.0) < 1.0, f"Expected $2100 deployed, got ${total_deployed}"
    assert abs(usdt_reserve - 900.0) < 1.0, f"Expected $900 USDT reserve, got ${usdt_reserve}"

    # Check order structure
    for order in orders:
        required_fields = ["symbol", "side", "type", "quantity", "price", "status", "order_type"]
        for field in required_fields:
            assert field in order, f"Missing required field '{field}' in order"

        assert order["side"] == "buy", f"Expected 'buy' side, got '{order['side']}'"
        assert order["type"] == "MARKET", f"Expected 'MARKET' type, got '{order['type']}'"
        assert order["status"] == "pending", f"Expected 'pending' status, got '{order['status']}'"
        assert order["order_type"] == "ENTRY", f"Expected 'ENTRY' order_type, got '{order['order_type']}'"

    print("\n✅ All tests passed! Initial deployment function works correctly.")

if __name__ == "__main__":
    test_initial_deployment()
