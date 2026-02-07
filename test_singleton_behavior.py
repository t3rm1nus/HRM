#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test singleton behavior of SimulatedExchangeClient"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l1_operational.simulated_exchange_client import SimulatedExchangeClient
from core.simulated_exchange_client import SimulatedExchangeClient as CoreSimulatedExchangeClient


async def test_l1_singleton():
    """Test l1_operational.SimulatedExchangeClient singleton"""
    print("🧪 Testing l1_operational.SimulatedExchangeClient singleton...")
    
    # Force reset to ensure clean state
    SimulatedExchangeClient.force_reset({
        'BTC': 0.01,
        'ETH': 0.1,
        'USDT': 2000
    })
    
    # Create first instance
    client1 = SimulatedExchangeClient()
    balances1 = await client1.get_account_balances()
    print(f"Client 1 balances: {balances1}")
    
    # Create second instance
    client2 = SimulatedExchangeClient()
    balances2 = await client2.get_account_balances()
    print(f"Client 2 balances: {balances2}")
    
    # Verify same instance
    assert id(client1) == id(client2), "Should be same instance"
    assert balances1 == balances2, "Balances should be identical"
    
    # Execute an order on first client
    client1.execute_order('BTCUSDT', 'BUY', 0.001, 50000)
    balances1_after = await client1.get_account_balances()
    print(f"Client 1 balances after order: {balances1_after}")
    
    # Check if second client has same balances
    balances2_after = await client2.get_account_balances()
    print(f"Client 2 balances after order: {balances2_after}")
    
    assert balances1_after == balances2_after, "Balances should be identical after order"
    
    print("✅ l1_operational.SimulatedExchangeClient singleton test PASSED")


async def test_core_singleton():
    """Test core.SimulatedExchangeClient singleton"""
    print("\n🧪 Testing core.SimulatedExchangeClient singleton...")
    
    # Force reset to ensure clean state
    CoreSimulatedExchangeClient.force_reset({
        'BTC': 0.01,
        'ETH': 0.1,
        'USDT': 2000
    })
    
    # Create first instance
    client1 = CoreSimulatedExchangeClient()
    balances1 = await client1.get_account_balances()
    print(f"Client 1 balances: {balances1}")
    
    # Create second instance
    client2 = CoreSimulatedExchangeClient()
    balances2 = await client2.get_account_balances()
    print(f"Client 2 balances: {balances2}")
    
    # Verify same instance
    assert id(client1) == id(client2), "Should be same instance"
    assert balances1 == balances2, "Balances should be identical"
    
    # Execute an order on first client
    await client1.create_order('BTCUSDT', 'BUY', 0.001, order_type="market")
    balances1_after = await client1.get_account_balances()
    print(f"Client 1 balances after order: {balances1_after}")
    
    # Check if second client has same balances
    balances2_after = await client2.get_account_balances()
    print(f"Client 2 balances after order: {balances2_after}")
    
    assert balances1_after == balances2_after, "Balances should be identical after order"
    
    print("✅ core.SimulatedExchangeClient singleton test PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(test_l1_singleton())
        asyncio.run(test_core_singleton())
        print("\n🎊 All singleton tests PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)