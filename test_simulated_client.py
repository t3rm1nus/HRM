#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test SimulatedExchangeClient balances"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.simulated_exchange_client import SimulatedExchangeClient


def test_client_balances():
    """Test SimulatedExchangeClient initial balances"""
    print('🎯 Testing SimulatedExchangeClient balances...')
    
    # Reset client before test to ensure clean state
    SimulatedExchangeClient.force_reset({
        'BTC': 0.01,
        'ETH': 0.1,
        'USDT': 2000
    })
    
    # Create client (will use singleton instance)
    client = SimulatedExchangeClient()
    
    # Get and print balances
    balances = asyncio.run(client.get_account_balances())
    print(f'Client balances: {balances}')
    
    # Verify balances
    assert balances['BTC'] == 0.01, 'BTC balance should be 0.01'
    assert balances['ETH'] == 0.1, 'ETH balance should be 0.1'
    assert balances['USDT'] == 2000, 'USDT balance should be 2000'
    
    print('✅ SimulatedExchangeClient balances are correct')
    
    # Test get_balance method
    btc_bal = client.get_balance('BTC')
    eth_bal = client.get_balance('ETH')
    usdt_bal = client.get_balance('USDT')
    
    print(f'BTC balance: {btc_bal:.6f}')
    print(f'ETH balance: {eth_bal:.4f}')
    print(f'USDT balance: {usdt_bal:.2f}')
    
    assert btc_bal == 0.01, 'get_balance(BTC) should return 0.01'
    assert eth_bal == 0.1, 'get_balance(ETH) should return 0.1'
    assert usdt_bal == 2000, 'get_balance(USDT) should return 2000'
    
    print('✅ get_balance method is correct')


if __name__ == "__main__":
    try:
        test_client_balances()
        print("\n🎊 All SimulatedExchangeClient tests PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)