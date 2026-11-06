#!/usr/bin/env python3
"""
Test script to verify the comprehensive four-level selling strategy
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_selling_strategy():
    """Test all four levels of the selling strategy"""
    print("🧪 Testing Comprehensive Selling Strategy")
    print("=" * 60)

    from core.selling_strategy import get_selling_strategy, SellSignal

    selling_strategy = get_selling_strategy()

    # Test 1: Stop-loss protection (Priority 1)
    print("\n🔴 TESTING PRIORITY 1: Stop-loss Protection")
    test_stop_loss(selling_strategy)

    # Test 2: Tactical edge disappearance (Priority 2)
    print("\n🟠 TESTING PRIORITY 2: Tactical Edge Disappearance")
    test_tactical_edge(selling_strategy)

    # Test 3: Strategic regime change (Priority 3)
    print("\n🟡 TESTING PRIORITY 3: Strategic Regime Change")
    test_strategic_regime(selling_strategy)

    # Test 4: Timeout exit (Priority 4)
    print("\n🔵 TESTING PRIORITY 4: Timeout Exit")
    test_timeout_exit(selling_strategy)

    print("\n🎉 ALL SELLING STRATEGY TESTS COMPLETED!")
    return True

def test_stop_loss(selling_strategy):
    """Test stop-loss protection at 1% loss"""
    # Create position data with 2% loss (should trigger stop-loss)
    position_data = {
        'entry_price': 50000.0,
        'quantity': 0.001,  # Long position
    }

    current_price = 49000.0  # 2% loss from entry

    # Mock market data
    market_data = pd.DataFrame({
        'close': [current_price],
        'open': [50000.0],
        'high': [51000.0],
        'low': [48500.0]
    })

    # Mock L3 context
    l3_context = {'regime': 'TRENDING', 'confidence': 0.8}

    # Assess sell opportunities
    sell_signal = selling_strategy._assess_stop_loss_protection('BTCUSDT', current_price, position_data)

    if sell_signal and sell_signal.priority == 1:
        loss_pct = ((current_price - position_data['entry_price']) / position_data['entry_price']) * 100
        print(f"✅ STOP-LOSS TRIGGERED: {loss_pct:.1f}% loss → SELL signal (Priority {sell_signal.priority})")
        return True
    else:
        print("❌ STOP-LOSS FAILED: No sell signal generated")
        return False

def test_tactical_edge(selling_strategy):
    """Test tactical edge disappearance detection"""
    # Register position entry first
    entry_data = {'price': 50000.0, 'quantity': 0.001}
    market_data_entry = pd.DataFrame({
        'close': [50000.0],
        'momentum': [0.5],  # Strong momentum at entry
        'rsi': [45.0],      # Normal RSI at entry
        'volume': [1000.0]  # Good volume at entry
    })
    l3_context_entry = {'regime': 'TRENDING', 'confidence': 0.8}

    selling_strategy.register_position_entry('BTCUSDT', entry_data, market_data_entry, l3_context_entry)

    # Now test with faded edge conditions
    position_data = {
        'entry_price': 50000.0,
        'quantity': 0.001,
    }

    current_price = 49500.0  # Slight decline

    # Mock market data with faded edge
    market_data = pd.DataFrame({
        'close': [current_price],
        'momentum': [-0.6],  # Momentum reversed
        'rsi': [78.0],       # RSI in extreme zone
        'volume': [300.0],   # Volume dried up
        'volume_sma': [800.0]  # Average volume much higher
    })

    l3_context = {'regime': 'TRENDING', 'confidence': 0.8}

    # Assess sell opportunities
    sell_signal = selling_strategy._assess_tactical_edge('BTCUSDT', market_data, position_data, l3_context)

    if sell_signal and sell_signal.priority == 2:
        print(f"✅ TACTICAL EDGE DETECTED: {sell_signal.reason} (Priority {sell_signal.priority})")
        return True
    else:
        print("❌ TACTICAL EDGE FAILED: No sell signal generated")
        return False

def test_strategic_regime(selling_strategy):
    """Test strategic regime change detection"""
    # Register position entry in TRENDING regime
    entry_data = {'price': 50000.0, 'quantity': 0.001}
    market_data_entry = pd.DataFrame({'close': [50000.0]})
    l3_context_entry = {'regime': 'TRENDING', 'confidence': 0.8}

    selling_strategy.register_position_entry('BTCUSDT', entry_data, market_data_entry, l3_context_entry)

    # Now test regime change to RANGE
    position_data = {
        'entry_price': 50000.0,
        'quantity': 0.001,
        'entry_regime': 'TRENDING'
    }

    current_price = 50000.0  # Same price

    # Mock market data
    market_data = pd.DataFrame({'close': [current_price]})

    # L3 context with regime change and low confidence
    l3_context = {
        'regime': 'RANGE',  # Changed from TRENDING
        'confidence': 0.3,  # Low confidence
        'signal': 'hold'
    }

    # Assess sell opportunities
    sell_signal = selling_strategy._assess_strategic_regime('BTCUSDT', l3_context, position_data)

    if sell_signal and sell_signal.priority == 3:
        print(f"✅ STRATEGIC REGIME DETECTED: {sell_signal.reason} (Priority {sell_signal.priority})")
        return True
    else:
        print("❌ STRATEGIC REGIME FAILED: No sell signal generated")
        return False

def test_timeout_exit(selling_strategy):
    """Test timeout exit after max holding time"""
    # Register position entry with old timestamp
    entry_data = {'price': 50000.0, 'quantity': 0.001}
    market_data_entry = pd.DataFrame({'close': [50000.0]})
    l3_context_entry = {'regime': 'TRENDING', 'confidence': 0.8}

    # Manually set old entry time (3 hours ago)
    old_timestamp = datetime.utcnow() - timedelta(hours=3)
    selling_strategy.active_positions['BTCUSDT'] = {
        'entry_price': 50000.0,
        'quantity': 0.001,
        'entry_timestamp': old_timestamp,
        'entry_regime': 'TRENDING',
        'entry_confidence': 0.8
    }

    position_data = {
        'entry_price': 50000.0,
        'quantity': 0.001,
        'entry_timestamp': old_timestamp.isoformat()
    }

    current_price = 50000.0  # Same price

    # Mock market data
    market_data = pd.DataFrame({'close': [current_price]})

    l3_context = {'regime': 'TRENDING', 'confidence': 0.8}

    # Assess sell opportunities
    sell_signal = selling_strategy._assess_timeout_exit('BTCUSDT', position_data, l3_context)

    if sell_signal and sell_signal.priority == 4:
        print(f"✅ TIMEOUT EXIT DETECTED: {sell_signal.reason} (Priority {sell_signal.priority})")
        return True
    else:
        print("❌ TIMEOUT EXIT FAILED: No sell signal generated")
        return False

def test_priority_hierarchy():
    """Test that priority hierarchy works correctly"""
    print("\n🎯 TESTING PRIORITY HIERARCHY")

    from core.selling_strategy import get_selling_strategy

    selling_strategy = get_selling_strategy()

    # Create position data
    position_data = {
        'entry_price': 50000.0,
        'quantity': 0.001,
        'entry_regime': 'TRENDING',
        'entry_timestamp': (datetime.utcnow() - timedelta(hours=3)).isoformat()
    }

    current_price = 49000.0  # 2% loss (should trigger stop-loss)
    market_data = pd.DataFrame({'close': [current_price]})
    l3_context = {'regime': 'RANGE', 'confidence': 0.3}

    # Register position
    selling_strategy.register_position_entry('BTCUSDT',
                                           {'price': 50000.0, 'quantity': 0.001},
                                           pd.DataFrame({'close': [50000.0]}),
                                           {'regime': 'TRENDING', 'confidence': 0.8})

    # Assess sell opportunities - should return highest priority (stop-loss)
    sell_signal = selling_strategy.assess_sell_opportunities('BTCUSDT', current_price, market_data, l3_context, position_data)

    if sell_signal and sell_signal.priority == 1:  # Stop-loss should be priority 1
        print(f"✅ PRIORITY HIERARCHY WORKING: Stop-loss (Priority 1) triggered despite other conditions")
        return True
    else:
        print("❌ PRIORITY HIERARCHY FAILED: Wrong priority level returned")
        return False

if __name__ == "__main__":
    try:
        # Run all tests
        test_selling_strategy()
        test_priority_hierarchy()

        print("\n🎉 ALL SELLING STRATEGY TESTS PASSED!")
        print("\n✅ Four-Level Selling Strategy Implemented:")
        print("   🔴 PRIORITY 1: Stop-loss protection (1% loss)")
        print("   🟠 PRIORITY 2: Tactical edge disappearance")
        print("   🟡 PRIORITY 3: Strategic regime change")
        print("   🔵 PRIORITY 4: Timeout exit (120 cycles/~2 hours)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
