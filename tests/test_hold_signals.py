#!/usr/bin/env python3
"""
Test script to verify HOLD signals are generated correctly
"""

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
import pandas as pd

def test_hold_signals():
    """Test that HOLD signals are generated as default"""

    # Create processor with proper config
    config = L2Config()
    processor = L2TacticProcessor(config)

    # Test L3 context that allows L2 autonomy (should generate HOLD by default)
    l3_context = {
        'signal': 'hold',
        'confidence': 0.5,
        'regime': 'RANGE',
        'allow_l2': True
    }

    # Mock market data
    market_data = {
        'BTCUSDT': {'historical_data': pd.DataFrame({'close': [100, 101, 102, 103, 104]})},
        'ETHUSDT': {'historical_data': pd.DataFrame({'close': [200, 201, 202, 203, 204]})}
    }

    # Generate signals
    signals = processor.generate_signals(market_data, l3_context)

    print(f"✅ Generated {len(signals)} signals:")
    hold_count = 0

    for sig in signals:
        print(f"  {sig.symbol}: {sig.side} (conf={sig.confidence:.3f}) - {sig.source}")
        if sig.side == 'hold':
            hold_count += 1
        if hasattr(sig, 'metadata') and sig.metadata:
            print(f"    Reason: {sig.metadata.get('reason', 'N/A')}")

    print(f"\n📊 Results:")
    print(f"  Total signals: {len(signals)}")
    print(f"  HOLD signals: {hold_count}")
    print(f"  HOLD percentage: {(hold_count/len(signals)*100):.1f}%" if signals else "0%")

    if hold_count > 0:
        print("✅ SUCCESS: HOLD signals are being generated!")
        return True
    else:
        print("❌ FAILURE: No HOLD signals generated")
        return False

if __name__ == "__main__":
    success = test_hold_signals()
    exit(0 if success else 1)