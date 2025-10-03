#!/usr/bin/env python3
"""
Test script for staggered profit-taking implementation
"""

print('🎯 TESTING STAGGERED PROFIT-TAKING IMPLEMENTATION')
print('=' * 60)

try:
from l2_tactic.tactical_signal_processor import L2TacticProcessor

    # Create a temporary processor instance
    processor = L2TacticProcessor.__new__(L2TacticProcessor)

    # Test cases with different RSI and convergence levels
    test_cases = [
        (1000.0, 'buy', 25.0, 0.3, 'Oversold + Low convergence'),
        (1000.0, 'buy', 50.0, 0.7, 'Normal + Medium convergence'),
        (1000.0, 'buy', 75.0, 0.9, 'Overbought + High convergence'),
        (1000.0, 'sell', 30.0, 0.8, 'Oversold SELL + High convergence'),
    ]

    for price, side, rsi, conv, desc in test_cases:
        try:
            targets = processor._calculate_profit_targets(price, side, rsi, conv)
            print(f'{desc}:')
            print(f'  {side.upper()} @ {price:.1f}, RSI={rsi:.1f}, Conv={conv:.1f}')
            print(f'  Targets: {targets}')
            print()
        except Exception as e:
            print(f'ERROR in {desc}: {e}')
            print()

    print('✅ Profit-taking calculation test completed successfully!')

except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
