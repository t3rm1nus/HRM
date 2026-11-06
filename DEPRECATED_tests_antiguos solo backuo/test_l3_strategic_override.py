#!/usr/bin/env python3
"""
Test script for L3 Strategic Override Processor
"""

from l3_strategy.decision_maker import strategic_override_processor

def test_strategic_override():
    """Test the strategic override processor with various scenarios."""

    print('🧪 Testing L3 Strategic Override Processor')
    print('=' * 50)

    # Test 1: L2 HOLD + L3 confidence < 0.80 (CRITICAL RULE)
    result1 = strategic_override_processor('HOLD', 0.62)
    print('Test 1 - L2 HOLD + L3 conf 0.62:')
    print(f'  Result: {result1}')
    print()

    # Test 2: L2 HOLD + L3 confidence >= 0.80 + all conditions met
    regime_info = {'regime': 'BULL', 'signal': 'BUY'}
    market_indicators = {'rsi': 55, 'has_pullback': True}
    result2 = strategic_override_processor('HOLD', 0.87, regime_info, market_indicators)
    print('Test 2 - L2 HOLD + L3 conf 0.87 + BULL regime + favorable conditions:')
    print(f'  Result: {result2}')
    print()

    # Test 3: L2 BUY + L3 confidence < 0.80 (should respect L2)
    result3 = strategic_override_processor('BUY', 0.65)
    print('Test 3 - L2 BUY + L3 conf 0.65:')
    print(f'  Result: {result3}')
    print()

    # Test 4: L2 BUY + TRENDING regime (should not override)
    regime_info_trending = {'regime': 'TRENDING', 'signal': 'BUY'}
    result4 = strategic_override_processor('BUY', 0.85, regime_info_trending)
    print('Test 4 - L2 BUY + L3 conf 0.85 + TRENDING regime:')
    print(f'  Result: {result4}')
    print()

    # Test 5: L2 BUY + momentum exhausted
    market_indicators_exhausted = {'rsi': 80, 'macd_divergence': True, 'volume_spike': True}
    result5 = strategic_override_processor('BUY', 0.82, {'regime': 'BULL'}, market_indicators_exhausted)
    print('Test 5 - L2 BUY + L3 conf 0.82 + momentum exhausted:')
    print(f'  Result: {result5}')
    print()

    # Test 6: L2 BUY + RSI extreme without pullback
    market_indicators_risk = {'rsi': 85, 'has_pullback': False}
    result6 = strategic_override_processor('BUY', 0.81, {'regime': 'BEAR'}, market_indicators_risk)
    print('Test 6 - L2 BUY + L3 conf 0.81 + RSI extreme without pullback:')
    print(f'  Result: {result6}')
    print()

    print('✅ All tests completed!')

if __name__ == "__main__":
    test_strategic_override()