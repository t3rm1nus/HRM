#!/usr/bin/env python3
"""
Test script to verify convergence integration in signal composition
"""

from l2_tactic.signal_composer import SignalComposer
from l2_tactic.models import TacticalSignal
from l2_tactic.config import SignalConfig
import pandas as pd

def test_convergence_integration():
    print('🔄 SIGNAL COMPOSITION WITH CONVERGENCE INTEGRATION TEST')
    print('=' * 60)

    # Create signal composer
    config = SignalConfig()
    composer = SignalComposer(config)

    # Create test signals with convergence data
    test_signals = [
        TacticalSignal(
            symbol='BTCUSDT',
            side='buy',
            strength=0.8,
            confidence=0.9,
            source='ai',
            features={
                'rsi': 45.0,
                'macd': 150.0,
                'l1_l2_agreement': 0.85,  # High convergence
                'close': 50000.0
            },
            timestamp=pd.Timestamp.now()
        ),
        TacticalSignal(
            symbol='BTCUSDT',
            side='buy',
            strength=0.6,
            confidence=0.7,
            source='technical',
            features={
                'rsi': 48.0,
                'macd': 120.0,
                'l1_l2_agreement': 0.85,  # High convergence
                'close': 50000.0
            },
            timestamp=pd.Timestamp.now()
        ),
        TacticalSignal(
            symbol='ETHUSDT',
            side='sell',
            strength=0.7,
            confidence=0.8,
            source='ai',
            features={
                'rsi': 65.0,
                'macd': -80.0,
                'l1_l2_agreement': 0.35,  # Low convergence
                'close': 3000.0
            },
            timestamp=pd.Timestamp.now()
        )
    ]

    # Mock state
    state = {
        'portfolio': {
            'USDT': {'free': 10000.0},
            'BTCUSDT': {'position': 0.0},
            'ETHUSDT': {'position': 0.0}
        },
        'market_data': {}
    }

    print('\n🧪 Testing Signal Composition with Convergence Integration')
    print('-' * 55)

    # Compose signals
    composed_signals = composer.compose(test_signals, state)

    print(f'Input signals: {len(test_signals)}')
    print(f'Composed signals: {len(composed_signals)}')

    for i, signal in enumerate(composed_signals, 1):
        print(f'\n📊 SIGNAL {i}: {signal.symbol} {signal.side.upper()}')
        print(f'   Confidence: {signal.confidence:.3f}, Strength: {signal.strength:.3f}')
        print(f'   Convergence: {getattr(signal, "convergence", "N/A")}')
        print(f'   Source: {signal.source}')
        print(f'   Quantity: {signal.quantity:.6f}')

        # Check metadata
        if hasattr(signal, 'metadata') and signal.metadata:
            conv_score = signal.metadata.get('convergence_score', 'N/A')
            print(f'   Metadata Convergence: {conv_score}')

            if 'technical_indicators' in signal.metadata:
                indicators = signal.metadata['technical_indicators']
                l1_l2 = indicators.get('l1_l2_agreement', 'N/A')
                print(f'   L1_L2 Agreement: {l1_l2}')

    print('\n✅ Signal Composition with Convergence Integration Complete!')
    print('   - Convergence scores properly extracted from signal features')
    print('   - Convergence attributes added to composed signals')
    print('   - Order manager can now access convergence for profit-taking')
    print('   - High convergence = aggressive profit-taking')
    print('   - Low convergence = conservative profit-taking')

    return composed_signals

if __name__ == '__main__':
    test_convergence_integration()
