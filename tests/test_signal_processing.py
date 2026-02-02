#!/usr/bin/env python3
"""
Test script to verify that BTCUSDT and ETHUSDT signals process without current_price errors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime

def test_signal_processing():
    """Test that both BTC and ETH signals process without UnboundLocalError."""
    print("🧪 Testing signal processing for BTCUSDT and ETHUSDT...")

    try:
        from l1_operational.order_manager import FullTradingCycleManager
        from l2_tactic.models import TacticalSignal

        # Create mock state manager and portfolio manager
        class MockPortfolioManager:
            def get_position(self, symbol):
                return 0.0  # No position initially
            def get_total_value(self):
                return 10000.0  # $10k portfolio
            def get_balance(self, asset):
                return 10000.0 if asset == 'USDT' else 0.0

        class MockStateManager:
            def test_method(self):  # Required by implementation
                return True

        # Create managers
        portfolio_manager = MockPortfolioManager()
        state_manager = MockStateManager()
        config = {
            'COOLDOWN_SECONDS': 60,
            'MIN_EXPORT_CONFIDENCE': 0.1,
            'MAX_SIGNAL_AGE': 300
        }

        manager = FullTradingCycleManager(state_manager, portfolio_manager, config)
        print("✅ FullTradingCycleManager initialized successfully")

        # Test market data (both DataFrame and dict)
        test_market_data = {
            'BTCUSDT': pd.DataFrame({
                'close': [65000.0],
                'timestamp': [datetime.now()]
            }),
            'ETHUSDT': pd.DataFrame({
                'close': [4200.0],
                'timestamp': [datetime.now()]
            })
        }

        print("✅ Test market data created")

        # Test signals that would trigger the bug
        signals = [
            TacticalSignal(
                symbol='BTCUSDT',
                side='sell',
                confidence=0.8,
                strength=0.7,
                stop_loss=64000.0,
                take_profit=68000.0,
                metadata={'test': True}
            ),
            TacticalSignal(
                symbol='ETHUSDT',
                side='sell',
                confidence=0.7,
                strength=0.6,
                stop_loss=4100.0,
                take_profit=4500.0,
                metadata={'test': True}
            )
        ]

        print("✅ Test signals created")

        results = []

        for signal in signals:
            symbol = signal.symbol
            print(f"\n🧪 Testing {symbol} signal processing...")

            try:
                market_data = test_market_data.get(symbol)
                result = manager.process_signal(signal, market_data)

                print(f"✅ {symbol} signal processed successfully")
                print(f"   Status: {result.get('status', 'unknown')}")
                print(f"   Reason: {result.get('reason', 'N/A')}")

                if result.get('status') == 'executed':
                    print(f"   ✅ SUCCESS: Signal processed without error")
                elif result.get('status') == 'rejected' and 'sell' in result.get('reason', '').lower():
                    print(f"   ✅ EXPECTED: Sell signal rejected (no position) - alternative action should trigger")
                else:
                    print(f"   ⚠️  UNEXPECTED: {result}")

                results.append((symbol, True, result))

            except Exception as e:
                print(f"❌ {symbol} signal processing FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append((symbol, False, str(e)))

        # Summary
        success_count = sum(1 for _, success, _ in results if success)
        total_count = len(results)

        print("\n" + "="*70)
        print(f"SIGNAL PROCESSING TEST RESULTS: {success_count}/{total_count} PASSED")
        print("="*70)

        for symbol, success, result in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} {symbol}: {result.get('status', str(result)) if success else result}")

        return success_count == total_count

    except Exception as e:
        print(f"❌ TEST SETUP FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alternative_action_logic():
    """Test the _handle_alternative_action method directly."""
    print("\n🧪 Testing _handle_alternative_action method directly...")

    try:
        from l1_operational.order_manager import FullTradingCycleManager
        from l2_tactic.models import TacticalSignal

        # Create mock managers
        class MockPortfolioManager:
            def get_position(self, symbol): return 0.0
            def get_total_value(self): return 10000.0
            def get_balance(self, asset): return 10000.0 if asset == 'USDT' else 0.0

        class MockStateManager:
            def test_method(self): return True

        config = {'COOLDOWN_SECONDS': 60}
        manager = FullTradingCycleManager(MockStateManager(), MockPortfolioManager(), config)

        # Test BTC sell signal with no position - should trigger alternative action
        signal = TacticalSignal(
            symbol='BTCUSDT',
            side='sell',
            confidence=0.8,
            strength=0.7,
            stop_loss=64000.0,
            take_profit=68000.0
        )

        market_data = pd.DataFrame({
            'close': [65000.0],
            'timestamp': [datetime.now()]
        })

        print("✅ Testing _handle_alternative_action directly...")

        # Call _handle_alternative_action directly - this is where the bug should occur
        result = manager._handle_alternative_action(signal, market_data, 65000.0)

        print(f"✅ _handle_alternative_action completed successfully")
        print(f"   Result: {result}")

        if result.get('status') == 'rejected':
            print("   ⚠️  Alternative action was rejected (expected if validation fails)")
            return True  # This is acceptable - the method ran without UnboundLocalError
        else:
            print("   ✅ Alternative action processed successfully")
            return True

    except Exception as e:
        print(f"❌ _handle_alternative_action test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE TEST FOR current_price UnboundLocalError FIX")
    print("=" * 70)

    tests = [
        ("signal_processing", test_signal_processing),
        ("alternative_action", test_alternative_action_logic)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50} {test_name.upper()} TEST {'='*50}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS SUMMARY")
    print("="*70)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print("="*70)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The UnboundLocalError fix is confirmed to work correctly.")
        print("✅ Both BTCUSDT and ETHUSDT signals process successfully.")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("   The current_price fix may still have issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
