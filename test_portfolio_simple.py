#!/usr/bin/env python3
"""
Simple test script to verify portfolio persistence without ML dependencies
"""
import os
import sys
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the problematic imports to avoid TensorFlow issues
sys.modules['l2_tactic.utils'] = type(sys)('mock_utils')
sys.modules['l2_tactic.utils'].safe_float = lambda x: float(x) if x is not None else 0.0

class MockPortfolioManager:
    """Simplified PortfolioManager for testing without ML dependencies"""

    def __init__(self, mode="live", initial_balance=3000.0, symbols=None):
        self.mode = mode
        self.initial_balance = initial_balance
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.portfolio = {}
        self.peak_value = initial_balance
        self.total_fees = 0.0
        self._init_portfolio()

    def _init_portfolio(self):
        if self.mode == "simulated":
            self.portfolio = {
                'BTCUSDT': {'position': 0.0, 'free': 0.0},
                'ETHUSDT': {'position': 0.0, 'free': 0.0},
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }
        else:  # live mode
            self.portfolio = {
                'USDT': {'free': self.initial_balance},
                'total': self.initial_balance,
                'peak_value': self.initial_balance,
                'total_fees': 0.0
            }
            for symbol in self.symbols:
                self.portfolio[symbol] = {'position': 0.0, 'free': 0.0}

    def get_portfolio_state(self):
        return self.portfolio.copy()

    def get_balance(self, symbol: str) -> float:
        if symbol == "USDT":
            return self.portfolio.get('USDT', {}).get('free', 0.0)
        else:
            return self.portfolio.get(symbol, {}).get('position', 0.0)

    def save_to_json(self):
        try:
            state_file = "portfolio_state_live.json" if self.mode == "live" else "portfolio_state.json"
            portfolio_state = {
                "portfolio": self.portfolio.copy(),
                "peak_value": self.peak_value,
                "total_fees": self.total_fees,
                "mode": self.mode,
                "initial_balance": self.initial_balance,
                "symbols": self.symbols,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_state, f, indent=2, default=str)
            print(f"💾 Portfolio saved to {state_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving portfolio: {e}")
            return False

    def load_from_json(self):
        try:
            state_file = "portfolio_state_live.json" if self.mode == "live" else "portfolio_state.json"
            if not os.path.exists(state_file):
                print(f"📄 File {state_file} does not exist")
                return False

            with open(state_file, 'r', encoding='utf-8') as f:
                portfolio_state = json.load(f)

            if portfolio_state.get("version") != "1.0":
                print("⚠️ Version mismatch")
                return False

            self.portfolio = portfolio_state.get("portfolio", {})
            self.peak_value = portfolio_state.get("peak_value", self.initial_balance)
            self.total_fees = portfolio_state.get("total_fees", 0.0)

            print(f"📂 Portfolio loaded from {state_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading portfolio: {e}")
            return False

def test_portfolio_persistence():
    """Test portfolio persistence functionality"""
    print("🧪 Testing Portfolio Persistence (Simple)...")

    # Clean up any existing files
    for f in ["portfolio_state_live.json", "portfolio_state.json"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🧹 Cleaned up {f}")

    # Test 1: Initialize PortfolioManager
    print("\n1. Testing PortfolioManager initialization...")
    pm = MockPortfolioManager(mode="live", initial_balance=3000.0, symbols=['BTCUSDT', 'ETHUSDT'])
    print(f"   Initial state: BTC={pm.get_balance('BTCUSDT'):.6f}, ETH={pm.get_balance('ETHUSDT'):.3f}, USDT={pm.get_balance('USDT'):.2f}")

    # Test 2: Save state
    print("\n2. Testing save functionality...")
    success = pm.save_to_json()
    if success:
        print("   ✅ State saved successfully")
        # Check file exists and has content
        state_file = "portfolio_state_live.json"
        if os.path.exists(state_file):
            size = os.path.getsize(state_file)
            print(f"   📁 File size: {size} bytes")
            if size > 0:
                print("   ✅ File has content")
            else:
                print("   ❌ File is empty")
        else:
            print("   ❌ File was not created")
    else:
        print("   ❌ Save failed")

    # Test 3: Load state
    print("\n3. Testing load functionality...")
    pm2 = MockPortfolioManager(mode="live", initial_balance=3000.0, symbols=['BTCUSDT', 'ETHUSDT'])
    loaded = pm2.load_from_json()
    if loaded:
        print("   ✅ State loaded successfully")
        print(f"   Loaded state: BTC={pm2.get_balance('BTCUSDT'):.6f}, ETH={pm2.get_balance('ETHUSDT'):.3f}, USDT={pm2.get_balance('USDT'):.2f}")

        # Verify consistency
        btc_match = abs(pm2.get_balance('BTCUSDT') - pm.get_balance('BTCUSDT')) < 0.000001
        eth_match = abs(pm2.get_balance('ETHUSDT') - pm.get_balance('ETHUSDT')) < 0.000001
        usdt_match = abs(pm2.get_balance('USDT') - pm.get_balance('USDT')) < 0.01

        if btc_match and eth_match and usdt_match:
            print("   ✅ State persistence is consistent!")
        else:
            print("   ❌ State persistence is inconsistent!")
            print(f"      Original: BTC={pm.get_balance('BTCUSDT')}, ETH={pm.get_balance('ETHUSDT')}, USDT={pm.get_balance('USDT')}")
            print(f"      Loaded:   BTC={pm2.get_balance('BTCUSDT')}, ETH={pm2.get_balance('ETHUSDT')}, USDT={pm2.get_balance('USDT')}")
    else:
        print("   ❌ Failed to load state")

    print("\n🎯 Simple portfolio persistence test completed!")

if __name__ == "__main__":
    test_portfolio_persistence()
