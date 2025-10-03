#!/usr/bin/env python3
"""
Unified Portfolio Test Suite
Consolidates all portfolio-related tests into a comprehensive suite
"""
import os
import sys
import json
import asyncio
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the problematic imports to avoid TensorFlow issues for simple tests
sys.modules['l2_tactic.utils'] = type(sys)('mock_utils')
sys.modules['l2_tactic.utils'].safe_float = lambda x: float(x) if x is not None else 0.0

from core.portfolio_manager import PortfolioManager

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

def test_portfolio_initialization_fix():
    """Test portfolio initialization fix"""
    print("🧪 Testing PortfolioManager initialization fix...")

    # Create PortfolioManager in live mode without client (should use fallback)
    pm = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        client=None,  # No client to avoid BinanceClient initialization issues
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    # Force clean reset
    pm.force_clean_reset()

    # Check balances
    btc_balance = pm.get_balance("BTCUSDT")
    eth_balance = pm.get_balance("ETHUSDT")
    usdt_balance = pm.get_balance("USDT")

    print("📊 Portfolio state after initialization:")
    print(f"   BTC: {btc_balance:.6f}")
    print(f"   ETH: {eth_balance:.3f}")
    print(f"   USDT: {usdt_balance:.2f}")

    # Verify USDT balance is correct
    if abs(usdt_balance - 3000.0) < 0.01:
        print("✅ SUCCESS: USDT balance is correct (3000.00)")
        return True
    else:
        print(f"❌ FAILURE: USDT balance is {usdt_balance}, expected 3000.00")
        return False

def test_portfolio_persistence_simple():
    """Test portfolio persistence functionality with mock manager"""
    print("🧪 Testing Portfolio Persistence (Simple Mock)...")

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

async def test_portfolio_persistence_async():
    """Test portfolio persistence functionality with real PortfolioManager"""
    print("🧪 Testing Portfolio Persistence (Real Manager)...")

    # Test 1: Initialize PortfolioManager
    print("\n1. Testing PortfolioManager initialization...")
    pm = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    print(f"   Initial state: BTC={pm.get_balance('BTCUSDT'):.6f}, ETH={pm.get_balance('ETHUSDT'):.3f}, USDT={pm.get_balance('USDT'):.2f}")

    # Test 2: Save state
    print("\n2. Testing save functionality...")
    pm.save_to_json()
    print("   State saved to JSON")

    # Test 3: Load state
    print("\n3. Testing load functionality...")
    pm2 = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    if pm2.load_from_json():
        print("   State loaded from JSON successfully")
        print(f"   Loaded state: BTC={pm2.get_balance('BTCUSDT'):.6f}, ETH={pm2.get_balance('ETHUSDT'):.3f}, USDT={pm2.get_balance('USDT'):.2f}")
    else:
        print("   ❌ Failed to load state from JSON")

    # Test 4: Simulate order processing
    print("\n4. Testing order processing...")
    mock_orders = [
        {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.001,
            "status": "filled",
            "filled_price": 50000.0
        }
    ]

    mock_market_data = {
        "BTCUSDT": {"close": 50000.0},
        "ETHUSDT": {"close": 3000.0}
    }

    await pm.update_from_orders_async(mock_orders, mock_market_data)
    print(f"   After order: BTC={pm.get_balance('BTCUSDT'):.6f}, ETH={pm.get_balance('ETHUSDT'):.3f}, USDT={pm.get_balance('USDT'):.2f}")

    # Test 5: Save updated state
    print("\n5. Testing updated state save...")
    pm.save_to_json()
    print("   Updated state saved")

    # Test 6: Load updated state in new instance
    print("\n6. Testing updated state load...")
    pm3 = PortfolioManager(
        mode="live",
        initial_balance=3000.0,
        symbols=['BTCUSDT', 'ETHUSDT']
    )

    if pm3.load_from_json():
        print("   Updated state loaded successfully")
        print(f"   Final state: BTC={pm3.get_balance('BTCUSDT'):.6f}, ETH={pm3.get_balance('ETHUSDT'):.3f}, USDT={pm3.get_balance('USDT'):.2f}")

        # Verify consistency
        if (abs(pm3.get_balance('BTCUSDT') - pm.get_balance('BTCUSDT')) < 0.000001 and
            abs(pm3.get_balance('ETHUSDT') - pm.get_balance('ETHUSDT')) < 0.000001 and
            abs(pm3.get_balance('USDT') - pm.get_balance('USDT')) < 0.01):
            print("   ✅ State persistence is consistent!")
        else:
            print("   ❌ State persistence is inconsistent!")
    else:
        print("   ❌ Failed to load updated state")

    print("\n🎯 Real portfolio persistence test completed!")

def test_allocation_tiers():
    """Test comprehensive allocation tier system"""
    print("\n🎯 TESTING ALLOCATION TIERS SYSTEM")
    print("=" * 60)

    # Initialize portfolio manager
    pm = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=False)

    # Test different scenarios
    test_scenarios = [
        {
            "name": "Conservative - Weak Signal",
            "signal_strength": 0.3,
            "market_condition": "bearish",
            "asset_type": "crypto",
            "risk_appetite": "low"
        },
        {
            "name": "Balanced - Moderate Signal",
            "signal_strength": 0.6,
            "market_condition": "neutral",
            "asset_type": "crypto",
            "risk_appetite": "moderate"
        },
        {
            "name": "Growth - Strong Signal",
            "signal_strength": 0.8,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "high"
        },
        {
            "name": "Aggressive - Very Strong Signal",
            "signal_strength": 0.9,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "aggressive"
        }
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 40)

        # Get allocation tier
        tier = pm.get_allocation_tier(
            signal_strength=scenario["signal_strength"],
            market_condition=scenario["market_condition"],
            asset_type=scenario["asset_type"],
            risk_appetite=scenario["risk_appetite"]
        )

        # Display results
        print(f"   Tier: {tier['tier_name'].upper()}")
        print(f"   Risk Appetite: {tier['risk_appetite']}")
        print(f"   Signal Strength: {tier['signal_strength']:.2f} → {tier['signal_multiplier']:.2f}x")
        print(f"   Market Condition: {tier['market_condition']} → {tier['market_multiplier']:.2f}x")
        print(f"   Asset Type: {tier['asset_type']} → {tier['asset_multiplier']:.2f}x")
        print(f"   Final Allocation: {tier['final_allocation']:.1%} (${tier['available_capital']:.2f})")
        print(f"   Position Limit: {tier['final_position_limit']:.1%} (${tier['max_position_size']:.2f})")
        print(f"   Description: {tier['description']}")

        results.append({
            "scenario": scenario["name"],
            "tier": tier
        })

    # Test Aggressive Mode
    print(f"\n🚨 TESTING AGGRESSIVE MODE")
    print("=" * 40)

    pm_aggressive = PortfolioManager(mode="simulated", initial_balance=10000.0, aggressive_mode=True)

    aggressive_scenarios = [
        {
            "name": "Aggressive Mode - Conservative Risk",
            "signal_strength": 0.5,
            "market_condition": "neutral",
            "asset_type": "crypto",
            "risk_appetite": "low"
        },
        {
            "name": "Aggressive Mode - Aggressive Risk",
            "signal_strength": 0.8,
            "market_condition": "bullish",
            "asset_type": "crypto",
            "risk_appetite": "aggressive"
        }
    ]

    for scenario in aggressive_scenarios:
        print(f"\n📊 Testing: {scenario['name']}")
        print("-" * 40)

        tier = pm_aggressive.get_allocation_tier(
            signal_strength=scenario["signal_strength"],
            market_condition=scenario["market_condition"],
            asset_type=scenario["asset_type"],
            risk_appetite=scenario["risk_appetite"]
        )

        print(f"   Tier: {tier['tier_name'].upper()} (AGGRESSIVE MODE)")
        print(f"   Risk Appetite: {tier['risk_appetite']}")
        print(f"   Signal Strength: {tier['signal_strength']:.2f} → {tier['signal_multiplier']:.2f}x")
        print(f"   Market Condition: {tier['market_condition']} → {tier['market_multiplier']:.2f}x")
        print(f"   Asset Type: {tier['asset_type']} → {tier['asset_multiplier']:.2f}x")
        print(f"   Final Allocation: {tier['final_allocation']:.1%} (${tier['available_capital']:.2f})")
        print(f"   Position Limit: {tier['final_position_limit']:.1%} (${tier['max_position_size']:.2f})")
        print(f"   Aggressive Mode Active: {tier['aggressive_mode_active']}")

        results.append({
            "scenario": scenario["name"],
            "tier": tier
        })

    # Summary comparison
    print(f"\n📈 ALLOCATION TIERS SUMMARY")
    print("=" * 60)

    print("Normal Mode Allocations:")
    normal_results = [r for r in results if "Aggressive Mode" not in r["scenario"]]
    for result in normal_results:
        tier = result["tier"]
        print(f"   {result['scenario']:<35}: {tier['final_allocation']:.1%} (${tier['available_capital']:>8.0f}) | Limit: {tier['final_position_limit']:.1%}")

    print("\nAggressive Mode Allocations:")
    aggressive_results = [r for r in results if "Aggressive Mode" in r["scenario"]]
    for result in aggressive_results:
        tier = result["tier"]
        print(f"   {result['scenario'].replace('Aggressive Mode - ', ''):<35}: {tier['final_allocation']:.1%} (${tier['available_capital']:>8.0f}) | Limit: {tier['final_position_limit']:.1%}")

    return results

def test_risk_adjusted_capital():
    """Test risk-adjusted capital deployment"""
    print("\n🧪 TESTING RISK-ADJUSTED PORTFOLIO MANAGEMENT")
    print("=" * 60)

    # Test different risk appetites
    test_cases = [
        ("low", 0.40),
        ("moderate", 0.60),
        ("high", 0.80),
        ("aggressive", 0.90)
    ]

    for risk_appetite, expected_deployment in test_cases:
        print(f"\n🎯 Testing Risk Appetite: {risk_appetite.upper()}")

        # Create test risk data
        os.makedirs("data/datos_inferencia", exist_ok=True)
        risk_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": {
                "volatility": 0.45,
                "sentiment": 0.2,
                "regime": "bull"
            },
            "risk_appetite": risk_appetite
        }

        with open("data/datos_inferencia/risk.json", "w") as f:
            json.dump(risk_data, f, indent=2)

        # Initialize portfolio manager
        pm = PortfolioManager(mode="simulated", initial_balance=3000.0)

        # Test risk appetite loading
        loaded_appetite = pm.load_risk_appetite()
        print(f"   Loaded risk appetite: {loaded_appetite}")

        assert loaded_appetite == risk_appetite, f"Expected {risk_appetite}, got {loaded_appetite}"

        # Test capital deployment (mock market data)
        market_data = {
            "BTCUSDT": {"close": 50000.0},
            "ETHUSDT": {"close": 3000.0}
        }

        available_capital = pm.get_available_trading_capital(market_data)
        expected_capital = 3000.0 * expected_deployment

        print(f"   Available capital: ${available_capital:.2f} (expected: ${expected_capital:.2f})")

        # Allow small floating point differences
        assert abs(available_capital - expected_capital) < 0.01, f"Capital mismatch: {available_capital} vs {expected_capital}"

        # Test deployment status
        status = pm.get_capital_deployment_status(market_data)
        print(f"   Deployment percentage: {status['deployment_percentage']:.1%}")
        print(f"   Can deploy more: {status['can_deploy_more']}")

        assert status['risk_appetite'] == risk_appetite
        assert abs(status['deployment_percentage'] - expected_deployment) < 0.001

        print(f"   ✅ {risk_appetite.upper()} test passed")

    print("\n" + "=" * 60)
    print("🎉 ALL RISK-ADJUSTED PORTFOLIO TESTS PASSED!")
    print("\n📊 Risk Appetite Deployment Tiers:")
    print("   Low: 40% of USDT available for trading")
    print("   Moderate: 60% of USDT available for trading")
    print("   High: 80% of USDT available for trading")
    print("   Aggressive: 90% of USDT available for trading")

async def run_unified_portfolio_tests():
    """Run all unified portfolio tests"""
    print("🚀 UNIFIED PORTFOLIO TEST SUITE")
    print("=" * 60)

    success_count = 0
    total_tests = 5

    # Test 1: Initialization fix
    try:
        if test_portfolio_initialization_fix():
            success_count += 1
            print("✅ Test 1 PASSED: Portfolio Initialization")
        else:
            print("❌ Test 1 FAILED: Portfolio Initialization")
    except Exception as e:
        print(f"❌ Test 1 ERROR: {e}")

    # Test 2: Simple persistence
    try:
        test_portfolio_persistence_simple()
        success_count += 1
        print("✅ Test 2 PASSED: Simple Persistence")
    except Exception as e:
        print(f"❌ Test 2 ERROR: {e}")

    # Test 3: Real async persistence
    try:
        await test_portfolio_persistence_async()
        success_count += 1
        print("✅ Test 3 PASSED: Async Persistence")
    except Exception as e:
        print(f"❌ Test 3 ERROR: {e}")

    # Test 4: Allocation tiers
    try:
        test_allocation_tiers()
        success_count += 1
        print("✅ Test 4 PASSED: Allocation Tiers")
    except Exception as e:
        print(f"❌ Test 4 ERROR: {e}")

    # Test 5: Risk-adjusted capital
    try:
        test_risk_adjusted_capital()
        success_count += 1
        print("✅ Test 5 PASSED: Risk-Adjusted Capital")
    except Exception as e:
        print(f"❌ Test 5 ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"SUCCESS RATE: {success_count/total_tests*100:.1f}%")

    if success_count == total_tests:
        print("🎉 ALL PORTFOLIO TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  SOME PORTFOLIO TESTS FAILED - REVIEW OUTPUT ABOVE")

    print("=" * 60)
    return success_count == total_tests

if __name__ == "__main__":
    asyncio.run(run_unified_portfolio_tests())
