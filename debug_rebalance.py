import asyncio
from core.simulated_exchange_client import SimulatedExchangeClient
from core.portfolio_manager import PortfolioManager
from core.position_rotator import AutoRebalancer
from l1_operational.position_manager import PositionManager

class MockStateManager:
    def __init__(self):
        self.state = {}

    def get_state(self):
        return self.state

    def update_state(self, key, value):
        self.state[key] = value

async def debug_rebalance():
    print("🔍 Debugging rebalance...")
    
    # Create simulated exchange client with initial balances that are highly imbalanced
    initial_balances = {
        "BTC": 0.08,
        "ETH": 0.0,
        "USDT": 200.0
    }
    
    client = SimulatedExchangeClient(initial_balances)
    
    # Create portfolio manager
    portfolio_manager = PortfolioManager(
        mode="simulated",
        initial_balance=3000.0,
        client=client
    )
    
    # Force system mode to simulated
    from core.config import set_config_value
    set_config_value("mode", "simulated")
    
    btc_price = 75000.0
    eth_price = 4000.0
    
    # Directly set portfolio balances
    portfolio_manager.portfolio = {
        "BTCUSDT": {"position": initial_balances["BTC"], "free": initial_balances["BTC"]},
        "ETHUSDT": {"position": initial_balances["ETH"], "free": initial_balances["ETH"]},
        "USDT": {"free": initial_balances["USDT"]},
        "total": initial_balances["BTC"] * btc_price + initial_balances["ETH"] * eth_price + initial_balances["USDT"],
        "peak_value": initial_balances["BTC"] * btc_price + initial_balances["ETH"] * eth_price + initial_balances["USDT"],
        "total_fees": 0.0,
    }
    
    # Create state manager
    state_manager = MockStateManager()
    
    # Create position manager
    config = {"PAPER_MODE": True}
    position_manager = PositionManager(state_manager, portfolio_manager, config)
    
    # Create auto rebalancer
    auto_rebalancer = AutoRebalancer(portfolio_manager)
    
    # Print current portfolio
    print("\nCurrent Portfolio:")
    print(f"BTC: {portfolio_manager.get_balance('BTCUSDT'):.6f}")
    print(f"ETH: {portfolio_manager.get_balance('ETHUSDT'):.6f}")
    print(f"USDT: {portfolio_manager.get_balance('USDT'):.2f}")
    print(f"Total: {portfolio_manager.portfolio['total']:.2f}")
    
    # Debug: Print allocation percentages
    btc_value = portfolio_manager.get_balance("BTCUSDT") * btc_price
    eth_value = portfolio_manager.get_balance("ETHUSDT") * eth_price
    usdt_value = portfolio_manager.get_balance("USDT")
    total_value = btc_value + eth_value + usdt_value
    
    print("\nAllocation Percentages:")
    print(f"BTC: {(btc_value / total_value) * 100:.1f}%")
    print(f"ETH: {(eth_value / total_value) * 100:.1f}%")
    print(f"USDT: {(usdt_value / total_value) * 100:.1f}%")
    
    # Check why rebalance isn't being triggered
    try:
        from core.config import get_config
        config = get_config("simulated")
        print(f"\nSystem Mode: {getattr(config, 'mode', 'unknown')}")
        
        # Call the rebalance method directly without decorator restrictions
        print("\nCalling check_and_execute_rebalance:")
        orders = await auto_rebalancer.check_and_execute_rebalance({
            "BTCUSDT": {"close": btc_price},
            "ETHUSDT": {"close": eth_price}
        }, l3_active=False, l3_decision={"allow_l2_signals": True})
        print(f"Generated orders: {len(orders)}")
        if orders:
            for order in orders:
                print(f"   - {order['action']} {order['quantity']:.4f} {order['symbol']}")
        
        # Check portfolio after rebalance
        await portfolio_manager.update_from_orders_async(orders, {
            "BTCUSDT": {"close": btc_price},
            "ETHUSDT": {"close": eth_price}
        })
        
        print("\nAfter Rebalance:")
        print(f"BTC: {portfolio_manager.get_balance('BTCUSDT'):.6f}")
        print(f"ETH: {portfolio_manager.get_balance('ETHUSDT'):.6f}")
        print(f"USDT: {portfolio_manager.get_balance('USDT'):.2f}")
        print(f"Total: {portfolio_manager.portfolio['total']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    try:
        asyncio.run(debug_rebalance())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())