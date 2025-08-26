# test_risk_controls.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import logging
from datetime import datetime
from l2_tactic.risk_controls.alerts import RiskLevel
from l2_tactic.risk_controls.manager import RiskControlManager
from l2_tactic.models import TacticalSignal, PositionSize, MarketFeatures
from l2_tactic.config import L2Config


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_test():
    # Config mínima
    config = L2Config(
        default_stop_pct=0.02,
        atr_multiplier=2.0,
        trailing_stop_pct=0.01,
        breakeven_threshold=1.5,
        take_profit_rr_min=1.5,
        take_profit_rr_max=2.5,
        max_correlation=0.7,
        max_portfolio_heat=0.8,
        daily_loss_limit=0.05,
        max_drawdown_limit=0.15,
        max_positions=3,
        min_liquidity_notional=5000,
        min_liquidity_ratio=0.1,
        max_signal_drawdown=0.2,
        max_strategy_drawdown=0.25,
    )

    risk_mgr = RiskControlManager(config)

    # Señal artificial (ej: compra BTC a 25k)
    signal = TacticalSignal(
        symbol="BTC/USDT",
        side="buy",
        price=25000,
        confidence=0.8,
        strength=0.7,
        source="TEST",
        timestamp=datetime.utcnow(),
    )

    # Tamaño propuesto (ejemplo: 0.5 BTC)
    pos_size = PositionSize(
        symbol="BTC/USDT",
        side="buy",
        price=25000,
        size=0.5,
        notional=12500,
        risk_amount=250,
        kelly_fraction=0.5,
        vol_target_leverage=1.0,
        max_loss=250,
    )

    # Market features artificiales
    mf = MarketFeatures(
        volatility=0.6,
        atr=400,
        support=24000,
        resistance=26000,
        adv_notional=20000,   # liquidez simulada
        volume=100,           # volumen (ejemplo)
        price=25000,
    )

    # Estado del portfolio
    portfolio_state = {
        "total_capital": 20000,   # bajo capital
        "daily_pnl": -2000,       # pérdida diaria fuerte
    }

    # Evaluar pre-trade
    allow, alerts, adjusted = risk_mgr.evaluate_pre_trade_risk(
        signal, pos_size, mf, portfolio_state
    )

    print("\n=== PRE-TRADE EVALUATION ===")
    print("Trade allowed:", allow)
    print("Adjusted size:", adjusted)
    for a in alerts:
        print("ALERT:", a)

    # Añadir posición
    if adjusted:
        risk_mgr.add_position(signal, adjusted, mf)

    # Simular precios que disparen stop-loss
    price_data = {"BTC/USDT": 23000}  # cae fuerte → stop debería saltar
    alerts = risk_mgr.monitor_existing_positions(price_data, portfolio_value=18000)

    print("\n=== MONITORING ===")
    for a in alerts:
        print("ALERT:", a)


if __name__ == "__main__":
    run_test()
