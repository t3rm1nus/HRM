#!/usr/bin/env python3
"""Test script to verify the import fixes work"""

try:
    from l2_tactic.models import TacticalSignal, PositionSize, RiskMetrics, L2State
    print("✅ All imports successful!")

    # Test TacticalSignal creation
    signal = TacticalSignal(
        symbol="BTCUSDT",
        strength=0.8,
        confidence=0.9,
        side="buy",
        features={"rsi": 45.0, "macd": -10.0}
    )
    print(f"✅ TacticalSignal created: {signal}")

    # Test PositionSize creation
    pos_size = PositionSize(
        symbol="BTCUSDT",
        side="buy",
        price=50000.0,
        size=0.01,
        notional=500.0,
        risk_amount=10.0,
        kelly_fraction=0.5,
        vol_target_leverage=1.0,
        max_loss=10.0
    )
    print(f"✅ PositionSize created: {pos_size.symbol}")

    # Test RiskMetrics creation
    risk_metrics = RiskMetrics(
        symbol="BTCUSDT",
        timestamp="2025-01-01T00:00:00",
        position_risk=10.0,
        portfolio_heat=0.1,
        correlation_risk=0.05,
        liquidity_risk=0.02,
        volatility_risk=0.03,
        max_drawdown_risk=0.15,
        total_risk_score=0.25
    )
    print(f"✅ RiskMetrics created: {risk_metrics.symbol}")

    # Test L2State creation
    l2_state = L2State()
    print(f"✅ L2State created with {len(l2_state.signals)} signals")

    print("\n🎉 All tests passed! The import fixes are working correctly.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
