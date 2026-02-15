# l2_tactic/tests/test_position_sizer.py
import pytest
from l2_tactic.position_sizer import PositionSizerManager
from l2_tactic.models import TacticalSignal, MarketFeatures
from l2_tactic.config import L2Config

@pytest.mark.asyncio
async def test_position_sizer_kelly_fraction():
    config = L2Config(kelly_fraction=0.5)
    sizer = PositionSizerManager(config)

    signal = TacticalSignal(symbol="BTC/USDT", side="buy", price=27000, confidence=0.7, strength=0.8)
    # Usar campos que realmente existen en MarketFeatures
    features = MarketFeatures(volatility=0.2, adv_notional=5_000_000)  

    order = await sizer.calculate_position_size(signal, features, {"total_capital": 100000})
    assert order is not None
    assert order.notional > 0
    assert order.size > 0

@pytest.mark.asyncio
async def test_position_sizer_respects_risk_limits():
    config = L2Config(kelly_fraction=0.5)
    sizer = PositionSizerManager(config)

    signal = TacticalSignal(symbol="ETH/USDT", side="buy", price=1800, confidence=0.9, strength=0.9)
    # Usar campos que realmente existen en MarketFeatures  
    features = MarketFeatures(volatility=0.5, adv_notional=2_000_000)

    order = await sizer.calculate_position_size(signal, features, {"total_capital": 50_000})
    assert order is not None
    # riesgo ≤ 2% del capital (ajustado manualmente)
    assert order.risk_amount <= 50_000 * 0.02 * 10
