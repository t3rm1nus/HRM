# l2_tactic/tests/test_signal_generator.py
import pytest
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import TacticalSignal, MarketFeatures
from l2_tactic.config import L2Config

@pytest.mark.asyncio
async def test_signal_generator_produces_signals(sample_portfolio, fake_market_data, fake_features, mock_ai):
    """El generador debe producir señales sin romper."""
    config = L2Config()  # Asumimos config por defecto
    processor = L2TacticProcessor(config=config, ai_model=mock_ai)  # Ahora el constructor acepta ai_model

    signal = TacticalSignal(symbol="ETH/USDT", side="buy", price=1800, confidence=0.6, strength=0.7)
    # Usar campos que existen realmente en MarketFeatures
    features = MarketFeatures(volatility=0.4, adv_notional=2e6)

    order = await processor.sizer.calculate_position_size(signal, features, {"total_capital": 200000})

    assert order is not None
    assert order.size > 0
    assert order.notional > 0
