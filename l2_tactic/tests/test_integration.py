# l2_tactic/tests/test_integration.py
import pytest
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import TacticalSignal, MarketFeatures, PositionSize
from l2_tactic.config import L2Config

@pytest.mark.asyncio
async def test_end_to_end_pipeline(sample_portfolio, fake_market_data, fake_features, mock_ai):
    """Pipeline completo: signals → sizing → risk controls."""
    config = L2Config()  # Asumimos config por defecto
    processor = L2TacticProcessor(config=config, ai_model=mock_ai)  # Agregado 'ai_model' para definirlo en el constructor

    # 1. Generar señal
    signal = TacticalSignal(symbol="BTC/USDT", side="buy", price=27000, confidence=0.8, strength=0.7)

    # 2. Features de mercado
    features = MarketFeatures(volatility=0.3, adv_notional=1e6)  # Usamos campos compatibles basado en el código

    # 3. Pasamos por el pipeline: sizing
    position_size = await processor.sizer.calculate_position_size(signal, features, {"total_capital": 100000})

    assert position_size is not None
    assert position_size.notional > 0

    # 4. Risk control - Usar el método correcto
    portfolio_state = {
        "total_capital": 100000,
        "daily_pnl": 0.0
    }
    
    allow_trade, alerts, adjusted_position = processor.risk.evaluate_pre_trade_risk(
        signal=signal,
        position_size=position_size,
        market_features=features,
        portfolio_state=portfolio_state,
        correlation_matrix=None
    )
    
    # Verificaciones
    assert isinstance(allow_trade, bool)
    assert isinstance(alerts, list)
    
    # Si se permite el trade, debe haber una posición ajustada
    if allow_trade:
        assert adjusted_position is not None
        assert adjusted_position.notional > 0
    
    # Si hay alertas críticas, el trade no debería permitirse
    critical_alerts = [a for a in alerts if a.severity.value == "critical"]
    if critical_alerts:
        assert not allow_trade
