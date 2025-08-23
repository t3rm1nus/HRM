# test_l2_signals.py
import logging
from l2_tactic.config import L2Config, AIModelConfig
from l2_tactic.signal_generator import SignalGenerator
import pandas as pd
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_l2_signals")

# Mock signals configuration
@dataclass
class SignalsConfig:
    universe: list
    min_signal_strength: float = 0.5
    signal_expiry_minutes: int = 60
    ai_model_weight: float = 0.5
    technical_weight: float = 0.3
    pattern_weight: float = 0.2

def test_signal_generator():
    try:
        # Mock L2Config
        config = L2Config(
            ai_model=AIModelConfig(model_name="test_model", model_params={}, signal_horizon_minutes=60),
            signals=SignalsConfig(universe=["BTC/USDT", "ETH/USDT"])
        )
        signal_gen = SignalGenerator(config)
        logger.info("SignalGenerator inicializado correctamente para pruebas")

        # Mock market data
        market_data = {
            "BTC/USDT": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
                "open": [50000.0 + i * 10 for i in range(100)],
                "high": [50500.0 + i * 10 for i in range(100)],
                "low": [49500.0 + i * 10 for i in range(100)],
                "close": [50200.0 + i * 10 for i in range(100)],
                "volume": [100.0] * 100
            }).set_index("timestamp"),
            "ETH/USDT": pd.DataFrame({
                "timestamp": pd.date_range(start="2025-08-23 21:03:00", periods=100, freq="1min", tz="UTC"),
                "open": [2000.0 + i * 5 for i in range(100)],
                "high": [2020.0 + i * 5 for i in range(100)],
                "low": [1980.0 + i * 5 for i in range(100)],
                "close": [2010.0 + i * 5 for i in range(100)],
                "volume": [50.0] * 100
            }).set_index("timestamp")
        }

        # Test signal generation for BTC/USDT
        state = {"mercado": market_data}
        logger.info("Testing signal generation for BTC/USDT")
        signals = signal_gen.generate(state, "BTC/USDT")
        assert len(signals) > 0, "No signals generated for BTC/USDT"
        assert signals[0]["symbol"] == "BTC/USDT", "Incorrect symbol for BTC/USDT"
        assert signals[0]["direction"] in ["buy", "sell", "hold"], f"Invalid direction: {signals[0]['direction']}"
        assert isinstance(signals[0]["confidence"], float), "Confidence is not a float for BTC/USDT"
        assert 0 <= signals[0]["confidence"] <= 1, "Confidence out of range for BTC/USDT"
        logger.info(f"Generated {len(signals)} signals for BTC/USDT: {signals}")

        # Test signal generation for ETH/USDT
        logger.info("Testing signal generation for ETH/USDT")
        signals = signal_gen.generate(state, "ETH/USDT")
        assert len(signals) > 0, "No signals generated for ETH/USDT"
        assert signals[0]["symbol"] == "ETH/USDT", "Incorrect symbol for ETH/USDT"
        assert signals[0]["direction"] in ["buy", "sell", "hold"], f"Invalid direction: {signals[0]['direction']}"
        assert isinstance(signals[0]["confidence"], float), "Confidence is not a float for ETH/USDT"
        assert 0 <= signals[0]["confidence"] <= 1, "Confidence out of range for ETH/USDT"
        logger.info(f"Generated {len(signals)} signals for ETH/USDT: {signals}")

        # Test signal generation for invalid symbol
        logger.info("Testing signal generation for invalid symbol")
        signals = signal_gen.generate(state, "INVALID")
        assert len(signals) == 0, "Signals generated for invalid symbol"
        logger.info("No signals generated for invalid symbol, as expected")

        logger.info("All tests passed successfully")
    except Exception as e:
        logger.error(f"Error en las pruebas: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    test_signal_generator()