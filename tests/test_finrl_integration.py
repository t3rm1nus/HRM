# tests/test_finrl_integration.py
from l2_tactic.finrl_integration import FinRLProcessor
import pytest
import numpy as np

def test_generate_signal():
    processor = FinRLProcessor("models/L2/ai_model_data_multiasset")
    market_data = {
        "BTCUSDT": {
            "ohlcv": {"close": 108957.13, "high": 109117.99, "low": 108880.0, "volume": 100},
            "indicators": {"rsi": 24.34, "macd": -22.6365, "macd_signal": 0, "bb_upper": 109200, "bb_lower": 108700, "sma_20": 109000, "ema_12": 109000},
            "change_24h": 0.01
        }
    }
    signal = processor.generate_signal(market_data, "BTCUSDT")
    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.side in ["buy", "sell"]
    assert signal.source == "ai"