import pytest
import asyncio
from datetime import datetime
from l2_tactic.models import TacticalSignal

@pytest.fixture
def sample_portfolio():
    """Portafolio inicial simulado."""
    return {"BTC/USDT": 1000.0, "ETH/USDT": 500.0, "USDT": 10000.0}

@pytest.fixture
def fake_market_data():
    """Datos OHLCV simulados mínimos para BTC y ETH."""
    now = datetime.utcnow().isoformat()
    return {
        "BTC/USDT": {
            "timestamp": now,
            "open": 27000,
            "high": 27200,
            "low": 26800,
            "close": 27100,
            "volume": 1234,
        },
        "ETH/USDT": {
            "timestamp": now,
            "open": 1800,
            "high": 1820,
            "low": 1790,
            "close": 1810,
            "volume": 5678,
        },
    }

@pytest.fixture
def fake_features():
    """Features dummy para alimentar al modelo IA."""
    return {
        "BTC/USDT": {"feature1": 1.0, "feature2": 2.0},
        "ETH/USDT": {"feature1": 0.5, "feature2": -1.0},
    }

@pytest.fixture(autouse=True)
def mock_ai(monkeypatch):
    """Mock global de la IA para todos los tests."""
    async def fake_predict(self, features):
        # Simula predicciones determinísticas
        return {
            "BTC/USDT": {"direction": "BUY", "confidence": 0.9},
            "ETH/USDT": {"direction": "SELL", "confidence": 0.7},
        }
    monkeypatch.setattr("l2_tactic.ai_model_integration.AIModelIntegration.predict", fake_predict)
    yield  # cleanup automático
