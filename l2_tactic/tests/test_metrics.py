# l2_tactic/tests/test_metrics.py
import pytest
from l2_tactic.models import TacticalSignal, PositionSize  # Import absoluto
from l2_tactic.metrics import L2Metrics
from l2_tactic.config import L2Config

def test_update_and_hit_rate():
    """Test básico de métricas: update y hit rate."""
    metrics = L2Metrics()
    
    # Simular algunos trades exitosos y fallidos
    metrics.update(success=True, pnl=100.0)
    metrics.update(success=False, pnl=-50.0)
    metrics.update(success=True, pnl=75.0)
    
    # Verificar hit rate
    hit_rate = metrics.hit_rate()
    assert hit_rate == 2/3  # 2 exitosos de 3 total
    
def test_sharpe_ratio_positive():
    """Test de Sharpe ratio con trades positivos."""
    metrics = L2Metrics()
    
    # Agregar trades con PnL positivo neto
    for _ in range(5):
        metrics.update(success=True, pnl=100.0)
    for _ in range(2):
        metrics.update(success=False, pnl=-30.0)
    
    sharpe = metrics.sharpe_ratio()
    assert sharpe > 0  # Debería ser positivo con PnL neto positivo

def test_drawdown_tracking():
    """Test de tracking de drawdown máximo."""
    metrics = L2Metrics()
    
    # Secuencia que genera drawdown
    metrics.update(success=True, pnl=100.0)   # +100
    metrics.update(success=True, pnl=50.0)    # +150
    metrics.update(success=False, pnl=-80.0)  # +70 (DD de 80 desde peak de 150)
    metrics.update(success=False, pnl=-30.0)  # +40 (DD de 110 desde peak de 150)
    
    max_dd = metrics.max_drawdown()
    assert max_dd >= 0  # Drawdown debe ser positivo
    assert max_dd <= 150  # No puede ser mayor que el peak

def test_precision_recall_calculation():
    """Test de cálculo de precisión y recall para predicciones IA."""
    metrics = L2Metrics()
    
    # Simular predicciones y outcomes
    # TP: 2, FP: 1, FN: 1, TN: 1
    metrics.update_prediction(y_true=1, y_pred=1)  # TP
    metrics.update_prediction(y_true=1, y_pred=1)  # TP  
    metrics.update_prediction(y_true=0, y_pred=1)  # FP
    metrics.update_prediction(y_true=1, y_pred=0)  # FN
    metrics.update_prediction(y_true=0, y_pred=0)  # TN
    
    model_metrics = metrics.model_metrics()
    
    # Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.67
    assert abs(model_metrics["precision"] - 2/3) < 0.01
    
    # Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.67  
    assert abs(model_metrics["recall"] - 2/3) < 0.01
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    expected_f1 = 2 * (2/3 * 2/3) / (2/3 + 2/3)
    assert abs(model_metrics["f1"] - expected_f1) < 0.01