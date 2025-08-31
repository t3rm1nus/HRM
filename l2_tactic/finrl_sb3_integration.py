# finrl_sb3_integration.py - Integración para FinRL SB3
from fix_finrl_sb3 import RobustFinRLSB3

def create_finrl_predictor():
    """Crear predictor FinRL robusto para SB3"""
    return RobustFinRLSB3()

# Para compatibilidad con sistema existente
class FinRLPredictor:
    """Wrapper de compatibilidad"""
    def __init__(self):
        self.predictor = RobustFinRLSB3()
    
    def predict_signals(self, market_data):
        return self.predictor.predict_signals(market_data)
