# l2_tactic/generators/technical_analyzer.py
from typing import Dict, List

class TechnicalAnalyzer:
    def __init__(self, config: Dict):
        self.config = config

    def generate_signals(self, market_data: Dict) -> List[Dict]:
        # Lógica para generar señales técnicas
        signals = []
        # Ejemplo de señal técnica
        signals.append({
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'confidence': 0.9,
            'strength': 0.8,
            'source': 'technical',
            'price': market_data['BTCUSDT']['price'],
            'stop_loss': market_data['BTCUSDT']['price'] * 0.98,
        })
        return signals