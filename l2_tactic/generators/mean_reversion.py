# l2_tactic/generators/mean_reversion.py
from typing import Dict, List

class MeanReversion:
    def __init__(self, config: Dict):
        self.config = config
        self.mean_reversion_threshold = config.get('mean_reversion_threshold', 0.1)

    def generate_signals(self, market_data: Dict) -> List[Dict]:
        signals = []
        for symbol, data in market_data.items():
            current_price = data.get('price', 0)
            mean_price = data.get('mean_price', 0)
            std_dev = data.get('std_dev', 0)

            if current_price > mean_price + self.mean_reversion_threshold * std_dev:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'confidence': 0.7,
                    'strength': 0.6,
                    'source': 'mean_reversion',
                    'price': current_price,
                    'stop_loss': current_price * 0.98,
                })
            elif current_price < mean_price - self.mean_reversion_threshold * std_dev:
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'confidence': 0.7,
                    'strength': 0.6,
                    'source': 'mean_reversion',
                    'price': current_price,
                    'stop_loss': current_price * 1.02,
                })
        return signals