# l2_tactic/generators/mean_reversion.py
# DISABLED: Pure trend-following system - mean-reversion deactivated
# This generator no longer produces signals as HRM shifts to pure trend-following

from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RangeBoundStrategy:
    """Strategy for range-bound markets"""

    def __init__(self, config: Dict):
        self.config = config

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signal(self, symbol, market_data):
        """Estrategia específica para ranges"""

        # Confirmar que estamos en range
        price_range_20 = (market_data['high'].iloc[-20:].max() -
                         market_data['low'].iloc[-20:].min())
        current_price = market_data['close'].iloc[-1]

        # Calcular posición dentro del range
        range_position = ((current_price - market_data['low'].iloc[-20:].min()) /
                         price_range_20)

        # Indicadores adicionales
        rsi = self._calculate_rsi(market_data)
        volume_surge = market_data['volume'].iloc[-1] > market_data['volume'].rolling(20).mean().iloc[-1] * 1.5

        # Señal de compra (fondo del range)
        if range_position < 0.3 and rsi < 35 and volume_surge:
            return {
                "action": "BUY",
                "confidence": 0.7,
                "stop_loss": current_price * 0.98,  # Stop ajustado
                "take_profit": current_price * 1.02,  # Target pequeño
                "position_size_multiplier": 0.6  # Posición reducida
            }

        # Señal de venta (techo del range)
        elif range_position > 0.7 and rsi > 65 and volume_surge:
            return {
                "action": "SELL",
                "confidence": 0.7,
                "stop_loss": current_price * 1.02,
                "take_profit": current_price * 0.98,
                "position_size_multiplier": 0.6
            }

        else:
            return {"action": "HOLD", "confidence": 0.5}

class MeanReversion:
    def __init__(self, config: Dict):
        self.config = config
        # Range-bound strategy now activated for range markets
        self.range_strategy = RangeBoundStrategy(config)

    def generate_signals(self, market_data: Dict) -> List[Dict]:
        # Now includes range-bound mean-reversion logic
        signals = []

        for symbol, data in market_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                signal = self.range_strategy.generate_signal(symbol, data)
                if signal['action'] != 'HOLD':
                    signals.append({
                        'symbol': symbol,
                        'signal': signal
                    })

        return signals
