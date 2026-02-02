#!/usr/bin/env python3
"""
Proveedor de datos simulados para respaldo en modo paper.
Este modulo proporciona datos de mercado simulados cuando no se pueden obtener datos reales.
"""

import random
from datetime import datetime
from typing import Dict, Any, List

class MockMarketData:
    """Generador de datos de mercado simulados."""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0
        }
        self.last_prices = self.base_prices.copy()
        
    def generate_price_data(self, symbol: str) -> Dict[str, Any]:
        """Genera datos de precio simulados para un simbolo."""
        
        # Simular movimiento de precios con volatilidad realista
        base_price = self.base_prices[symbol]
        volatility = random.uniform(0.001, 0.02)  # 0.1% a 2% de volatilidad
        trend = random.uniform(-1, 1)  # Tendencia aleatoria
        
        # Calcular nuevo precio con drift hacia la media
        mean_reversion = (base_price - self.last_prices[symbol]) * 0.001
        price_change = base_price * volatility * trend + mean_reversion
        new_price = max(0.01, self.last_prices[symbol] + price_change)
        
        self.last_prices[symbol] = new_price
        
        # Generar datos de vela
        open_price = self.last_prices[symbol]
        close_price = new_price
        high_price = max(open_price, close_price) + random.uniform(0, 100)
        low_price = min(open_price, close_price) - random.uniform(0, 100)
        
        return {
            'symbol': symbol,
            'price': new_price,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': random.uniform(10, 1000),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'mock_data',
            'status': 'simulated'
        }
    
    def get_all_symbols_data(self) -> List[Dict[str, Any]]:
        """Obtiene datos simulados para todos los simbolos."""
        return [self.generate_price_data(symbol) for symbol in self.symbols]

# Instancia global para uso en el sistema
mock_data_provider = MockMarketData()

def get_mock_market_data(symbols: List[str] = None) -> List[Dict[str, Any]]:
    """Obtiene datos de mercado simulados."""
    if symbols:
        provider = MockMarketData(symbols)
        return provider.get_all_symbols_data()
    else:
        return mock_data_provider.get_all_symbols_data()
