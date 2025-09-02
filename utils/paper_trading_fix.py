import time


class PaperTradingFix:
    def __init__(self):
        self.current_prices = {
            'BTCUSDT': 65000.0,
            'ETHUSDT': 3200.0
        }
    
    def get_current_price(self, symbol):
        """Get current price with fallback"""
        return self.current_prices.get(symbol, 0.0)
    
    def execute_paper_order(self, symbol, side, quantity, price=None):
        """Execute paper order with proper price filling"""
        if price is None or price == 0:
            price = self.get_current_price(symbol)
        
        order = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'status': 'filled',
            'order_id': f'paper_{symbol}_{int(time.time())}',
            'execution_mode': 'paper',
            'filled_price': price,
            'commission': 0,
            'timestamp': time.time()
        }
        
        return order
