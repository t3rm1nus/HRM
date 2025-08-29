import os
from binance.client import Client
from dotenv import load_dotenv

# Cargar desde la raíz
load_dotenv()  

def test_binance_connection():
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_API_SECRET')
        
        print(f"API Key: {'✅' if api_key else '❌'}")
        print(f"Secret Key: {'✅' if secret_key else '❌'}")
        
        if not api_key or not secret_key:
            print("❌ Faltan credenciales en .env")
            return False
        
        client = Client(api_key, secret_key, testnet=True)
        
        # Test básico de conexión
        print("Ping:", client.ping())
        print("Server time:", client.get_server_time())
        
        # Test de datos de mercado (no requiere private key)
        btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
        eth_price = client.get_symbol_ticker(symbol="ETHUSDT")
        
        print(f"BTC Price: {btc_price['price']}")
        print(f"ETH Price: {eth_price['price']}")
        
        print("✅ Conexión a Binance Testnet exitosa!")
        return True
        
    except Exception as e:
        print(f"❌ Error en conexión: {e}")
        return False

if __name__ == "__main__":
    test_binance_connection()