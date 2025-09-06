from binance.client import Client
client = Client("TU_API_KEY", "TU_API_SECRET", testnet=True)  # o False si es mainnet
klines = client.get_klines(symbol="BTCUSDT", interval="1m", limit=5)
print(klines)