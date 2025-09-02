from binance.client import Client

client = Client(api_key="", api_secret="")  # vacío para público
klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=5)
print(klines)