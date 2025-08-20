# comms/config.py
import os

# Claves API (cargar desde variables de entorno por seguridad)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "EfI5aIIX4TeKu9hUhcGLOSi0RkSROoRZPKx90zx17rxncuDbAJ1KOWqOkVE9Jkq6")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "So7jYGP7jDm1XlOTA5jdxN99bYP9bw87Ajnr3cELgxoTO1rWn3ty3O99pklxlzO5")

# Opciones
USE_TESTNET = True  # True = usar Binance Testnet, False = real
SYMBOL = "BTC/USDT"