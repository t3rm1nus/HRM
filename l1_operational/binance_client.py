# l1_operational/binance_client.py
import os
import ccxt

# Lectura segura desde variables de entorno
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')
MODE = os.getenv('BINANCE_MODE', 'PAPER').upper()  # PAPER | LIVE

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
})

# Modo por defecto: PAPER (sin claves reales). Cambiar a LIVE bajo tu responsabilidad.
if MODE == 'PAPER':
    try:
        exchange.set_sandbox_mode(True)
    except Exception:
        pass
