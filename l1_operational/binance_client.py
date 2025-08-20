# l1_operational/binance_client.py
import ccxt

exchange = ccxt.binance({
    'apiKey': 'EfI5aIIX4TeKu9hUhcGLOSi0RkSROoRZPKx90zx17rxncuDbAJ1KOWqOkVE9Jkq6',
    'secret': 'So7jYGP7jDm1XlOTA5jdxN99bYP9bw87Ajnr3cELgxoTO1rWn3ty3O99pklxlzO5',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
})
exchange.set_sandbox_mode(True)  # habilita testnet
