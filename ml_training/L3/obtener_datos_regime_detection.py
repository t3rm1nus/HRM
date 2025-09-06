import requests
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import os
import logging
import time

# ========= CONFIG =========
symbols = ["BTCUSDT", "ETHUSDT"]
intervals = ["1d", "1w", "1M"]
years_back = 10
output_folder = "data/datos_para_modelos_l3/modelo1_regime_detector"
os.makedirs(output_folder, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========= FUNCIONES =========
def get_historical_data(symbol, interval, start_time, end_time):
    """Descarga datos históricos desde Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit = 1000

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logging.error(f"Error {response.status_code}: {response.text}")
            break
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1
        time.sleep(1.1)

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    numeric_cols = ['open','high','low','close','volume','quote_asset_volume','trades',
                    'taker_buy_base_asset_volume','taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df

def add_features_and_labels(df):
    """Añade features técnicos y etiquetas de régimen"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']).diff()
    df['volatility'] = df['return'].rolling(30).std()

    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # MACD
    macd, macdsig, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macdsig'] = macdsig
    df['macdhist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
    df['boll_upper'] = upper
    df['boll_middle'] = middle
    df['boll_lower'] = lower

    # ======== ETIQUETAS DE RÉGIMEN ========
    # Definición simple: tendencia (retornos medios) + volatilidad
    trend = df['return'].rolling(30).mean()
    vol = df['volatility']

    conditions = [
        (trend > 0.001) & (vol < 0.02),   # Bull market
        (trend < -0.001) & (vol < 0.02),  # Bear market
        (vol >= 0.02)                     # Volatile
    ]
    choices = ['bull', 'bear', 'volatile']
    df['regime'] = np.select(conditions, choices, default='range')

    df.dropna(inplace=True)
    return df

# ========= DESCARGA =========
start_time = int((datetime.now() - timedelta(days=365*years_back)).timestamp() * 1000)
end_time = int(datetime.now().timestamp() * 1000)

all_data = []
for symbol in symbols:
    for interval in intervals:
        logging.info(f"Descargando {interval} de {symbol}...")
        df = get_historical_data(symbol, interval, start_time, end_time)
        df['symbol'] = symbol
        df['interval'] = interval

        # Añadir features + etiquetas
        df = add_features_and_labels(df)

        # Guardar individual
        out_file = os.path.join(output_folder, f"{symbol}_{interval}_features_labels.csv")
        df.to_csv(out_file, index=False)
        logging.info(f"Guardado {out_file} ({len(df)} filas)")

        all_data.append(df)

# ========= COMBINAR Y GUARDAR =========
combined = pd.concat(all_data, axis=0)
combined_file = os.path.join(output_folder, "market_data_features_labels.csv")
combined.to_csv(combined_file, index=False)
logging.info(f"✅ Dataset combinado con features + etiquetas guardado en '{combined_file}'")
