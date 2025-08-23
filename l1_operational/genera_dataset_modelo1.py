#!/usr/bin/env python3
"""
Generador de datasets individuales para BTC/USDT y ETH/USDT con features (1m + 5m) usando CCXT y data/loaders.

Descarga velas históricas reales desde Binance (REST público) y construye para cada activo:
- data/<symbol>_1m.csv
- data/<symbol>_features_train.csv
- data/<symbol>_features_test.csv

Uso (desde la raíz del repo):
  python l1_operational/genera_datasets_individuales.py --days 30 --output-dir data
"""

import argparse
import math
import os
import sys
from datetime import timedelta
import ccxt
import pandas as pd

# Asegurar que el repo raíz esté en sys.path para importar data.loaders
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.loaders import (
    ensure_datetime_index,
    normalize_btc_columns,
    normalize_eth_columns,
    build_multitimeframe_features,
    temporal_train_test_split,
)


def _init_exchange() -> ccxt.binance:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })


def fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, until_ms, limit_per_call=1000) -> pd.DataFrame:
    """Descarga paginada de OHLCV en [since_ms, until_ms]."""
    all_rows = []
    fetch_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit_per_call)
        if not batch:
            break
        all_rows.extend(batch)
        fetch_since = batch[-1][0] + 1
        if fetch_since >= until_ms or len(all_rows) >= 1_000_000:
            break

    if not all_rows:
        raise RuntimeError(f"No se pudo descargar OHLCV para {symbol} (resultado vacío).")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert(None)
    return df


def process_symbol(symbol: str, exchange, timeframe: str, since_ms: int, until_ms: int,
                   target_rows: int, warmup_rows: int, output_dir: str):
    """Descarga, normaliza y construye features para un símbolo."""
    print(f"\nProcesando {symbol}...")
    df = fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, until_ms)

    # Limitar tamaño
    max_raw = target_rows + warmup_rows
    if len(df) > max_raw:
        df = df.tail(max_raw)

    base_symbol = symbol.split("/")[0].lower()

    # Seleccionar normalizador
    if base_symbol == "eth":
        normalizer = normalize_eth_columns if "normalize_eth_columns" in globals() else normalize_btc_columns
    else:
        normalizer = normalize_btc_columns

    # Normalizar y construir features
    df_norm = normalizer(ensure_datetime_index(df.reset_index().rename(columns={"index": "timestamp"})))
    features = build_multitimeframe_features(df_1m=df_norm)
    features = features.dropna().tail(target_rows)

    # Split temporal
    train, test = temporal_train_test_split(features, test_size=0.2)

    # Guardar archivos
    raw_path = os.path.join(output_dir, f"{base_symbol}_1m.csv")
    train_path = os.path.join(output_dir, f"{base_symbol}_features_train.csv")
    test_path = os.path.join(output_dir, f"{base_symbol}_features_test.csv")

    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(raw_path, index=False)
    train.to_csv(train_path)
    test.to_csv(test_path)

    print(f"Guardado {symbol}:")
    print(f"- Crudo: {raw_path} ({len(df)} filas)")
    print(f"- Train: {train_path} ({train.shape})")
    print(f"- Test: {test_path} ({test.shape})")


def main():
    parser = argparse.ArgumentParser(description="Generar datasets individuales para BTC y ETH")
    parser.add_argument("--days", type=int, default=None, help="Días históricos a descargar")
    parser.add_argument("--timeframe", type=str, default="1m", help="Timeframe base")
    parser.add_argument("--output-dir", type=str, default="data", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exchange = _init_exchange()

    TARGET_ROWS = 200_000
    WARMUP_ROWS = 2_000
    rows_per_day = 1440 if args.timeframe == "1m" else (288 if args.timeframe == "5m" else 1440)
    days = args.days if args.days is not None else math.ceil((TARGET_ROWS + WARMUP_ROWS) / rows_per_day)

    now = pd.Timestamp.now(tz="UTC")
    since = now - timedelta(days=days)

    print(f"Descargando datos desde {since} hasta {now} (timeframe {args.timeframe})")

    # Procesar ambos símbolos
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        process_symbol(symbol, exchange, args.timeframe, int(since.timestamp() * 1000),
                       int(now.timestamp() * 1000), TARGET_ROWS, WARMUP_ROWS, args.output_dir)


if __name__ == "__main__":
    main()
