#!/usr/bin/env python3
"""
Generador de dataset BTC/USDT con features (1m + 5m) usando CCXT y data/loaders.

Descarga velas históricas reales desde Binance (REST público) y construye:
- data/btc_1m.csv            (crudo descargado)
- data/btc_features_train.csv
- data/btc_features_test.csv

Uso (desde la raíz del repo):
  python l1_operational/genera_dataset_modelo1.py --days 30 --symbol BTC/USDT --output-dir data
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

# Asegurar que el repo raíz esté en sys.path para importar data.loaders
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.loaders import (
    ensure_datetime_index,
    normalize_btc_columns,
    build_multitimeframe_features,
    temporal_train_test_split,
)


def _init_exchange() -> ccxt.binance:
    # Sin claves; REST público es suficiente para OHLCV histórico
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
    })
    return exchange


def fetch_ohlcv_paginated(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """Descarga paginada de OHLCV en [since_ms, until_ms]."""
    all_rows = []
    fetch_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit_per_call)
        if not batch:
            break
        all_rows.extend(batch)

        last_ts = batch[-1][0]
        # Avanzar siguiente ventana: +1 ms para evitar repetición
        fetch_since = int(last_ts) + 1

        # Cortar si alcanzamos el until
        if fetch_since >= until_ms:
            break

        # Protección de tamaño (aprox. 1e6 filas máx.)
        if len(all_rows) >= 1_000_000:
            break

    if not all_rows:
        raise RuntimeError("No se pudo descargar OHLCV (resultado vacío).")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert(None)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generar dataset de BTC/USDT con features 1m+5m")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Símbolo (ej. BTC/USDT o BTC/USD)")
    parser.add_argument("--days", type=int, default=None, help="Días históricos a descargar (se calcula si no se da)")
    parser.add_argument("--timeframe", type=str, default=None, help="Timeframe base para descarga (por defecto 1m)")
    parser.add_argument("--output-dir", type=str, default="data", help="Directorio de salida")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exchange = _init_exchange()

    # Objetivo fijo: 200k filas finales de features
    TARGET_ROWS = 200_000
    WARMUP_ROWS = 2_000  # margen para ventanas de indicadores
    timeframe = args.timeframe or "1m"
    rows_per_day = 1440 if timeframe == "1m" else (288 if timeframe == "5m" else 1440)
    days = args.days if args.days is not None else int(math.ceil((TARGET_ROWS + WARMUP_ROWS) / rows_per_day))
    print(f"Objetivo fijo -> target_rows={TARGET_ROWS}, days={days}, timeframe={timeframe}")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    print(f"Descargando OHLCV real de {args.symbol} {timeframe} desde {since} hasta {now}...")

    df_1m = fetch_ohlcv_paginated(
        exchange=exchange,
        symbol=args.symbol,
        timeframe=timeframe,
        since_ms=int(since.timestamp() * 1000),
        until_ms=int(now.timestamp() * 1000),
    )

    # Guardar crudo
    raw_path = os.path.join(args.output_dir, "btc_1m.csv")
    df_1m.reset_index().rename(columns={"index": "timestamp"}).to_csv(raw_path, index=False)
    print(f"Guardado crudo 1m: {raw_path} ({len(df_1m)} filas, rango: {df_1m.index.min()} -> {df_1m.index.max()})")

    # Recorte previo crudo para no exceder demasiado: mantener últimos target+warmup
    max_raw = TARGET_ROWS + WARMUP_ROWS
    if len(df_1m) > max_raw:
        df_1m = df_1m.tail(max_raw)

    # Normalizar columnas y preparar features
    df_1m_norm = normalize_btc_columns(
        ensure_datetime_index(df_1m.reset_index().rename(columns={"index": "timestamp"}))
    )
    features = build_multitimeframe_features(df_1m=df_1m_norm)
    features = features.dropna().tail(TARGET_ROWS)
    # Split temporal
    train, test = temporal_train_test_split(features, test_size=args.test_size)

    train_path = os.path.join(args.output_dir, "btc_features_train.csv")
    test_path = os.path.join(args.output_dir, "btc_features_test.csv")
    train.to_csv(train_path)
    test.to_csv(test_path)

    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Guardados: {train_path} y {test_path}")

    # Nota sobre suficiencia de datos
    total_minutes = len(df_1m)
    total_days = total_minutes / 1440.0
    print(
        "\nSuficiencia de datos:\n"
        f"- Minutos descargados: {total_minutes} (~{total_days:.1f} días).\n"
        f"- Filas finales de features (train+test): {len(features)} (objetivo {TARGET_ROWS}).\n"
        "- Nota: para deep learning necesitarás más datos (meses/años) y/o más símbolos/timeframes.\n"
    )


if __name__ == "__main__":
    main()


