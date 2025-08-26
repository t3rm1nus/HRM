# l2_tactical/technical/multi_timeframe.py
from __future__ import annotations
import pandas as pd

def resample_and_consensus(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
    close_1m, close_5m, close_15m, close_1h
    y su consenso (media simple, fácil de cambiar).
    """
    # Asegurar DatetimeIndex
    df = df_1m.copy()
    df["close_1m"] = df["close"]  # <-- añade esta línea
    df["close_5m"]  = df["close"].resample("5T").last().reindex(df.index, method="ffill")
    df["close_15m"] = df["close"].resample("15T").last().reindex(df.index, method="ffill")
    df["close_1h"]  = df["close"].resample("1H").last().reindex(df.index, method="ffill")
    df["consensus"] = df[["close_1m", "close_5m", "close_15m", "close_1h"]].mean(axis=1)
    return df