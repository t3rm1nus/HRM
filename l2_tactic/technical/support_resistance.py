# l2_tactical/technical/support_resistance.py
from __future__ import annotations
import pandas as pd
import numpy as np

def swing_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    highs = df["high"]
    lows  = df["low"]

    # Picos máximos
    max_idx = highs.rolling(window=left + right + 1, center=True).max()
    pivots_high = (highs == max_idx)

    # Picos mínimos
    min_idx = lows.rolling(window=left + right + 1, center=True).min()
    pivots_low  = (lows == min_idx)

    # Últimos 5 niveles
    last_supports = df.loc[pivots_low, "low"].tail(5).values
    last_resist   = df.loc[pivots_high, "high"].tail(5).values

    return pd.Series({
        "support": last_supports[-1] if len(last_supports) else np.nan,
        "resistance": last_resist[-1] if len(last_resist) else np.nan,
    })