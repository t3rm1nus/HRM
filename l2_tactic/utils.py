"""
Utility functions for L2 Tactic module
"""
import numpy as np
import pandas as pd


def safe_float(x):
    """
    Convierte a float el último valor de un array, lista o Serie.
    Evita el error "only length-1 arrays can be converted to Python scalars".

    Args:
        x: Value to convert to float

    Returns:
        float: Converted value or np.nan if conversion fails
    """
    if isinstance(x, (list, np.ndarray, pd.Series)):
        if len(x) == 0:
            return np.nan
        return float(x[-1])  # último valor
    try:
        return float(x)
    except Exception:
        return np.nan
