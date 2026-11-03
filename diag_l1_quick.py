#!/usr/bin/env python3
"""
Diagnóstico rápido de L1Model — sin bucle de velas, sin backtest.
Ejecutar desde raíz del proyecto: python diag_l1_quick.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# ── Importar solo L1 ──
from l1_operational.models import L1Model, L1Signal

# ── Crear datos de mercado sintéticos realistas (50 velas de BTC) ──
np.random.seed(42)
n = 100
close = 95000 + np.cumsum(np.random.randn(n) * 200)
df_btc = pd.DataFrame({
    'open':   close - np.random.rand(n) * 100,
    'high':   close + np.random.rand(n) * 150,
    'low':    close - np.random.rand(n) * 150,
    'close':  close,
    'volume': np.random.rand(n) * 1e6 + 5e5,
})
df_btc.index = pd.date_range('2026-01-01', periods=n, freq='1min')

df_eth = df_btc.copy()
df_eth['close'] = df_eth['close'] / 20  # ETH ~ BTC/20

market_data = {'BTCUSDT': df_btc, 'ETHUSDT': df_eth}

# ── Instanciar modelo ──
print("\n=== Instanciando L1Model ===")
model = L1Model(config={})

# ── Llamar predict ──
print("\n=== Llamando predict() ===")
result = model.predict(market_data)

print(f"\nTipo de result: {type(result)}")
print(f"Keys de result: {list(result.keys()) if isinstance(result, dict) else 'NO ES DICT'}")

signals = result.get('signals', []) if isinstance(result, dict) else []
print(f"\nTotal señales: {len(signals)}")

for i, s in enumerate(signals):
    print(f"\n--- Señal {i} ---")
    print(f"  type(s)      : {type(s)}")
    print(f"  vars(s)      : {vars(s) if hasattr(s, '__dict__') else 'sin __dict__'}")
    # Intentar acceder a campos clave
    for attr in ['symbol', 'direction', 'confidence', 'strength', 'signal_type', 'action', 'side']:
        val = getattr(s, attr, '<<NO EXISTE>>')
        print(f"  s.{attr:12s}: {val}")

print("\n=== FIN DIAGNÓSTICO ===")