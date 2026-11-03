#!/usr/bin/env python3
"""
Script de diagnóstico para verificar por qué L1Model no genera señales
"""

import os
import sys
import pandas as pd
import numpy as np

# Añadir el directorio raíz al path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from l1_operational.models import L1Model, MomentumModel
from core.technical_indicators import calculate_technical_indicators

print("="*80)
print("DIAGNÓSTICO L1 MODEL")
print("="*80)

# Cargar datos de prueba
data_path = os.path.join(REPO_ROOT, 'backtesting', 'data', 'btcusdt_1m.csv')
if not os.path.exists(data_path):
    print(f"❌ No se encontró el archivo: {data_path}")
    sys.exit(1)

print(f"\n📁 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path)
print(f"✅ Datos cargados: {len(df)} filas")
print(f"   Columnas: {list(df.columns)}")

# Convertir timestamp a datetime si es necesario
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

# Usar solo las primeras 100 filas para prueba
df_test = df.head(100).copy()
print(f"\n📊 Usando primeras {len(df_test)} filas para prueba")
print(f"   Index: {df_test.index[0]} a {df_test.index[-1]}")

# Normalizar nombres de columnas
df_test.columns = [c.lower() for c in df_test.columns]
print(f"   Columnas normalizadas: {list(df_test.columns)}")

# Verificar columnas requeridas
required = ['open', 'high', 'low', 'close', 'volume']
missing = [c for c in required if c not in df_test.columns]
if missing:
    print(f"❌ Columnas faltantes: {missing}")
else:
    print(f"✅ Todas las columnas requeridas presentes")

# Probar calculate_technical_indicators
print("\n" + "="*80)
print("PROBANDO: calculate_technical_indicators")
print("="*80)

try:
    indicators_dict = calculate_technical_indicators({'BTCUSDT': df_test})
    print(f"✅ Indicadores calculados")
    print(f"   Keys: {list(indicators_dict.keys())}")
    
    if 'BTCUSDT' in indicators_dict:
        ind_df = indicators_dict['BTCUSDT']
        print(f"   Indicadores shape: {ind_df.shape}")
        print(f"   Indicadores columns: {list(ind_df.columns)}")
        print(f"\n   Primeras 3 filas de indicadores:")
        print(ind_df.head(3).to_string())
        print(f"\n   Últimas 3 filas de indicadores:")
        print(ind_df.tail(3).to_string())
    else:
        print(f"❌ 'BTCUSDT' no está en indicators_dict")
except Exception as e:
    print(f"❌ Error calculando indicadores: {e}")
    import traceback
    traceback.print_exc()

# Probar MomentumModel
print("\n" + "="*80)
print("PROBANDO: MomentumModel")
print("="*80)

momentum_model = MomentumModel()
print(f"   short_period: {momentum_model.short_period}")
print(f"   medium_period: {momentum_model.medium_period}")
print(f"   momentum_threshold: {momentum_model.momentum_threshold}")

# Validar datos
print(f"\n   Validando datos:")
print(f"   - df is None: {df_test is None}")
print(f"   - df.empty: {df_test.empty}")
print(f"   - len(df): {len(df_test)}")
print(f"   - required cols present: {all(c in df_test.columns for c in required)}")

try:
    signals = momentum_model.generate_signals({'BTCUSDT': df_test})
    print(f"\n✅ MomentumModel.generate_signals() retornó {len(signals)} señales")
    for i, signal in enumerate(signals[:5]):  # Mostrar máximo 5
        print(f"   Signal {i}: {signal.direction}, conf={signal.confidence:.3f}, type={signal.signal_type.value}")
except Exception as e:
    print(f"❌ Error en MomentumModel: {e}")
    import traceback
    traceback.print_exc()

# Probar L1Model completo
print("\n" + "="*80)
print("PROBANDO: L1Model (completo)")
print("="*80)

try:
    l1_model = L1Model()
    result = l1_model.predict({'BTCUSDT': df_test})
    
    print(f"✅ L1Model.predict() ejecutado")
    print(f"   Total signals: {len(result['signals'])}")
    print(f"   Metrics: {result['metrics']}")
    
    # Contar por dirección
    buy_count = sum(1 for s in result['signals'] if s.direction == 'buy')
    sell_count = sum(1 for s in result['signals'] if s.direction == 'sell')
    hold_count = sum(1 for s in result['signals'] if s.direction == 'hold')
    print(f"   BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
    
except Exception as e:
    print(f"❌ Error en L1Model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNÓSTICO COMPLETADO")
print("="*80)
