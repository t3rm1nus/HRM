# C:\proyectos\HRM\backtesting\predict_l1.py

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from datetime import datetime, timedelta

# Asegurar que se puede importar el recolector de datos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from getdata import BinanceDataCollector

# Definir la configuración para el recolector de datos (ajustar si es necesario)
# Estos valores por defecto deben coincidir con tu configuración.json
BINANCE_CONFIG = {
    'api_key': '',
    'api_secret': '',
    'testnet': True,
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    'intervals': ['1m', '5m', '15m', '1h'],
}

def load_models() -> Dict:
    """Carga los modelos .pkl usando joblib."""
    models_path = Path(__file__).parent.parent / 'models' / 'L1'
    models = {}
    try:
        models['LogisticRegression'] = joblib.load(models_path / 'modelo1_lr.pkl')
        models['RandomForest'] = joblib.load(models_path / 'modelo2_rf.pkl')
        models['LightGBM'] = joblib.load(models_path / 'modelo3_lgbm.pkl')
        print("✅ Modelos de L1 cargados con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar los modelos: {e}")
        # En un entorno real, manejarías este error de forma más robusta
    return models

def prepare_data_for_prediction(data: Dict) -> pd.DataFrame:
    """
    Prepara los datos combinados para la predicción del modelo multiasset.
    Esta función asume que los modelos esperan una columna 'is_btc' y 'is_eth'.
    """
    dfs = []
    for symbol, df in data.items():
        if df.empty:
            continue
        df = df.copy()
        # Crear la columna one-hot 'is_btc' o 'is_eth' para los modelos multiasset
        df['is_btc'] = (symbol == 'BTCUSDT').astype(int)
        df['is_eth'] = (symbol == 'ETHUSDT').astype(int)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs).sort_index()
    # Los modelos esperan features específicas. Asegúrate de que las columnas coinciden.
    # Por ahora, solo usamos las que el modelo necesita, el resto de la lógica de finrl se encargará.
    # Es crucial que aquí coloques las features que tus modelos esperan
    features = [col for col in combined_df.columns if col not in ['is_btc', 'is_eth']]
    X = combined_df[features]

    return X

async def main():
    """
    Recolecta datos, hace predicciones con los modelos L1 y guarda los resultados.
    """
    # Recolectar datos
    print("⏳ Recolectando datos históricos de Binance...")
    collector = BinanceDataCollector(BINANCE_CONFIG)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    historical_data = await collector.collect_historical_data(
        symbols=BINANCE_CONFIG['symbols'],
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intervals=BINANCE_CONFIG['intervals']
    )
    
    if not historical_data:
        print("❌ No se pudieron recolectar datos. Saliendo.")
        return

    # Cargar modelos y hacer predicciones
    models = load_models()
    if not models:
        print("❌ No se pudieron cargar los modelos. Saliendo.")
        return

    predictions = {}
    
    # Aquí es donde se hace la predicción, iterando sobre los datos recolectados
    # La lógica para preparar los datos necesita ser adaptada a lo que los modelos L1 esperan
    
    # Por simplicidad, guardaremos predicciones mock en este ejemplo.
    # La lógica real de predicción necesitará ser implementada aquí
    for model_name in models.keys():
        predictions[model_name] = {
            'accuracy': 0.85,
            'precision': 0.80,
            'f1_score': 0.82,
            'profit_contribution': 1000,
            'latency_ms': 50
        }
    
    output_path = Path(__file__).parent / 'L1_predictions.json'
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"✅ Predicciones de L1 simuladas guardadas en: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())