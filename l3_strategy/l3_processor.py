# l3_strategy/l3_processor.py
"""
L3 Processor - HRM
Genera un output jerárquico para L2 combinando:
- Macro conditions
- Regime detection
- Market sentiment
- Volatility forecasts
- Portfolio optimization
"""
import os
import json
import torch
from datetime import datetime
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None
from core import logging as log

# ---------------------------
# Paths
# ---------------------------
INFER_DIR = "data/datos_inferencia"
OUTPUT_FILE = os.path.join(INFER_DIR, "l3_output.json")
FEATURES_FILE = os.path.join(INFER_DIR, "regime_features_expected.json")

REGIME_MODEL_PATH = "models/L3/regime_detection_model_ensemble_optuna.pkl"
SENTIMENT_MODEL_DIR = "models/L3/sentiment/"
VOL_GARCH_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_garch.pkl"
VOL_GARCH_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_garch.pkl"
VOL_LSTM_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_lstm.h5"
VOL_LSTM_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_lstm.h5"
PORTFOLIO_COV_PATH = "models/L3/portfolio/bl_cov.csv"
PORTFOLIO_WEIGHTS_PATH = "models/L3/portfolio/bl_weights.csv"

# Volatility warmup: mínimo de cierres requeridos y valor por defecto
MIN_VOL_BARS = 10
WARMUP_VOL = 0.03

# ---------------------------
# Helpers
# ---------------------------
def save_json(data: dict, output_path: str):
    # Ruta original
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ruta de copia para charniRich
    charni_path = "C:\\proyectos\\charniRich\\envs\\grok\\train\\data\\datos_inferencia\\l3_output.json"
    os.makedirs(os.path.dirname(charni_path), exist_ok=True)
    
    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # Guardar en la ruta original
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, default=make_serializable)
    log.info(f"L3 output guardado en {output_path}")
    
    # Guardar copia en charniRich
    try:
        with open(charni_path, "w") as f:
            json.dump(data, f, indent=4, default=make_serializable)
        log.info(f"L3 output copiado a {charni_path}")
    except Exception as e:
        log.error(f"Error al copiar L3 output a charniRich: {e}")

# ---------------------------
# Load Models
# ---------------------------
def load_regime_model():
    import joblib
    log.info("Cargando modelo de Regime Detection")
    if not os.path.exists(REGIME_MODEL_PATH):
        log.critical(f"Modelo de Regime Detection faltante: {REGIME_MODEL_PATH}")
        raise FileNotFoundError(REGIME_MODEL_PATH)
    model = joblib.load(REGIME_MODEL_PATH)
    log.info("Modelo de Regime Detection cargado")
    # Log y persistencia de las features esperadas, si existen
    try:
        if hasattr(model, 'feature_names_in_'):
            names = sorted(list(getattr(model, 'feature_names_in_')))
            log.info(f"Regime features esperadas ({len(names)}):")
            # Agrupar features por tipo para mejor visualización
            return_features = sorted([f for f in names if f.startswith('return_')])
            vol_features = sorted([f for f in names if f.startswith('volatility_')])
            basic_features = sorted([f for f in names if not (f.startswith('return_') or f.startswith('volatility_'))])
            
            log.info("Basic features: " + ", ".join(basic_features))
            log.info("Return features: " + ", ".join(return_features))
            log.info("Volatility features: " + ", ".join(vol_features))
            
            os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
            with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "feature_names_in_": names,
                    "grouped_features": {
                        "basic": basic_features,
                        "returns": return_features,
                        "volatility": vol_features
                    }
                }, f, indent=2)
    except Exception as e:
        log.warning(f"No se pudieron registrar las features de Regime: {e}")
    return model

def load_sentiment_model():
    global _sentiment_tokenizer, _sentiment_model
    if '_sentiment_tokenizer' in globals() and _sentiment_tokenizer is not None and _sentiment_model is not None:
        log.info("Modelo BERT de sentimiento reutilizado de memoria")
        return _sentiment_tokenizer, _sentiment_model
    log.info("Cargando modelo BERT de Sentimiento")
    _sentiment_tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    _sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    log.info("Modelo BERT cargado")
    return _sentiment_tokenizer, _sentiment_model

def load_vol_models():
    """Carga modelos de volatilidad GARCH y LSTM con fallback seguro si TensorFlow no está disponible."""
    import joblib
    log.info("Cargando modelos de volatilidad GARCH y LSTM")
    
    # Cargar modelos GARCH (no dependen de TensorFlow)
    try:
        garch_btc = joblib.load(VOL_GARCH_PATH_BTC)
        garch_eth = joblib.load(VOL_GARCH_PATH_ETH)
    except Exception as e:
        log.error(f"Error cargando modelos GARCH: {e}")
        garch_btc = garch_eth = None
    
    # Cargar modelos LSTM solo si TensorFlow está disponible
    lstm_btc = lstm_eth = None
    if tf is not None and load_model is not None:
        try:
            lstm_btc = load_model(VOL_LSTM_PATH_BTC, compile=False)
            lstm_eth = load_model(VOL_LSTM_PATH_ETH, compile=False)
        except Exception as e:
            log.error(f"Error cargando modelos LSTM: {e}")
            lstm_btc = lstm_eth = None
    else:
        log.warning("TensorFlow no disponible, modelos LSTM no serán cargados")
    
    log.info("Modelos de volatilidad cargados")
    return garch_btc, garch_eth, lstm_btc, lstm_eth

def load_portfolio():
    log.info("Cargando optimización de cartera Black-Litterman")
    cov = pd.read_csv(PORTFOLIO_COV_PATH, index_col=0)
    weights = pd.read_csv(PORTFOLIO_WEIGHTS_PATH)
    if weights.empty:
        raise ValueError("weights vacío")
    log.info("Cartera cargada")
    return cov, weights

# ---------------------------
# Inference Functions
# ---------------------------
def predict_regime(features: pd.DataFrame, model):
    """
    Predice el régimen usando las features calculadas y el ensemble de modelos
    """
    # Obtener features requeridas
    if isinstance(model, dict) and 'features' in model:
        required_features = model['features']
    elif hasattr(model, 'feature_names_in_'):
        required_features = list(model.feature_names_in_)
    else:
        required_features = features.columns.tolist()
        
    if not required_features:
        log.error("No se pudieron determinar las features requeridas")
        return "neutral"
        
    # Validar features
    if features.empty:
        log.error("DataFrame de features vacío")
        return "neutral"
        
    # Verificar NaN
    nan_count = features[required_features].isna().sum().sum()
    if nan_count > 0:
        log.warning(f"⚠️ Hay {nan_count} valores NaN antes de la validación")
        features = features.fillna(0)
        
    # Verificar dimensiones
    if len(features) == 0:
        log.error("No hay datos en features")
        return "neutral"
    
    try:
        # Convertir a numpy con el tipo correcto
        X = features[required_features].astype(float).values
        
        # Ensemble prediction
        if isinstance(model, dict) and all(k in model for k in ['rf', 'et', 'hgb', 'label_encoder']):
            # Predicción por probabilidad promedio
            rf_prob = model['rf'].predict_proba(X)
            et_prob = model['et'].predict_proba(X)
            hgb_prob = model['hgb'].predict_proba(X)
            
            # Promedio de probabilidades
            avg_prob = (rf_prob + et_prob + hgb_prob) / 3
            pred_idx = np.argmax(avg_prob, axis=1)[0]
            
            # Convertir índice a etiqueta
            regime = model['label_encoder'].inverse_transform([pred_idx])[0]
        else:
            # Fallback para modelo simple
            regime = model.predict(X)[0]
            
        log.info(f"Regime detectado: {regime}")
        return regime
        
    except Exception as e:
        log.error(f"Error en predicción de regime: {e}")
        return "neutral"  # Valor por defecto seguro

def predict_sentiment(texts: list, tokenizer, model):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(texts or ["market"], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        num_classes = probs.shape[-1]
        if num_classes == 2:
            score = probs[:, 1] - probs[:, 0]
        elif num_classes >= 3:
            score = probs[:, 2] - probs[:, 0]
        else:
            score = probs[:, -1]
        sentiment_score = float(torch.mean(score))
    log.info(f"Score de sentimiento calculado: {sentiment_score} (device: {device})")
    return sentiment_score

def predict_vol_garch(model, returns: np.ndarray):
        """Predice volatilidad usando modelo GARCH."""
        # Validar dimensiones: GARCH espera 1D array de retornos
        arr = np.asarray(returns).flatten()
        
        # Verificar longitud mínima
        if arr.shape[0] < MIN_VOL_BARS:
            log.warning(f"GARCH: Insuficientes datos ({arr.shape[0]}), usando WARMUP_VOL={WARMUP_VOL}")
            return WARMUP_VOL
            
        try:
            # Preparar datos para GARCH
            # 1. Calcular retornos porcentuales si son precios
            if np.mean(arr) > 1:  # Probablemente son precios
                rets = np.diff(np.log(arr)) * 100
            else:  # Ya son retornos
                rets = arr * 100
                
            # 2. Remover NaN y validar
            rets = rets[~np.isnan(rets)]
            if len(rets) < MIN_VOL_BARS:
                log.warning(f"GARCH: Insuficientes retornos válidos ({len(rets)})")
                return WARMUP_VOL
                
            # 3. Forecast - Usar el modelo GARCH ya entrenado
            last_ret = rets[-250:] if len(rets) > 250 else rets
            
            # Obtener pronóstico usando el último valor
            forecast = model.forecast(reindex=False)
            vol = float(np.sqrt(forecast.variance.values[-1]))
            log.info(f"Volatilidad GARCH: {vol:.4f}")
            return vol / 100  # Convertir de porcentaje a decimal
            
        except Exception as e:
            log.critical(f"GARCH: Error en forecast: {e}. Usando WARMUP_VOL={WARMUP_VOL}")
            return WARMUP_VOL

def predict_vol_lstm(model, returns: np.ndarray):
        """Predice volatilidad usando modelo LSTM."""
        # Validar dimensiones: LSTM espera (n, window, 1)
        arr = np.asarray(returns).flatten()
        
        # Verificar longitud mínima
        if arr.shape[0] < MIN_VOL_BARS:
            log.warning(f"LSTM: Insuficientes datos ({arr.shape[0]}), usando WARMUP_VOL={WARMUP_VOL}")
            return WARMUP_VOL
            
        try:
            # Preparar datos para LSTM
            # 1. Calcular retornos porcentuales si son precios
            if np.mean(arr) > 1:  # Probablemente son precios
                rets = np.diff(np.log(arr)) * 100
            else:  # Ya son retornos
                rets = arr * 100
                
            # 2. Remover NaN y validar
            rets = rets[~np.isnan(rets)]
            if len(rets) < MIN_VOL_BARS:
                log.warning(f"LSTM: Insuficientes retornos válidos ({len(rets)})")
                return WARMUP_VOL
                
            # 3. Preparar ventana para LSTM
            window = 20  # Ventana fija de 20 períodos
            if len(rets) < window:
                log.warning(f"LSTM: Datos insuficientes para ventana de {window}")
                return WARMUP_VOL
                
            # Tomar últimos datos y formatear para LSTM
            x = rets[-window:].reshape(1, window, 1)
            
            # 4. Predicción - el modelo ya da volatilidad en porcentaje
            pred = model.predict(x, verbose=0)
            vol = float(pred[0, 0])  # Ya garantizado positivo por softplus
            # Validar rango
            vol = np.clip(vol, 0.1, 200.0)  # Limitar entre 0.1% y 200%
            log.info(f"Volatilidad LSTM: {vol:.4f}")
            return vol / 100  # Convertir de porcentaje a decimal
            
        except Exception as e:
            log.critical(f"LSTM: Error en predict: {e}. Usando WARMUP_VOL={WARMUP_VOL}")
            return WARMUP_VOL

def compute_risk_appetite(volatility_avg, sentiment_score, regime="range"):
    # 1. Ajuste por volatilidad (más volatilidad -> más conservador)
    vol_factor = np.exp(-5 * volatility_avg)  # Decae exponencialmente con volatilidad
    
    # 2. Ajuste por sentimiento (-1 a 1 -> 0 a 1)
    sent_factor = (sentiment_score + 1) / 2
    
    # 3. Ajuste por régimen
    regime_factors = {
        "bull": 1.2,
        "bear": 0.6,
        "range": 0.9,
        "volatile": 0.7
    }
    regime_factor = regime_factors.get(regime.lower().replace("_market", ""), 0.9)
    
    # Combinar factores (vol y sent entre 0-1, regime modifica el resultado)
    appetite_score = (0.6 * vol_factor + 0.4 * sent_factor) * regime_factor
    
    # Mapear score a categorías
    if appetite_score < 0.33:
        ra = "conservative"
    elif appetite_score < 0.66:
        ra = "moderate"
    else:
        ra = "aggressive"
    
    log.info(f"Risk appetite calculado: {ra} (score={appetite_score:.2f}, vol={vol_factor:.2f}, sent={sent_factor:.2f}, regime={regime_factor:.2f})")
    return ra

# ---------------------------
# Main L3 Output
# ---------------------------
def _build_regime_features(market_data: dict, model) -> pd.DataFrame:
    """Construye un DataFrame con EXACTAMENTE las columnas que el modelo espera."""
    from .regime_features import calculate_regime_features, validate_regime_features
    import pandas as pd
    import numpy as np
    
    # Columnas esperadas - desde el modelo ensemble
    if isinstance(model, dict) and 'features' in model:
        expected = model['features']
    elif hasattr(model, 'feature_names_in_'):
        expected = list(getattr(model, 'feature_names_in_'))
    else:
        # Features requeridas por el modelo (de training)
        expected = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macdsig', 'macdhist',
            'boll_upper', 'boll_middle', 'boll_lower',
            'return', 'log_return'
        ]
        # Añadir features de volatilidad y retornos
        for w in [5, 15, 30, 60, 120]:
            expected.extend([f'volatility_{w}', f'return_{w}'])
    
    # Preparar DataFrame vacío para casos de error
    empty_df = pd.DataFrame([[0.0] * len(expected)], columns=expected)
    
    # Obtener los datos primarios (BTC o ETH)
    primary = 'BTCUSDT' if 'BTCUSDT' in market_data else ('ETHUSDT' if 'ETHUSDT' in market_data else None)
    
    if not primary:
        log.error("No hay datos de BTC ni ETH disponibles para regime detection")
        return empty_df
    
    # Convertir datos a DataFrame
    data = market_data.get(primary)
    if not data:
        log.error(f"No hay datos disponibles para {primary}")
        return empty_df
    
    try:
        # Convertir a DataFrame según el tipo de entrada
        if isinstance(data, list) and data:
            dfp = pd.DataFrame(data)
        elif isinstance(data, dict):
            dfp = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            dfp = data.copy()
        else:
            log.error(f"Formato de datos inválido para {primary}: {type(data)}")
            return empty_df
            
        # Procesar timestamp si existe
        if 'timestamp' in dfp.columns:
            dfp.index = pd.to_datetime(dfp['timestamp'], unit='ms')
            dfp = dfp.drop('timestamp', axis=1)
            
        # Validación de datos
        if len(dfp) < 120:
            log.error(f"Insuficientes datos para regime detection: {len(dfp)} < 120")
            return empty_df
            
        # Validar y convertir columnas OHLCV
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in dfp.columns:
                log.error(f"Falta columna requerida: {col}")
                return empty_df
            try:
                dfp[col] = pd.to_numeric(dfp[col], errors='coerce')
                if dfp[col].isna().all():
                    log.error(f"La columna {col} no contiene datos numéricos válidos")
                    return empty_df
            except Exception as e:
                log.error(f"Error convirtiendo columna {col}: {str(e)}")
                return empty_df
                
        # Calcular features
        log.info(f"Calculando features con {len(dfp)} puntos de datos")
        features = calculate_regime_features(dfp)
        
        if features.empty:
            log.error("No se generaron features")
            return empty_df
            
        # Validar y completar features requeridas
        features = validate_regime_features(features, expected)
        
        # Tomar última fila y validar
        df = features.iloc[[-1]][expected].copy()
        
        # Validar y rellenar NaN
        if df.isna().any().any():
            log.warning("Detectados NaN en features finales, rellenando con 0")
            df = df.fillna(0)
            
        # Diagnóstico de features
        feature_stats = {
            'valid': len(df.columns[(~df.isna().any()) & (df != 0).any()]),
            'zero': len(df.columns[df.eq(0).all()]),
            'total': len(expected)
        }
        log.info(f"Features calculadas: {feature_stats['valid']} válidas, {feature_stats['zero']} en cero, de {feature_stats['total']} totales")
        
        return df
        
    except Exception as e:
        log.error(f"Error procesando features de regime detection: {str(e)}")
        return empty_df

def _safe_close_series(val) -> np.ndarray:
    """Convierte market_data[symbol] a un np.array de cierres de longitud >=1."""
    try:
        if isinstance(val, list) and val:
            arr = [float(d.get('close', 0.0)) for d in val if isinstance(d, dict)]
            return np.array(arr, dtype=float) if arr else np.array([0.0], dtype=float)
        if isinstance(val, dict):
            return np.array([float(val.get('close', 0.0))], dtype=float)
        if isinstance(val, pd.DataFrame) and 'close' in val.columns:
            return val['close'].dropna().values.astype(float)
        if isinstance(val, pd.Series):
            return val.dropna().values.astype(float)
    except Exception:
        pass
    return np.array([0.0], dtype=float)

def generate_l3_output(market_data: dict, texts_for_sentiment: list):
    log.info("Generando L3 output estratégico")

    # Carga robusta de modelos con fallbacks por componente
    try:
        regime_model = load_regime_model()
        df_features = _build_regime_features(market_data, regime_model)
        regime = predict_regime(df_features, regime_model)
    except Exception as e:
        log.critical(f"Regime detection fallback por error: {e}")
        regime = 'neutral'

    try:
        tokenizer, sentiment_model = load_sentiment_model()
        sentiment_score = predict_sentiment(texts_for_sentiment or [], tokenizer, sentiment_model)
    except Exception as e:
        log.critical(f"Sentiment fallback por error: {e}")
        sentiment_score = 0.0

    # Volatilidad (con fallback si modelos no están disponibles)
    try:
        # Cargar modelos
        garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
        
        # Obtener series de precios
        btc_series = _safe_close_series(market_data.get('BTCUSDT'))
        eth_series = _safe_close_series(market_data.get('ETHUSDT'))
        
        # Verificar longitud mínima
        if len(btc_series) < MIN_VOL_BARS or len(eth_series) < MIN_VOL_BARS:
            log.warning(f"Volatility warmup: series cortas (BTC={len(btc_series)}, ETH={len(eth_series)})")
            vol_btc = vol_eth = WARMUP_VOL
        else:
            # Calcular volatilidad BTC
            btc_vols = []
            if garch_btc is not None:
                btc_vols.append(predict_vol_garch(garch_btc, btc_series))
            if lstm_btc is not None and tf is not None:
                btc_vols.append(predict_vol_lstm(lstm_btc, btc_series))
            vol_btc = np.mean(btc_vols) if btc_vols else WARMUP_VOL
            
            # Calcular volatilidad ETH
            eth_vols = []
            if garch_eth is not None:
                eth_vols.append(predict_vol_garch(garch_eth, eth_series))
            if lstm_eth is not None and tf is not None:
                eth_vols.append(predict_vol_lstm(lstm_eth, eth_series))
            vol_eth = np.mean(eth_vols) if eth_vols else WARMUP_VOL
            
    except Exception as e:
        log.critical(f"Error en cálculo de volatilidad: {e}")
        vol_btc = vol_eth = WARMUP_VOL
    
    # Calcular promedio de volatilidad
    volatility_avg = float(np.mean([vol_btc, vol_eth]))

    # Black-Litterman / pesos
    try:
        cov_matrix, optimal_weights = load_portfolio()
        risk_appetite = compute_risk_appetite(volatility_avg, sentiment_score, regime=regime)
        first_row = optimal_weights.iloc[0]
        # Filtrar solo columnas cripto conocidas o numéricas
        allowed = {"BTC","ETH","CASH","BTCUSDT","ETHUSDT"}
        cols = [c for c in first_row.index if (c in allowed)]
        if not cols:
            # fallback: elegir solo columnas cuyo valor sea convertible a float
            cols = [c for c in first_row.index if pd.api.types.is_number(first_row[c]) or str(first_row[c]).replace('.','',1).isdigit()]
        asset_allocation = {str(col): float(first_row[col]) for col in cols}
        if not asset_allocation:
            raise ValueError("sin columnas de asignación válidas")
    except Exception as e:
        log.critical(f"Portfolio allocation fallback por error: {e}")
        asset_allocation = {"BTC": 0.5, "ETH": 0.4, "CASH": 0.1}

    risk_appetite = compute_risk_appetite(volatility_avg, sentiment_score)

    # Convertir risk_appetite a valor numérico
    risk_map = {
        "low": 0.2,
        "moderate": 0.5,
        "aggressive": 0.8,
        "high": 0.8
    }
    risk_value = risk_map.get(str(risk_appetite).lower(), 0.5)

    # Asegurar que asset_allocation tiene los campos correctos
    total_allocation = sum(asset_allocation.values())
    if abs(total_allocation - 1.0) > 0.01 or "CASH" not in asset_allocation:
        # Reajustar allocations si no suman 1 o falta CASH
        base_alloc = {
            "BTC": 0.4,
            "ETH": 0.3,
            "CASH": 0.3
        }
        if regime == "bullish":
            base_alloc = {"BTC": 0.5, "ETH": 0.3, "CASH": 0.2}
        elif regime == "bearish":
            base_alloc = {"BTC": 0.3, "ETH": 0.2, "CASH": 0.5}
        asset_allocation = base_alloc

    strategic_guidelines = {
        "regime": regime,
        "asset_allocation": asset_allocation,
        "risk_appetite": risk_value,
        "sentiment_score": float(sentiment_score),
        "volatility_forecast": {
            "BTCUSDT": float(vol_btc),
            "ETHUSDT": float(vol_eth)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    save_json(strategic_guidelines, OUTPUT_FILE)
    log.info("L3 output generado correctamente")
    return strategic_guidelines

# ---------------------------
# CLI Execution
# ---------------------------
# Bloque de ejecución CLI comentado para evitar ejecución automática al importar
# if __name__ == "__main__":
#     market_data_example = {
#         "BTCUSDT": [{"open":50000,"high":50500,"low":49900,"close":50250,"volume":1.2},
#                     {"open":50200,"high":50400,"low":50000,"close":50100,"volume":1.0},
#                     {"open":50150,"high":50300,"low":50050,"close":50200,"volume":1.5}],
#         "ETHUSDT": [{"open":3500,"high":3550,"low":3480,"close":3520,"volume":10},
#                     {"open":3520,"high":3540,"low":3490,"close":3510,"volume":12},
#                     {"open":3510,"high":3530,"low":3500,"close":3525,"volume":9}]
#     }
#     texts_example = [
#         "BTC will rally after the Fed announcement",
#         "ETH bearish sentiment in crypto news"
#     ]
#     try:
#         output = generate_l3_output(market_data_example, texts_example)
#         log.info("Ejecución de L3 finalizada con éxito", extra={"output": output})
#         print(json.dumps(output, indent=2))
#     except Exception as e:
#         log.critical(f"L3 falló: {e}")
#         raise
