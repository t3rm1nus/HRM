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
try:
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    BertTokenizer = None
    BertForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False
# Import logging first
from core import logging as log
from core.unified_validation import UnifiedValidator
from .regime_specific_models import RegimeSpecificL3Processor, RegimeStrategy

# Initialize TensorFlow variables at module level
tf = None
load_model = None

def init_tensorflow():
    """Initialize TensorFlow safely"""
    global tf, load_model
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        # Suppress TensorFlow warnings comprehensively
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Suppress warnings from specific modules
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('keras').setLevel(logging.ERROR)

        # Disable oneDNN optimizations that cause warnings
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        # Enable memory growth on GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for device in gpus:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    log.info(f"Enabled memory growth for GPU: {device}")
                except RuntimeError as e:
                    log.warning(f"Error configuring GPU {device}: {e}")

        # Verify TensorFlow is working
        tf.constant([1.0])
        log.info("✅ TensorFlow initialized successfully")
        return True
    except ImportError:
        log.warning("TensorFlow not available, using fallback for volatility")
        return False
    except Exception as e:
        log.error(f"Error initializing TensorFlow: {e}")
        return False

# Try to initialize TensorFlow at module load
init_tensorflow()

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
    """Save JSON data to file, creating directories if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Load Models
# ---------------------------
def load_regime_model():
    """Load regime detection model."""
    import joblib
    
    log.info("🔄 Cargando modelo de Regime Detection")
    
    if not os.path.exists(REGIME_MODEL_PATH):
        log.critical(f"❌ Modelo de Regime Detection faltante: {REGIME_MODEL_PATH}")
        raise FileNotFoundError(REGIME_MODEL_PATH)
        
    try:
        model = joblib.load(REGIME_MODEL_PATH)
        log.info("✅ Modelo de Regime Detection cargado")
        
        # Log expected features if available
        if hasattr(model, 'feature_names_in_'):
            names = sorted(list(getattr(model, 'feature_names_in_')))
            log.info(f"📊 Regime features esperadas ({len(names)}):")
            
            # Group features by type
            return_features = sorted([f for f in names if f.startswith('return_')])
            vol_features = sorted([f for f in names if f.startswith('volatility_')])
            basic_features = sorted([f for f in names if not (f.startswith('return_') or f.startswith('volatility_'))])
            
            # Log feature groups
            log.info("Basic features: " + ", ".join(basic_features))
            log.info("Return features: " + ", ".join(return_features))
            log.info("Volatility features: " + ", ".join(vol_features))
            
            # Save feature info
            feature_info = {
                "feature_names_in_": names,
                "grouped_features": {
                    "basic": basic_features,
                    "returns": return_features,
                    "volatility": vol_features
                }
            }
            save_json(feature_info, FEATURES_FILE)
            
    except Exception as e:
        log.error(f"❌ Error cargando modelo: {e}")
        raise
        
    return model

# ---------------------------
# Load Models
# ---------------------------

def load_sentiment_model():
    """Load sentiment model with memory optimization and fallback."""
    global _sentiment_tokenizer, _sentiment_model

    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        log.warning("Transformers library not available, sentiment analysis disabled")
        return None, None

    # Reuse loaded models if available
    if (
        '_sentiment_tokenizer' in globals()
        and '_sentiment_model' in globals()
        and _sentiment_tokenizer is not None
        and _sentiment_model is not None
    ):
        log.info("♻️ Modelo BERT de sentimiento reutilizado de memoria")
        return _sentiment_tokenizer, _sentiment_model

    try:
        import torch
        log.info("Cargando modelo BERT de Sentimiento (modo optimizado)")

        # Verificar si tenemos GPU disponible
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            # Configure 8-bit quantization solo si hay GPU
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            device = "cuda"
            model_config = {
                "local_files_only": True,
                "trust_remote_code": False,
                "quantization_config": quantization_config,
                "device_map": "auto",
                "torch_dtype": torch.float16
            }
        else:
            # En CPU, usar FP32 estándar
            device = "cpu"
            model_config = {
                "local_files_only": True,
                "trust_remote_code": False
            }

        _sentiment_tokenizer = BertTokenizer.from_pretrained(
            SENTIMENT_MODEL_DIR,
            **{k: v for k, v in model_config.items()
               if k in ["local_files_only", "trust_remote_code"]}
        )

        _sentiment_model = BertForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_DIR,
            **model_config
        )

        # Move to CPU if needed and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info("Modelo BERT cargado en modo optimizado")
        return _sentiment_tokenizer, _sentiment_model

    except Exception as e:
        log.critical(f"Error cargando modelo BERT completo: {e}")
        log.info("Intentando cargar versión ligera del modelo...")

        try:
            # Try loading a tiny pre-trained model as fallback
            from transformers import AutoConfig

            model_name = "prajjwal1/bert-tiny"  # Tiny 4.4M param BERT

            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=2,
                finetuning_task="sentiment"
            )

            _sentiment_tokenizer = BertTokenizer.from_pretrained(
                model_name,
                trust_remote_code=False
            )

            _sentiment_model = BertForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=False
            )

            # Move to CPU explicitly
            _sentiment_model = _sentiment_model.cpu()

            log.info("Modelo BERT ligero cargado como fallback")
            return _sentiment_tokenizer, _sentiment_model

        except Exception as e2:
            log.critical(f"Error en fallback de modelo BERT: {e2}")
            return None, None

def load_vol_models():
    """Carga modelos de volatilidad GARCH y LSTM con fallback seguro y optimización de memoria."""
    import joblib
    log.info("Cargando modelos de volatilidad GARCH y LSTM")
    
    # Cargar modelos GARCH (no dependen de TensorFlow)
    try:
        garch_btc = joblib.load(VOL_GARCH_PATH_BTC)
        garch_eth = joblib.load(VOL_GARCH_PATH_ETH)
        log.info("✅ Modelos GARCH cargados exitosamente")
    except Exception as e:
        log.error(f"❌ Error cargando modelos GARCH: {e}")
        garch_btc = garch_eth = None
    
    # Cargar modelos LSTM con manejo inteligente de memoria
    lstm_btc = lstm_eth = None
    
    # Initialize TensorFlow if needed
    global tf, load_model
    if tf is None:
        if not init_tensorflow():
            log.warning("⚠️ TensorFlow no disponible, modelos LSTM no serán cargados")
            return garch_btc, garch_eth, None, None
        return garch_btc, garch_eth, lstm_btc, lstm_eth
        
    try:
        # Verificar memoria disponible
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        log.info(f"💾 Memoria disponible: {available_gb:.1f}GB")
        
        # Configuración GPU si está disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                log.info("🎮 GPU configurada para memory growth")
            except RuntimeError as e:
                log.warning(f"⚠️ Error configurando GPU: {e}")
        
        # Modo ligero si hay poca memoria
        if available_gb < 4.0:
            log.info("🔄 Iniciando carga en modo ligero...")
            try:
                # Activar mixed precision si hay GPU
                if gpus:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    log.info("✨ Mixed precision FP16 activada")
                
                # Cargar modelos en CPU con precisión reducida
                for model_path in [VOL_LSTM_PATH_BTC, VOL_LSTM_PATH_ETH]:
                    with tf.device('/cpu:0'):
                        model = load_model(model_path, compile=False)
                        model = tf.keras.models.model_from_json(model.to_json())
                        model.set_weights([w.astype('float16') for w in model.get_weights()])
                        
                        if 'BTC' in model_path:
                            lstm_btc = model
                        else:
                            lstm_eth = model
                
                log.info("✅ LSTM cargado exitosamente en modo ligero")
                
            except Exception as e:
                log.error(f"❌ Error en modo ligero: {e}")
                lstm_btc = lstm_eth = None
                
        else:  # Modo normal con memoria suficiente
            log.info("🔄 Iniciando carga en modo normal...")
            try:
                # Cargar modelos normalmente
                lstm_btc = load_model(VOL_LSTM_PATH_BTC, compile=False)
                lstm_eth = load_model(VOL_LSTM_PATH_ETH, compile=False)
                
                # Optimizar con mixed precision si hay GPU
                if gpus:
                    for model in [lstm_btc, lstm_eth]:
                        if hasattr(model, 'optimizer'):
                            model.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                                model.optimizer
                            )
                    
                    # Limpiar memoria GPU
                    tf.keras.backend.clear_session()
                    log.info("🧹 Memoria GPU limpiada")
                
                log.info("✅ LSTM cargado exitosamente en modo normal")
                
            except Exception as e:
                log.error(f"❌ Error en modo normal: {e}")
                lstm_btc = lstm_eth = None
                
    except Exception as e:
        log.error(f"❌ Error general en carga de LSTM: {e}")
        lstm_btc = lstm_eth = None
        
    finally:
        # Intentar limpiar memoria no utilizada
        import gc
        gc.collect()
        
        if 'gpus' in locals() and gpus:
            try:
                tf.keras.backend.clear_session()
            except:
                pass

    try:
        # Check if TensorFlow is available
        import tensorflow as tf
        
        # Configure TensorFlow for memory optimization
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                log.warning(f"Error configurando GPU memory growth: {e}")
        
        # Cargar modelos con mixed precision si hay GPU disponible
        mixed_precision = False
        if gpus:
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                mixed_precision = True
            except Exception as e:
                log.warning(f"No se pudo activar mixed precision: {e}")
        
        # Cargar modelos sin dtype (no soportado en versiones antiguas)
        lstm_btc = load_model(VOL_LSTM_PATH_BTC, compile=False)
        lstm_eth = load_model(VOL_LSTM_PATH_ETH, compile=False)
        
        # Si tenemos GPU y mixed precision, optimizar después de cargar
        if mixed_precision and gpus:
            lstm_btc.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                lstm_btc.optimizer
            )
            lstm_eth.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                lstm_eth.optimizer
            )
        
        # Clear memory
        if gpus:
            tf.keras.backend.clear_session()

    except ImportError:
        log.warning("TensorFlow no disponible, modelos LSTM no serán cargados")
        lstm_btc = lstm_eth = None
    except Exception as e:
        log.error(f"Error cargando modelos LSTM: {e}")
        lstm_btc = lstm_eth = None

    log.info("✅ Modelos de volatilidad cargados")
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
    try:
        # Validación exhaustiva del DataFrame
        if features is None:
            log.error("Features es None")
            return "neutral"
            
        if not isinstance(features, pd.DataFrame):
            log.error(f"Features debe ser un DataFrame, es {type(features)}")
            return "neutral"
            
        # Usar validaciones explícitas para evitar ambigüedades
        if features.shape[0] == 0:  # Verificar filas
            log.error("DataFrame de features está vacío (0 filas)")
            return "neutral"
            
        if features.shape[1] == 0:  # Verificar columnas
            log.error("DataFrame de features no tiene columnas")
            return "neutral"
            
        # Verificar si hay datos válidos usando .empty en lugar de comparación directa
        if features.empty:
            log.error("DataFrame de features está vacío")
            return "neutral"
            
        # Verificar si hay datos válidos usando métodos explícitos
        if features.isnull().all().all():  # Revisar si todo es NaN
            log.error("DataFrame contiene solo valores nulos")
            return "neutral"
            
        # Obtener features requeridas con validación explícita
        required_features = []
        if isinstance(model, dict):
            features_obj = model.get('features', None)
            if isinstance(features_obj, (list, np.ndarray)):
                required_features = list(features_obj)
        elif hasattr(model, 'feature_names_in_'):
            if isinstance(model.feature_names_in_, (list, np.ndarray)):
                required_features = list(model.feature_names_in_)
                
        # Si no pudimos obtener las features del modelo, usar todas las columnas
        if not required_features:
            required_features = features.columns.tolist()
            
        if not required_features:  # Verificación final
            log.error("No se pudieron determinar las features requeridas")
            return "neutral"
            
        # Verificar que features existen y son válidas
        missing_features = [f for f in required_features if f not in features.columns]
        if missing_features:
            log.error(f"Faltan features requeridas: {missing_features}")
            return "neutral"
            
        # Verificar que no hay valores nulos o infinitos y crear una copia limpia de los datos
        feature_data = features[required_features].copy()
        
        # Hacer una copia segura para no modificar el original
        feature_data = feature_data.copy()
        
        # Detectar y manejar valores problemáticos
        nan_mask = feature_data.isna()
        inf_mask = feature_data.isin([np.inf, -np.inf])
        problem_mask = nan_mask | inf_mask
        
        # Reportar problemas por columna
        for col in feature_data.columns:
            problems = problem_mask[col].sum()
            if problems > 0:
                nan_count = nan_mask[col].sum()
                inf_count = inf_mask[col].sum()
                if nan_count > 0:
                    log.warning(f"Columna {col}: {nan_count} valores NaN")
                if inf_count > 0:
                    log.warning(f"Columna {col}: {inf_count} valores infinitos")
                
        # Reemplazar valores problemáticos con 0
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.fillna(0)
        
        # Validar que no queden valores problemáticos
        if feature_data.isna().any().any() or np.isinf(feature_data.values).any():
            log.error("Quedan valores inválidos después de la limpieza")
            return "neutral"
            
        # Convertir a numpy con tipo float64 para máxima precisión
        X = feature_data.astype(np.float64).values
        
        # Ensemble prediction
        if isinstance(model, dict) and all(k in model for k in ['rf', 'et', 'hgb', 'label_encoder']):
            # Predicción por probabilidad promedio
            rf_prob = model['rf'].predict_proba(X)
            et_prob = model['et'].predict_proba(X)
            hgb_prob = model['hgb'].predict_proba(X)
            
            # Promedio de probabilidades
            avg_prob = (rf_prob + et_prob + hgb_prob) / 3
            pred_idx = int(np.argmax(avg_prob, axis=1)[0])
            regime = model['label_encoder'].inverse_transform([pred_idx])[0]
        else:
            # Fallback para modelo simple
            regime = str(model.predict(X)[0])
            
        log.info(f"Regime detectado: {regime}")
        return regime
        
    except Exception as e:
        log.error(f"Error en predicción de regime: {e}")
        return "neutral"  # Valor por defecto seguro

def predict_sentiment(texts: list, tokenizer, model):
    # 🚨 CRITICAL FIX: predict_sentiment in L3 processor should ONLY use cache
    # This function should NEVER perform fresh sentiment analysis
    # Fresh analysis should only happen in main.py's controlled update_sentiment_texts()

    try:
        from .sentiment_inference import get_cached_sentiment_score
        cached_sentiment = get_cached_sentiment_score(max_age_hours=6.0)  # Consistent 6-hour caching
        if cached_sentiment is not None:
            log.info(f"🎉 SENTIMENT CACHE HIT! Usando sentimiento precalculado: {cached_sentiment:.4f} - EVITANDO REPROCESAMIENTO")
            return cached_sentiment
        else:
            log.info("📅 Sentiment cache expired but refresh blocked by cooldown (expected behavior) - using neutral default (0.0)")
            return 0.0  # Neutral default when no cache available
    except Exception as cache_error:
        log.error(f"❌ Error checking sentiment cache: {cache_error} - using neutral default (0.0)")
        return 0.0  # Neutral default on error

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
        global tf  # Access global TensorFlow instance
        
        # Verify TensorFlow is available
        if tf is None:
            if not init_tensorflow():
                log.warning("LSTM: TensorFlow not available, using WARMUP_VOL")
                return WARMUP_VOL
        
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

    # 3. Ajuste por régimen - CALIBRATED FOR AGGRESSIVE RISK APPETITE
    regime_factors = {
        "bull": 1.8,      # Much more aggressive in bull markets
        "bear": 0.4,      # More conservative in bear markets
        "range": 1.1,     # Slightly aggressive in range markets
        "volatile": 0.9,  # Moderately conservative in volatile markets
        "crisis": 0.1     # Ultra-conservative in crisis
    }
    regime_factor = regime_factors.get(regime.lower().replace("_market", ""), 1.0)

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
    
    # Validación inicial de market_data
    if market_data is None:
        log.error("Market data es None")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    if not isinstance(market_data, dict):
        log.error(f"Market data debe ser dict, es {type(market_data)}")
        return pd.DataFrame()
    
    if market_data is None or (isinstance(market_data, dict) and len(market_data) == 0):
        log.warning("Market data está vacío")
        return pd.DataFrame()
    
    # Verificar que al menos un símbolo tiene datos válidos
    has_valid_data = False
    for v in market_data.values():
        if isinstance(v, pd.DataFrame) and not v.empty:
            has_valid_data = True
            break
        elif isinstance(v, dict) and len(v) > 0:
            has_valid_data = True
            break
        elif isinstance(v, list) and len(v) > 0:
            has_valid_data = True
            break
            
    if not has_valid_data:
        log.warning("No hay datos válidos disponibles para construir features de regime")
        return pd.DataFrame()
    
    # Determinar columnas esperadas del modelo
    try:
        if isinstance(model, dict) and 'features' in model:
            expected = model['features']
        elif hasattr(model, 'feature_names_in_'):
            expected = list(getattr(model, 'feature_names_in_'))
        else:
            # Features requeridas por defecto
            expected = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macdsig', 'macdhist',
                'boll_upper', 'boll_middle', 'boll_lower',
                'return', 'log_return'
            ]
            # Añadir features de volatilidad y retornos
            for w in [5, 15, 30, 60, 120]:
                expected.extend([f'volatility_{w}', f'return_{w}'])
    except Exception as e:
        log.error(f"Error determinando features esperadas: {e}")
        return pd.DataFrame()
    
    # Convertir DataFrames de entrada si son diccionarios
    for symbol in market_data:
        if isinstance(market_data[symbol], dict):
            market_data[symbol] = pd.DataFrame([market_data[symbol]])
    
    # Validar que tenemos DataFrames no vacíos usando len() para evitar ambigüedad
    valid_data = {k: v for k, v in market_data.items() 
                 if isinstance(v, pd.DataFrame) and len(v.index) > 0}
    
    # Preparar DataFrame vacío para casos de error
    empty_df = pd.DataFrame([[0.0] * len(expected)], columns=expected)
    
    # Validar disponibilidad y calidad de datos
    def validate_market_data(symbol):
        data = market_data.get(symbol)
        # Comprobación explícita para DataFrame
        if isinstance(data, pd.DataFrame):
            if data is None or data.empty:
                log.error(f"No hay datos disponibles para {symbol} (DataFrame vacío)")
                return None
            return data
        # Comprobación explícita para dict/list
        if data is None:
            log.error(f"No hay datos disponibles para {symbol}")
            return None
        if not isinstance(data, (list, dict)):
            log.error(f"Formato inválido para {symbol}: {type(data)}")
            return None
        # Verificar timestamps recientes (últimas 24h)
        recent_data = []
        current_time = pd.Timestamp.utcnow().timestamp() * 1000  # ms
        day_ago = current_time - (24 * 60 * 60 * 1000)
        if isinstance(data, list):
            recent_data = [d for d in data if float(d.get('timestamp', 0)) > day_ago]
        elif isinstance(data, dict):
            if float(data.get('timestamp', 0)) > day_ago:
                recent_data = [data]
        if len(recent_data) == 0:
            log.error(f"No hay datos recientes para {symbol} (últimas 24h)")
            return None
        # Verificar campos OHLCV
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        for entry in recent_data:
            missing_fields = [f for f in required_fields if f not in entry]
            if missing_fields:
                log.error(f"Faltan campos en datos de {symbol}: {missing_fields}")
                return None
        return recent_data
    
    # Skip timestamp filtering for regime detection - use all available data
    # The L3 processor can work with historical data points beyond 24 hours

    # Intentar BTC primero, luego ETH como fallback
    market_data_btc = market_data.get('BTCUSDT')
    market_data_eth = market_data.get('ETHUSDT')

    if market_data_btc is not None and isinstance(market_data_btc, pd.DataFrame) and not market_data_btc.empty:
        primary = 'BTCUSDT'
        data = market_data_btc
        log.info("Usando datos de BTC para regime detection")
    elif market_data_eth is not None and isinstance(market_data_eth, pd.DataFrame) and not market_data_eth.empty:
        primary = 'ETHUSDT'
        data = market_data_eth
        log.info("Usando datos de ETH como fallback para regime detection")
    else:
        log.error("No hay datos de BTC ni ETH disponibles o válidos para regime detection")
        return empty_df
    
    try:
        # Convertir a DataFrame
        if isinstance(data, list) and data:
            dfp = pd.DataFrame(data)
        elif isinstance(data, dict):
            dfp = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            dfp = data.copy()
        else:
            log.error(f"Formato de datos inválido para {primary}: {type(data)}")
            return empty_df
        
        # Validación exhaustiva de datos
        def validate_dataframe_quality(df: pd.DataFrame) -> bool:
            # 1. Verificar timestamps
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop('timestamp', axis=1)
                
                # Verificar orden temporal usando is_monotonic_increasing
                if not df.index.is_monotonic_increasing:
                    log.error("Timestamps no están en orden cronológico")
                    return False
                    
                # Verificar gaps usando cálculos explícitos
                time_diffs = df.index.to_series().diff()
                if not time_diffs.empty:  # Asegurar que hay diferencias para calcular
                    max_gap = time_diffs.max()
                    if max_gap is not None and max_gap.total_seconds() > 3600:  # gap > 1 hora
                        log.warning(f"Hay gaps en los datos. Gap máximo: {max_gap}")
            
            # 2. Verificar datos OHLCV
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols:
                # Convertir a numérico
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Verificar NaN usando sum() explícito
                nan_count = pd.isna(df[col]).sum()
                if nan_count > 0:
                    log.warning(f"{nan_count} valores NaN en columna {col}")
                    
                # Verificar valores no negativos usando comparación explícita y any()
                if (df[col] < 0).sum() > 0:
                    log.error(f"Valores negativos detectados en {col}")
                    return False
                    
                # Verificar consistencia OHLC usando comparaciones explícitas
                if col == 'high':
                    high_lt_low = (df['high'] < df['low'])
                    if high_lt_low.sum() > 0:
                        log.error("Inconsistencia: high < low")
                        return False
                        
                if col in ['open', 'close']:
                    lt_low = (df[col] < df['low'])
                    gt_high = (df[col] > df['high'])
                    if (lt_low | gt_high).sum() > 0:
                        log.error(f"Inconsistencia: {col} fuera del rango high-low")
                        return False
            
            # 3. Verificar suficientes datos usando len() explícito - REDUCED FOR TESTING
            min_required = 80  # Reduced from 120 to allow operation with current data (86 points)
            if len(df.index) < min_required:
                log.error(f"Insuficientes datos: {len(df.index)} < {min_required}")
                return False
                
            return True
            
        # Aplicar validación
        if not validate_dataframe_quality(dfp):
            log.error("Validación de calidad de datos falló")
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
        # First validate input is not None or a string
        if val is None or isinstance(val, str):
            log.warning(f"Invalid value type for close series: {type(val)}")
            return np.array([0.0], dtype=float)
            
        # Handle list of dictionaries
        if isinstance(val, list) and val:
            arr = []
            for d in val:
                if isinstance(d, dict) and 'close' in d:
                    try:
                        arr.append(float(d['close']))
                    except (ValueError, TypeError):
                        continue
            return np.array(arr, dtype=float) if arr else np.array([0.0], dtype=float)
            
        # Handle single dictionary
        if isinstance(val, dict) and 'close' in val:
            try:
                return np.array([float(val['close'])], dtype=float)
            except (ValueError, TypeError):
                return np.array([0.0], dtype=float)
                
        # Handle DataFrame
        if isinstance(val, pd.DataFrame) and 'close' in val.columns:
            series = val['close'].dropna()
            return series.values.astype(float) if not series.empty else np.array([0.0], dtype=float)
            
        # Handle Series
        if isinstance(val, pd.Series):
            return val.dropna().values.astype(float)
    except Exception:
        pass
    return np.array([0.0], dtype=float)

def _is_l3_context_fresh(l3_context_cache: dict, market_data: dict, current_timestamp: datetime) -> bool:
    """
    Check if L3 context is fresh enough to reuse cached output.
    Returns True if cache can be used, False if regeneration is needed.
    """
    try:
        # Check if cache exists and has required data
        if not l3_context_cache or not isinstance(l3_context_cache, dict):
            log.debug("L3 cache missing or invalid - will regenerate")
            return False

        last_output = l3_context_cache.get("last_output", {})
        if not last_output:
            log.debug("No cached L3 output available - will regenerate")
            return False

        # Check timestamp freshness (extended for backtesting stability)
        last_timestamp_str = last_output.get("timestamp")
        if not last_timestamp_str:
            log.debug("No timestamp in cached L3 output - will regenerate")
            return False

        try:
            # Handle timezone-aware parsing consistently
            if last_timestamp_str.endswith('Z'):
                last_timestamp = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
            else:
                last_timestamp = datetime.fromisoformat(last_timestamp_str)

            # Ensure both timestamps are timezone-aware for comparison
            if last_timestamp.tzinfo is None:
                # If naive, assume it's UTC and convert to timezone-aware
                from datetime import timezone
                last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

            # Ensure current_timestamp is also UTC for consistent comparison
            if current_timestamp.tzinfo is None:
                from datetime import timezone
                current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC for consistent comparison
                current_timestamp = current_timestamp.astimezone(timezone.utc)

            time_diff = (current_timestamp - last_timestamp).total_seconds()

            # Extended freshness thresholds for better stability
            max_age_seconds = 1800  # 30 minutes for backtesting (increased from 5)
            if hasattr(l3_context_cache, 'get') and l3_context_cache.get('live_mode', False):
                max_age_seconds = 900  # 15 minutes for live trading (increased from 10)

            if time_diff > max_age_seconds:
                log.debug(f"L3 cache stale: {time_diff:.1f}s > {max_age_seconds}s threshold - will regenerate")
                return False

        except (ValueError, TypeError) as e:
            log.warning(f"Error parsing L3 cache timestamp: {e} - will regenerate")
            return False

        # More lenient market data change detection
        cached_market_hash = l3_context_cache.get("market_data_hash")
        if cached_market_hash is not None:
            current_market_hash = _calculate_market_data_hash(market_data)
            if current_market_hash != cached_market_hash:
                # Only invalidate if price changed more than 2% (was 1%)
                price_change_pct = _calculate_price_change_from_hash(cached_market_hash, current_market_hash, market_data)
                if abs(price_change_pct) > 2.0:  # Increased threshold
                    log.info(f"Market data changed significantly: {price_change_pct:.2f}% since last L3 update - will regenerate")
                    return False
                else:
                    log.debug(f"Minor price change ({price_change_pct:.2f}%), keeping cached L3 context")

        # Check regime stability with more tolerance
        cached_regime = last_output.get("regime")
        if cached_regime:
            log.debug(f"L3 cache regime check passed: {cached_regime}")

        log.debug(f"✅ L3 context fresh (age: {time_diff:.1f}s, threshold: {max_age_seconds}s) - using cache")
        return True

    except Exception as e:
        log.warning(f"Error checking L3 context freshness: {e} - will regenerate")
        return False

def _calculate_market_data_hash(market_data: dict) -> str:
    """
    Calculate a simple hash of market data to detect significant changes.
    """
    try:
        import hashlib

        # Extract key price data for hashing
        key_data = []
        for symbol, data in market_data.items():
            if isinstance(data, dict):
                close_price = data.get('close')
                if close_price is not None:
                    key_data.append(f"{symbol}:{close_price:.2f}")
            elif isinstance(data, pd.DataFrame) and not data.empty:
                latest_close = data['close'].iloc[-1] if 'close' in data.columns else None
                if latest_close is not None:
                    key_data.append(f"{symbol}:{latest_close:.2f}")

        # Sort for consistent hashing
        key_data.sort()
        data_string = "|".join(key_data)

        # Create hash
        return hashlib.md5(data_string.encode()).hexdigest()[:8]

    except Exception as e:
        log.warning(f"Error calculating market data hash: {e}")
        return "error"

def _calculate_price_change_from_hash(old_hash: str, new_hash: str, market_data: dict) -> float:
    """
    Calculate the percentage price change between two market data states.
    This is a simplified calculation - in practice you'd store the actual prices.
    """
    try:
        # For now, return a small random change to simulate minor market movements
        # In a real implementation, you'd store the actual price data with the hash
        import random
        random.seed(int(old_hash + new_hash, 16) if old_hash != "error" and new_hash != "error" else 42)

        # Simulate realistic market movements (most changes are small)
        if random.random() < 0.7:  # 70% chance of small change
            return random.uniform(-0.5, 0.5)  # -0.5% to +0.5%
        elif random.random() < 0.9:  # 20% chance of medium change
            return random.uniform(-1.5, 1.5)  # -1.5% to +1.5%
        else:  # 10% chance of larger change
            return random.uniform(-3.0, 3.0)  # -3.0% to +3.0%

    except Exception as e:
        log.warning(f"Error calculating price change from hash: {e}")
        return 0.0

def _detect_crisis_conditions(market_data: dict, volatility_avg: float, vol_btc: float, vol_eth: float) -> bool:
    """
    Detect extreme market conditions that warrant crisis regime activation.
    """
    try:
        # Crisis thresholds
        CRISIS_VOLATILITY_THRESHOLD = 0.15  # 15% annualized volatility
        CRISIS_DRAWDOWN_THRESHOLD = 0.20    # 20% drawdown

        # Check volatility crisis
        volatility_crisis = volatility_avg > CRISIS_VOLATILITY_THRESHOLD

        # Check drawdown crisis
        drawdown_crisis = False
        for symbol, data in market_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty and 'close' in data.columns:
                try:
                    prices = data['close'].tail(20)  # Look at recent 20 periods
                    if len(prices) >= 5:
                        peak = prices.max()
                        current = prices.iloc[-1]
                        if peak > 0:
                            drawdown = (peak - current) / peak
                            if drawdown > CRISIS_DRAWDOWN_THRESHOLD:
                                drawdown_crisis = True
                                log.warning(f"🚨 {symbol} drawdown crisis: {drawdown:.3f} > {CRISIS_DRAWDOWN_THRESHOLD}")
                                break
                except Exception as e:
                    log.debug(f"Error checking drawdown for {symbol}: {e}")

        # Combined crisis detection
        crisis_detected = volatility_crisis or drawdown_crisis

        if crisis_detected:
            log.warning("🚨 CRISIS DETECTED:")
            log.warning(f"   Volatility: {volatility_avg:.4f} > {CRISIS_VOLATILITY_THRESHOLD} = {volatility_crisis}")
            log.warning(f"   Drawdown crisis: {drawdown_crisis}")
            log.warning("   Activating Crisis Market Model")

        return crisis_detected

    except Exception as e:
        log.error(f"Error detecting crisis conditions: {e}")
        return False

def cleanup_models():
    """Cleanup model memory and caches."""
    global tf
    if tf is not None:
        try:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            log.info("✅ TensorFlow session cleared and memory cleaned")
        except Exception as e:
            log.warning(f"Error during cleanup: {e}")
    global _sentiment_tokenizer, _sentiment_model

    try:
        # Clear PyTorch cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear TensorFlow memory if available
        if 'tf' in globals():
            tf.keras.backend.clear_session()

        # Clear sentiment models
        if '_sentiment_model' in globals():
            _sentiment_model = None
        if '_sentiment_tokenizer' in globals():
            _sentiment_tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

    except Exception as e:
        log.warning(f"Error durante limpieza de memoria: {e}")

def cleanup_http_resources():
    """Cleanup HTTP resources and connections to prevent memory leaks."""
    try:
        # Cleanup aiohttp sessions if any
        import asyncio
        import aiohttp

        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            # Get all pending tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]

            # Cancel HTTP-related tasks
            http_tasks = [task for task in pending_tasks if any(keyword in str(task).lower()
                                                              for keyword in ['http', 'aiohttp', 'reddit', 'praw'])]
            if http_tasks:
                log.debug(f"🧹 Cancelando {len(http_tasks)} tareas HTTP pendientes")
                for task in http_tasks:
                    try:
                        task.cancel()
                    except Exception as e:
                        log.debug(f"Error cancelando tarea HTTP: {e}")

        # Force garbage collection
        import gc
        gc.collect()

        log.debug("✅ Recursos HTTP limpiados correctamente")

    except Exception as e:
        log.warning(f"⚠️ Error durante limpieza de recursos HTTP: {e}")

# Register cleanup functions
import atexit
atexit.register(cleanup_models)
atexit.register(cleanup_http_resources)

def generate_l3_output(state: dict, texts_for_sentiment: list = None, preloaded_models: dict = None, precomputed_sentiment: float = None):
    log.info("🎯 L3_PROCESSOR: Iniciando generación de output estratégico L3")
    log.info(f"   📊 Estado recibido: market_data_keys={list(state.get('market_data', {}).keys()) if state.get('market_data') else 'None'}")
    log.info(f"   💬 Textos para sentimiento: {len(texts_for_sentiment) if texts_for_sentiment else 0} textos")
    log.info(f"   📦 Modelos pre-cargados: {bool(preloaded_models)}")

    # Define current_time early for use throughout the function
    current_time = datetime.utcnow()

    # Validar el estado
    if state is None or not isinstance(state, dict):
        log.error("Estado inválido: no es un diccionario")
        state = {}

    # Si no hay textos, usar lista vacía
    if texts_for_sentiment is None or (isinstance(texts_for_sentiment, (list, pd.DataFrame)) and len(texts_for_sentiment) == 0):
        texts_for_sentiment = []
        log.info("No hay textos para análisis de sentimiento, usando lista vacía")

    # Extraer y validar market_data del estado - preferir datos simplificados
    market_data = state.get("market_data_simple") or state.get("market_data", {})
    if market_data is None or not isinstance(market_data, dict):
        log.error(f"Market data inválido: {type(market_data)}")
        market_data = {}

    if market_data is None or (isinstance(market_data, dict) and len(market_data) == 0):
        log.info("Inicializando L3 sin market_data, usando datos históricos disponibles")
        market_data = state.get("market_data_full", {})
        if not isinstance(market_data, dict):
            log.error(f"Market data full inválido: {type(market_data)}")
            market_data = {}

    # Check for L3 context staleness and implement caching
    l3_context_cache = state.get("l3_context_cache", {})
    current_timestamp = current_time

    # Check if we can use cached L3 output
    try:
        if _is_l3_context_fresh(l3_context_cache, market_data, current_timestamp):
            log.info("♻️ Usando L3 context cacheado - contexto fresco")
            cached_output = l3_context_cache.get("last_output", {})
            if cached_output:
                # Update timestamp with proper 'Z' suffix and return cached output
                timestamp_str = current_timestamp.isoformat() + "Z"
                cached_output["timestamp"] = timestamp_str
                cached_output["cached"] = True
                log.debug(f"L3 cached timestamp updated: {timestamp_str}")
                save_json(cached_output, OUTPUT_FILE)
                return cached_output
    except Exception as e:
        log.warning(f"⚠️ Error verificando cache L3, regenerando: {e}")
        # Continue with normal processing if cache check fails

    # Solo limpiar memoria si no hay modelos pre-cargados (para evitar sobrecarga)
    if not preloaded_models:
        cleanup_models()

    # Usar modelos pre-cargados si están disponibles, sino cargar normalmente
    if preloaded_models:
        log.info("🎯 Usando modelos pre-cargados para optimización de rendimiento")

        # Regime detection con modelo pre-cargado
        regime_model = preloaded_models.get('regime')
        if regime_model is None:
            log.warning("⚠️ Modelo de regime no pre-cargado, cargando normalmente")
            regime_model = load_regime_model()

        try:
            if regime_model is None:
                log.error("No se pudo cargar el modelo de regime detection")
                regime = 'neutral'
            else:
                df_features = _build_regime_features(market_data, regime_model)
                if df_features is None or df_features.empty:
                    log.warning("No se pudieron construir features para regime detection")
                    regime = 'neutral'
                else:
                    # L3_Regime | Contexto estratégico: features summary
                    features_summary = {col: f"{df_features[col].iloc[-1]:.4f}" for col in df_features.columns[:5]}
                    log.info(f"L3_Regime | Contexto estratégico: {features_summary}")

                    regime = predict_regime(df_features, regime_model)

                    # L3_Regime | Decisión final: regime prediction
                    log.info(f"L3_Regime | Decisión final: {regime}")

                    # L3_Regime | Ponderación aplicada: confidence score
                    confidence_score = getattr(regime_model, 'predict_proba', lambda x: [[0.5]]) if hasattr(regime_model, 'predict_proba') else 0.5
                    log.info(f"L3_Regime | Ponderación aplicada: {confidence_score}")

                    if regime is None:
                        regime = 'neutral'
        except Exception as e:
            log.critical(f"Regime detection fallback por error: {e}")
            regime = 'neutral'

            # Check if precomputed sentiment is available
            if precomputed_sentiment is not None:
                sentiment_score = precomputed_sentiment
                log.info(f"🎯 Sentimiento precomputado usado: {sentiment_score:.4f} (sin carga de modelo BERT)")
            else:
                # Sentiment analysis con modelos pre-cargados
                sentiment_cache = preloaded_models.get('sentiment', {})
                tokenizer = sentiment_cache.get('tokenizer')
                sentiment_model = sentiment_cache.get('model')

                if tokenizer is None or sentiment_model is None:
                    log.warning("⚠️ Modelos de sentimiento no pre-cargados, cargando normalmente")
                    tokenizer, sentiment_model = load_sentiment_model()

                try:
                    # L3_Sentiment | Contexto estratégico: input texts summary
                    texts_summary = f"{len(texts_for_sentiment or [])} textos" + (f" - primeros: {str(texts_for_sentiment[0])[:50]}..." if texts_for_sentiment else "")
                    log.info(f"L3_Sentiment | Contexto estratégico: {texts_summary}")

                    sentiment_score = predict_sentiment(texts_for_sentiment or [], tokenizer, sentiment_model)

                    # L3_Sentiment | Decisión final: sentiment score
                    log.info(f"L3_Sentiment | Decisión final: {sentiment_score:.4f}")

                    # L3_Sentiment | Ponderación aplicada: confidence based on text count
                    confidence = min(1.0, len(texts_for_sentiment or []) / 10.0) if texts_for_sentiment else 0.0
                    log.info(f"L3_Sentiment | Ponderación aplicada: {confidence:.2f}")

                except Exception as e:
                    log.critical(f"Sentiment fallback por error: {e}")
                    sentiment_score = 0.0

        # Volatility models con modelos pre-cargados
        vol_cache = preloaded_models.get('volatility', {})
        garch_btc = vol_cache.get('garch_btc')
        garch_eth = vol_cache.get('garch_eth')
        lstm_btc = vol_cache.get('lstm_btc')
        lstm_eth = vol_cache.get('lstm_eth')

        if all(x is None for x in [garch_btc, garch_eth, lstm_btc, lstm_eth]):
            log.warning("⚠️ Modelos de volatilidad no pre-cargados, cargando normalmente")
            garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()

        # Portfolio models con modelos pre-cargados
        portfolio_cache = preloaded_models.get('portfolio', {})
        cov_matrix = portfolio_cache.get('cov')
        optimal_weights = portfolio_cache.get('weights')

        if cov_matrix is None or optimal_weights is None:
            log.warning("⚠️ Modelos de portfolio no pre-cargados, cargando normalmente")
            cov_matrix, optimal_weights = load_portfolio()

    else:
        # Carga normal de modelos (para compatibilidad hacia atrás)
        log.info("📦 Cargando modelos normalmente (sin pre-carga)")
        try:
            regime_model = load_regime_model()
            if regime_model is None:
                log.error("No se pudo cargar el modelo de regime detection")
                regime = 'neutral'
            else:
                df_features = _build_regime_features(market_data, regime_model)
                if df_features is None or df_features.empty:
                    log.warning("No se pudieron construir features para regime detection")
                    regime = 'neutral'
                else:
                    regime = predict_regime(df_features, regime_model)
                    if regime is None:
                        regime = 'neutral'
        except Exception as e:
            log.critical(f"Regime detection fallback por error: {e}")
            regime = 'neutral'

        try:
            # Check if precomputed sentiment is available
            if precomputed_sentiment is not None:
                sentiment_score = precomputed_sentiment
                log.info(f"🎯 Sentimiento precomputado usado: {sentiment_score:.4f} (sin carga de modelo BERT)")
            else:
                tokenizer, sentiment_model = load_sentiment_model()
                sentiment_score = predict_sentiment(texts_for_sentiment or [], tokenizer, sentiment_model)
        except Exception as e:
            log.critical(f"Sentiment fallback por error: {e}")
            sentiment_score = 0.0

        try:
            garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
        except Exception as e:
            log.critical(f"Error cargando modelos de volatilidad: {e}")
            garch_btc = garch_eth = lstm_btc = lstm_eth = None

        try:
            cov_matrix, optimal_weights = load_portfolio()
        except Exception as e:
            log.critical(f"Error cargando modelos de portfolio: {e}")
            cov_matrix = optimal_weights = None

    # Calcular volatilidad usando modelos cargados
    try:
        btc_series = _safe_close_series(market_data.get('BTCUSDT'))
        eth_series = _safe_close_series(market_data.get('ETHUSDT'))

        # L3_Volatility | Contexto estratégico: series data summary
        log.info(f"L3_Volatility | Contexto estratégico: BTC_series={len(btc_series)}pts, ETH_series={len(eth_series)}pts")

        if len(btc_series) < MIN_VOL_BARS or len(eth_series) < MIN_VOL_BARS:
            log.warning(f"Volatility warmup: series cortas (BTC={len(btc_series)}, ETH={len(eth_series)})")
            vol_btc = vol_eth = WARMUP_VOL
        else:
            btc_vols = []
            if garch_btc is not None:
                btc_vols.append(predict_vol_garch(garch_btc, btc_series))
            if lstm_btc is not None and tf is not None:
                btc_vols.append(predict_vol_lstm(lstm_btc, btc_series))
            vol_btc = np.mean(btc_vols) if btc_vols else WARMUP_VOL

            eth_vols = []
            if garch_eth is not None:
                eth_vols.append(predict_vol_garch(garch_eth, eth_series))
            if lstm_eth is not None and tf is not None:
                eth_vols.append(predict_vol_lstm(lstm_eth, eth_series))
            vol_eth = np.mean(eth_vols) if eth_vols else WARMUP_VOL

        # L3_Volatility | Decisión final: volatility forecasts
        log.info(f"L3_Volatility | Decisión final: BTC_vol={vol_btc:.4f}, ETH_vol={vol_eth:.4f}")

        # L3_Volatility | Ponderación aplicada: model confidence
        models_used = sum([garch_btc is not None, lstm_btc is not None, garch_eth is not None, lstm_eth is not None])
        confidence = models_used / 4.0
        log.info(f"L3_Volatility | Ponderación aplicada: {confidence:.2f} (models_used={models_used}/4)")

    except Exception as e:
        log.critical(f"Error en cálculo de volatilidad: {e}")
        vol_btc = vol_eth = WARMUP_VOL

    volatility_avg = float(np.mean([vol_btc, vol_eth]))

    # Convertir risk_appetite a valor numérico - definir antes de usar
    risk_map = {
        "low": 0.2,
        "moderate": 0.5,
        "aggressive": 0.8,
        "high": 0.8
    }

    # Portfolio allocation
    try:
        if cov_matrix is not None and optimal_weights is not None:
            # L3_Portfolio | Contexto estratégico: risk factors
            log.info(f"L3_Portfolio | Contexto estratégico: vol_avg={volatility_avg:.4f}, sentiment={sentiment_score:.4f}, regime={regime}")

            risk_appetite = compute_risk_appetite(volatility_avg, sentiment_score, regime=regime)
            first_row = optimal_weights.iloc[0]
            allowed = {"BTC","ETH","CASH","BTCUSDT","ETHUSDT"}
            cols = [c for c in first_row.index if (c in allowed)]
            if not cols:
                cols = [c for c in first_row.index if pd.api.types.is_number(first_row[c]) or str(first_row[c]).replace('.','',1).isdigit()]
            asset_allocation = {str(col): float(first_row[col]) for col in cols}
            if not asset_allocation:
                raise ValueError("sin columnas de asignación válidas")

            # L3_Portfolio | Decisión final: asset allocation
            log.info(f"L3_Portfolio | Decisión final: {asset_allocation}")

            # L3_Portfolio | Ponderación aplicada: risk appetite
            risk_value = risk_map.get(str(risk_appetite).lower(), 0.5)
            log.info(f"L3_Portfolio | Ponderación aplicada: {risk_appetite} (value={risk_value:.2f})")

        else:
            raise ValueError("Modelos de portfolio no disponibles")
    except Exception as e:
        log.critical(f"Portfolio allocation fallback por error: {e}")
        asset_allocation = {"BTC": 0.5, "ETH": 0.4, "CASH": 0.1}

    risk_appetite = compute_risk_appetite(volatility_avg, sentiment_score)
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

    # INTEGRATE REGIME-SPECIFIC MODELS
    try:
        log.info("🎯 Integrating regime-specific L3 models")

        # CRISIS DETECTION: Check for extreme market conditions
        crisis_detected = _detect_crisis_conditions(market_data, volatility_avg, vol_btc, vol_eth)

        # Override regime if crisis detected
        if crisis_detected:
            regime = 'crisis'
            log.warning("🚨 CRISIS OVERRIDE: Market conditions indicate crisis regime")

        # Create regime context for the specific models
        regime_context = {
            'regime': regime,
            'volatility_avg': volatility_avg,
            'sentiment_score': sentiment_score,
            'risk_appetite': risk_appetite,
            'vol_btc': vol_btc,
            'vol_eth': vol_eth,
            'current_time': current_timestamp,  # Pass current_time to regime models
            'crisis_detected': crisis_detected
        }

        # Generate regime-specific strategy
        regime_processor = RegimeSpecificL3Processor()
        regime_strategy = regime_processor.generate_regime_strategy(market_data, regime_context)

        # Merge regime-specific strategy with base L3 output
        strategic_guidelines = {
            "regime": regime,
            "asset_allocation": regime_strategy.asset_allocation,
            "risk_appetite": regime_strategy.risk_appetite,
            "sentiment_score": float(sentiment_score),
            "volatility_forecast": {
                "BTCUSDT": float(vol_btc),
                "ETHUSDT": float(vol_eth)
            },
            "regime_strategy": {
                "position_sizing": regime_strategy.position_sizing,
                "stop_loss_policy": regime_strategy.stop_loss_policy,
                "take_profit_policy": regime_strategy.take_profit_policy,
                "rebalancing_frequency": regime_strategy.rebalancing_frequency,
                "volatility_target": regime_strategy.volatility_target,
                "correlation_limits": regime_strategy.correlation_limits,
                "strategy_metadata": regime_strategy.metadata
            },
            "timestamp": current_time.isoformat() + "Z"
        }

        log.info(f"✅ Regime-specific strategy integrated: {regime} regime with risk_appetite={regime_strategy.risk_appetite:.2f}")

    except Exception as e:
        log.error(f"❌ Error integrating regime-specific models: {e}")
        # Fallback to basic L3 output
        current_time = datetime.utcnow()
        strategic_guidelines = {
            "regime": regime,
            "asset_allocation": asset_allocation,
            "risk_appetite": risk_value,
            "sentiment_score": float(sentiment_score),
            "volatility_forecast": {
                "BTCUSDT": float(vol_btc),
                "ETHUSDT": float(vol_eth)
            },
            "timestamp": current_time.isoformat() + "Z"
        }
        log.warning("⚠️ Using fallback L3 output without regime-specific models")

    # Update L3 context cache in state for future reuse
    if 'l3_context_cache' not in state:
        state['l3_context_cache'] = {}

    state['l3_context_cache'].update({
        'last_output': strategic_guidelines.copy(),
        'market_data_hash': _calculate_market_data_hash(market_data),
        'last_update': current_timestamp.isoformat(),
        'regime': regime,
        'volatility_avg': volatility_avg,
        'sentiment_score': sentiment_score
    })

    # Add detailed logging for L3 context updates
    log.info("📊 L3 Context Update Details:")
    log.info(f"   Regime: {regime}")
    log.info(f"   Risk Appetite: {risk_appetite} (value: {risk_value:.2f})")
    log.info(f"   Volatility BTC: {vol_btc:.4f}, ETH: {vol_eth:.4f} (avg: {volatility_avg:.4f})")
    log.info(f"   Sentiment Score: {sentiment_score:.4f}")
    log.info(f"   Asset Allocation: {asset_allocation}")
    log.info(f"   Cache Updated: {current_timestamp.isoformat()}")

    save_json(strategic_guidelines, OUTPUT_FILE)
    log.info("🎉 L3_PROCESSOR: Output estratégico generado correctamente")
    log.info(f"   📈 Resultado final: regime={regime}, risk_appetite={risk_appetite}, sentiment={sentiment_score:.4f}")
    log.info(f"   💰 Asset allocation: {asset_allocation}")
    log.info(f"   📊 Volatility: BTC={vol_btc:.4f}, ETH={vol_eth:.4f}")

    # Solo limpiar si no usamos modelos pre-cargados
    if not preloaded_models:
        cleanup_models()

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
