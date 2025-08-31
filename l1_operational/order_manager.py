# l1_operational/order_manager.py - CORREGIDO
import logging
import joblib
import pickle
import os
from typing import Dict, Any, Optional
from .config import ConfigObject
from .models import Signal, SignalType, SignalSource, create_signal

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, binance_client=None):
        """
        Inicializa el OrderManager usando la configuración de constantes.
        """
        # Usar la configuración de constantes directamente
        self.config = ConfigObject  # Tu módulo config.py
        
        # Cliente Binance
        self.binance_client = binance_client
        
        # Inicializar modelos IA con lazy loading
        self.models = {}
        self._models_loaded = False
        
        # Otras inicializaciones
        self.active_orders = {}
        self.execution_stats = {}
        
        # Límites de riesgo (desde tu config)
        self.risk_limits = ConfigObject.RISK_LIMITS
        self.portfolio_limits = ConfigObject.PORTFOLIO_LIMITS
        self.execution_config = ConfigObject.EXECUTION_CONFIG
        
        # Cargar modelos en background SIN TIMEOUT
        self._load_ai_models_async()
        
        logger.info(f"✅ OrderManager inicializado - Modo: {ConfigObject.OPERATION_MODE}")
        logger.info(f"✅ Límites BTC: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_BTC']}, ETH: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")

    def _load_ai_models_async(self):
        """
        Carga los modelos de forma asíncrona SIN timeout
        """
        import threading
        
        def load_models_thread():
            try:
                self._load_ai_models()
                self._models_loaded = True
                logger.info("✅ Modelos L1 cargados completamente")
            except Exception as e:
                logger.error(f"❌ Error cargando modelos L1: {e}")
                # Continuar funcionamiento sin modelos IA
                self._models_loaded = True  # Permitir que el sistema continue
        
        # Iniciar carga en hilo separado
        thread = threading.Thread(target=load_models_thread, daemon=True)
        thread.start()

    def _load_ai_models(self):
        """
        Carga los 3 modelos IA de L1 SIN timeout para modelos pesados
        """
        model_configs = [
            ('logreg', 'models/L1/modelo1_lr.pkl'),
            ('random_forest', 'models/L1/modelo2_rf.pkl'),
            ('lightgbm', 'models/L1/modelo3_lgbm.pkl')
        ]
        
        # Cargar modelos secuencialmente para evitar problemas de memoria
        for model_name, model_path in model_configs:
            try:
                model = self._load_model_safely(model_path, model_name)
                if model:
                    self.models[model_name] = model
                    logger.info(f"✅ Modelo {model_name} cargado desde {model_path}")
                else:
                    logger.error(f"❌ Falló carga de {model_name}")
            except Exception as e:
                logger.error(f"❌ Error crítico cargando {model_name}: {e}")
                # Continuar con otros modelos

    def _load_model_safely(self, path: str, model_name: str):
        """
        Carga con verificación previa y sin timeout
        """
        if not os.path.exists(path):
            logger.error(f"❌ Archivo de modelo no encontrado: {path}")
            return None
        
        # Verificación rápida de tamaño
        file_size = os.path.getsize(path)
        if file_size < 100:
            logger.error(f"❌ Archivo demasiado pequeño: {path} ({file_size} bytes)")
            return None
        
        logger.info(f"🔄 Cargando {model_name} ({file_size/1024:.1f}KB)...")
        
        # Para modelos grandes, usar método más eficiente
        if file_size > 1024 * 1024:  # > 1MB
            logger.info(f"📦 Modelo grande detectado, usando carga optimizada...")
        
        # Métodos de carga en orden de eficiencia
        load_methods = [
            ('joblib', lambda: joblib.load(path)),
            ('pickle_rb', lambda: pickle.load(open(path, 'rb'))),
            ('joblib_mmap', lambda: joblib.load(path, mmap_mode='r') if file_size > 10*1024*1024 else None),
        ]
        
        for method_name, load_func in load_methods:
            try:
                model = load_func()
                if model is not None:
                    logger.info(f"✅ {model_name} cargado con {method_name}")
                    return model
            except Exception as e:
                logger.debug(f"⚠️ {method_name} falló para {model_name}: {str(e)[:100]}")
                continue
        
        logger.error(f"❌ Todos los métodos fallaron para {model_name}")
        return None

    def get_ai_prediction(self, features: list, symbol: str) -> Dict[str, float]:
        """
        Predicciones con verificación de carga y fallback robusto
        """
        if not self._models_loaded:
            logger.debug("[L1] Modelos aún cargando, usando predicción por defecto")
            return {'ensemble': 0.5}  # Predicción neutral
        
        if not self.models:
            logger.debug("[L1] No hay modelos cargados, usando predicción por defecto")
            return {'ensemble': 0.5}
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model is None:
                    continue
                
                # Preparar features según el modelo
                prediction_value = self._get_model_prediction(model, features)
                predictions[model_name] = float(prediction_value)
                
            except Exception as e:
                logger.error(f"❌ Error en predicción {model_name}: {e}")
                predictions[model_name] = 0.5  # Neutral en caso de error
        
        # Ensemble de los modelos disponibles
        if predictions:
            ensemble_score = sum(predictions.values()) / len(predictions)
            predictions['ensemble'] = ensemble_score
            logger.debug(f"[L1] Ensemble IA para {symbol}: {ensemble_score:.3f} (modelos: {len(predictions)-1})")
        else:
            predictions['ensemble'] = 0.5  # Neutral si no hay predicciones
            logger.warning(f"[L1] Sin modelos disponibles para {symbol}, usando neutral")
        
        return predictions

    def _get_model_prediction(self, model, features: list) -> float:
        """
        Obtiene predicción optimizada según tipo de modelo
        """
        try:
            if hasattr(model, 'predict_proba'):
                # Para LogReg y RandomForest
                pred = model.predict_proba([features])[0]
                return max(pred) if len(pred) > 1 else pred[0]
            elif hasattr(model, 'predict'):
                # Para LightGBM y otros
                return model.predict([features])[0]
            else:
                return 0.5  # Neutral
        except Exception:
            return 0.5  # Fallback neutral

    def process_signals(self, signals: list) -> Dict[str, Any]:
        """
        Procesa señales tácticas con modelos IA cuando estén disponibles
        """
        processed_orders = []
        
        for signal in signals:
            try:
                # Si hay features disponibles y modelos cargados, usar IA
                if hasattr(signal, 'features') and signal.features and self._models_loaded:
                    # Preparar features para IA
                    feature_vector = self._extract_features_from_signal(signal)
                    if feature_vector:
                        ai_predictions = self.get_ai_prediction(feature_vector, signal.symbol)
                        
                        # Ajustar strength basado en ensemble IA
                        ensemble_score = ai_predictions.get('ensemble', 0.5)
                        original_strength = signal.strength
                        
                        # Modular la señal con IA (ensemble > 0.6 = reforzar, < 0.4 = atenuar)
                        if ensemble_score > 0.6:
                            signal.strength *= min(ensemble_score * 1.5, 1.0)
                        elif ensemble_score < 0.4:
                            signal.strength *= max(ensemble_score * 1.5, 0.1)
                        
                        logger.debug(f"[L1] Señal {signal.symbol} ajustada por IA: {original_strength:.3f} → {signal.strength:.3f}")
                
                # Procesar señal (validación, ejecución, etc.)
                order_result = self._process_single_signal(signal)
                if order_result:
                    processed_orders.append(order_result)
                    
            except Exception as e:
                logger.error(f"❌ Error procesando señal {signal.symbol}: {e}")
                continue
        
        return {
            'orders_processed': len(processed_orders),
            'orders': processed_orders,
            'models_loaded': self._models_loaded,
            'available_models': list(self.models.keys())
        }
    
    def _extract_features_from_signal(self, signal) -> list:
        """
        Extrae features numéricas de una señal para los modelos IA
        """
        try:
            features = []
            
            if hasattr(signal, 'features') and isinstance(signal.features, dict):
                # Extraer features OHLCV
                ohlcv = signal.features.get('ohlcv', {})
                features.extend([
                    ohlcv.get('open', 0),
                    ohlcv.get('high', 0),
                    ohlcv.get('low', 0),
                    ohlcv.get('close', 0),
                    ohlcv.get('volume', 0)
                ])
                
                # Extraer indicators
                indicators = signal.features.get('indicators', {})
                features.extend([
                    indicators.get('rsi', 50),
                    indicators.get('macd', 0),
                    indicators.get('macd_signal', 0),
                    signal.strength,
                    signal.confidence
                ])
            
            # Completar hasta 12 features si es necesario
            while len(features) < 12:
                features.append(0.0)
            
            return features[:12]  # Limitar a 12 features
            
        except Exception as e:
            logger.error(f"❌ Error extrayendo features: {e}")
            return None
    
    def _process_single_signal(self, signal):
        """
        Procesa una señal individual (placeholder)
        """
        # Aquí iría la lógica de validación y ejecución
        return {
            'symbol': signal.symbol,
            'side': signal.side,
            'strength': signal.strength,
            'status': 'processed'
        }