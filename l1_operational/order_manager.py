# l1_operational/order_manager.py
import logging
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from .models import Signal, ExecutionReport
from .risk_guard import RiskGuard
from .executor import Executor

logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self):
        self.risk_guard = RiskGuard()
        self.executor = Executor()
        self.processed_signals = {}
        
        # CARGAR LOS 3 MODELOS IA
        self.models = self._load_ai_models()
        
    def _load_ai_models(self) -> Dict[str, Any]:
        """Carga los 3 modelos IA de L1"""
        models = {}
        model_paths = {
            'logreg': Path('models/L1/modelo1_lr.pkl'),
            'random_forest': Path('models/L1/modelo2_rf.pkl'), 
            'lightgbm': Path('models/L1/modelo3_lgbm.pkl')
        }
        
        for name, path in model_paths.items():
            try:
                if path.exists():
                    with open(path, 'rb') as f:
                        models[name] = pickle.load(f)
                    logger.info(f"✅ Modelo {name} cargado desde {path}")
                else:
                    logger.warning(f"⚠️ Modelo {name} no encontrado en {path}")
            except Exception as e:
                logger.error(f"❌ Error cargando {name}: {e}")
        
        return models
    
    def _ai_filter_signal(self, signal: Signal) -> Dict[str, float]:
        """Filtra señal usando los 3 modelos IA"""
        if not self.models:
            logger.warning("No hay modelos IA cargados, usando validación básica")
            return {'logreg': 0.5, 'random_forest': 0.5, 'lightgbm': 0.5}
        
        # Simular features (deberías extraer features reales del signal)
        features = self._extract_features(signal)
        
        scores = {}
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Para modelos de clasificación
                    prob = model.predict_proba(features.reshape(1, -1))[0]
                    scores[model_name] = prob[1] if len(prob) > 1 else prob[0]
                else:
                    # Para modelos de regresión
                    score = model.predict(features.reshape(1, -1))[0]
                    scores[model_name] = max(0, min(1, score))
                    
                logger.info(f"[AI-{model_name}] Score: {scores[model_name]:.3f}")
                
            except Exception as e:
                logger.error(f"Error en modelo {model_name}: {e}")
                scores[model_name] = 0.0
        
        return scores
    
    def _extract_features(self, signal: Signal) -> np.ndarray:
        """Extrae features del signal para los modelos IA"""
        # IMPLEMENTAR: extraer features reales como RSI, MACD, etc.
        # Por ahora, features simuladas
        return np.array([0.5, 0.3, 0.7, 0.6, 0.4])  # 5 features ejemplo
    
    async def handle_signal(self, signal: Signal) -> ExecutionReport:
        """Flujo L1 con IA integrada"""
        logger.info(f"[L1] Ejecutando capa Operational...")
        start_time = time.time()

        # PASO 1: Validación hard-coded
        validation_result = self.risk_guard.validate_signal(signal)
        if not validation_result.is_valid:
            return ExecutionReport(
                signal_id=signal.signal_id,
                status="REJECTED_SAFETY",
                reason=validation_result.reason,
                timestamp=time.time()
            )

        # PASO 2: FILTRO IA (NUEVO)
        ai_scores = self._ai_filter_signal(signal)
        
        # Decisión basada en ensemble de los 3 modelos
        avg_score = np.mean(list(ai_scores.values()))
        ai_threshold = 0.6  # Configurable
        
        if avg_score < ai_threshold:
            logger.info(f"Signal {signal.signal_id} filtrada por IA: score={avg_score:.3f} < {ai_threshold}")
            return ExecutionReport(
                signal_id=signal.signal_id,
                status="REJECTED_AI",
                reason=f"AI ensemble score {avg_score:.3f} below threshold {ai_threshold}",
                ai_scores=ai_scores,
                timestamp=time.time()
            )

        # PASO 3: Ejecución
        try:
            execution_result = await self.executor.execute_order(signal)
            
            report = ExecutionReport(
                signal_id=signal.signal_id,
                status="EXECUTED",
                executed_qty=execution_result.filled_qty,
                executed_price=execution_result.avg_price,
                ai_scores=ai_scores,  # Incluir scores en reporte
                timestamp=time.time()
            )
            
            logger.info(f"✅ Signal {signal.signal_id} ejecutada con AI scores: {ai_scores}")

        except Exception as e:
            report = ExecutionReport(
                signal_id=signal.signal_id,
                status="EXECUTION_ERROR", 
                reason=str(e),
                timestamp=time.time()
            )

        self.processed_signals[signal.signal_id] = report
        return report

# Crear instancia global para uso en __init__.py
order_manager = OrderManager()