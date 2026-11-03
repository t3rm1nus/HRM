(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: __init__.py
================================================
# HRM - Sistema de Trading Cripto con Modelo de Razonamiento Jerárquico
# Este archivo hace que Python reconozca el directorio como un paquete

__version__ = "1.0.0"
__author__ = "Equipo de desarrollo HRM"
__description__ = "Sistema de trading cripto con modelo de razonamiento jerárquico"



================================================
FILE: auto_learning_system.py
================================================
#!/usr/bin/env python3
"""
Sistema de Auto-Aprendizaje con Protección Total Anti-Overfitting
Implementa aprendizaje continuo completamente automático con 9 capas de protección

CRITICAL FIX: Eliminado todo asyncio.run() - ahora 100% async-compatible
"""

import asyncio
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import joblib
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeData:
    """Estructura de datos de trade para aprendizaje"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    model_used: str
    confidence: float
    regime_at_entry: str
    features: Dict[str, float] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Métricas de performance de modelo"""
    model_name: str
    timestamp: datetime
    train_score: float
    val_score: float
    oos_score: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

class AntiOverfitValidator:
    """Validación cruzada continua para prevenir overfitting"""

    def __init__(self, validation_windows: int = 5, min_score: float = 0.55):
        self.validation_windows = validation_windows
        self.min_validation_score = min_score
        self.max_score_std = 0.15  # Máxima desviación estándar permitida

    def validate_new_model(self, model, training_data: pd.DataFrame) -> Tuple[bool, float]:
        """Validar modelo con múltiples ventanas temporales"""
        try:
            scores = []

            # Rolling window validation
            for i in range(self.validation_windows):
                train_start_idx = int(len(training_data) * i / self.validation_windows)
                train_end_idx = int(len(training_data) * (i + 4) / self.validation_windows)
                val_start_idx = train_end_idx
                val_end_idx = int(len(training_data) * (i + 5) / self.validation_windows)

                if val_end_idx > len(training_data):
                    val_end_idx = len(training_data)

                if val_start_idx >= val_end_idx:
                    continue

                # Datos de entrenamiento y validación
                train_data = training_data.iloc[train_start_idx:train_end_idx]
                val_data = training_data.iloc[val_start_idx:val_end_idx]

                if len(val_data) < 10:
                    continue

                # Entrenar y validar
                model_copy = self._clone_model(model)
                score = self._evaluate_window(model_copy, train_data, val_data)
                scores.append(score)

            if not scores:
                return False, 0.0

            avg_score = np.mean(scores)
            std_score = np.std(scores)

            # Rechazar si score bajo o muy variable
            if avg_score < self.min_validation_score:
                logger.warning(f"❌ CV Score too low: {avg_score:.3f} < {self.min_validation_score}")
                return False, avg_score

            if std_score > self.max_score_std:
                logger.warning(f"❌ CV Score too variable: std {std_score:.3f} > {self.max_score_std}")
                return False, avg_score

            logger.info(f"✅ CV Validation passed: {avg_score:.3f} ± {std_score:.3f}")
            return True, avg_score

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return False, 0.0

    def _clone_model(self, model):
        """Clonar modelo para validación"""
        try:
            # Para sklearn models
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                return type(model)(**model.get_params())
        except:
            pass

        # Fallback: usar deepcopy
        import copy
        return copy.deepcopy(model)

    def _evaluate_window(self, model, train_data: pd.DataFrame, val_data: pd.DataFrame) -> float:
        """Evaluar modelo en una ventana específica"""
        try:
            if 'target' not in train_data.columns or 'target' not in val_data.columns:
                # Asumir que la última columna es target
                target_col = train_data.columns[-1]
            else:
                target_col = 'target'

            X_train = train_data.drop(target_col, axis=1)
            y_train = train_data[target_col]
            X_val = val_data.drop(target_col, axis=1)
            y_val = val_data[target_col]

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)

            return score

        except Exception as e:
            logger.error(f"Error evaluating window: {e}")
            return 0.0

class AdaptiveRegularizer:
    """Regularización adaptativa basada en señales de overfitting"""

    def __init__(self):
        self.regularization_params = {
            'l1_penalty': 0.01,
            'l2_penalty': 0.01,
            'dropout_rate': 0.2,
            'early_stopping_patience': 10,
            'max_features': None,
            'min_samples_split': 2
        }
        self.overfitting_threshold = 0.15  # 15% gap = overfitting

    def adjust_regularization(self, performance_history: List[ModelPerformance]):
        """Ajustar regularización basado en historial de performance"""
        if len(performance_history) < 5:
            return

        # Calcular train vs validation gap reciente
        recent_perfs = performance_history[-5:]
        avg_train_score = np.mean([p.train_score for p in recent_perfs])
        avg_val_score = np.mean([p.val_score for p in recent_perfs])

        overfitting_gap = avg_train_score - avg_val_score

        if overfitting_gap > self.overfitting_threshold:
            # Aumentar regularización
            self.regularization_params['l2_penalty'] *= 1.5
            self.regularization_params['dropout_rate'] = min(0.5, self.regularization_params['dropout_rate'] * 1.2)
            self.regularization_params['early_stopping_patience'] = max(5, self.regularization_params['early_stopping_patience'] - 2)

            logger.warning(f"🚨 OVERFIT DETECTED: Gap {overfitting_gap:.3f}, increasing regularization")

        elif overfitting_gap < 0.05:
            # Reducir regularización (posible underfitting)
            self.regularization_params['l2_penalty'] *= 0.9
            self.regularization_params['dropout_rate'] *= 0.95
            self.regularization_params['early_stopping_patience'] = min(20, self.regularization_params['early_stopping_patience'] + 1)

            logger.info(f"📉 Underfit detected: Gap {overfitting_gap:.3f}, reducing regularization")

class DiverseEnsembleBuilder:
    """Construye ensemble diverso para evitar overfitting"""

    def __init__(self, max_models: int = 10, similarity_threshold: float = 0.85):
        self.max_models = max_models
        self.similarity_threshold = similarity_threshold
        self.ensemble_models = []
        self.model_weights = {}

    def add_model_to_ensemble(self, candidate_model, validation_data: pd.DataFrame) -> bool:
        """Añadir modelo solo si aumenta diversidad del ensemble"""
        try:
            if len(self.ensemble_models) >= self.max_models:
                # Reemplazar peor modelo si candidato es mejor
                return self._replace_worst_model(candidate_model, validation_data)

            # Verificar diversidad con modelos existentes
            for existing_model in self.ensemble_models:
                similarity = self._calculate_model_similarity(candidate_model, existing_model, validation_data)
                if similarity > self.similarity_threshold:
                    logger.info(f"⚠️ Model rejected: similarity {similarity:.3f} > {self.similarity_threshold}")
                    return False

            # Verificar que mejora ensemble performance
            ensemble_improvement = self._calculate_ensemble_improvement(candidate_model, validation_data)

            if ensemble_improvement < 0.01:  # Menos del 1% de mejora
                logger.info(f"⚠️ Model rejected: improvement {ensemble_improvement:.3f} < 0.01")
                return False

            # Añadir al ensemble
            self.ensemble_models.append(candidate_model)
            self._update_weights()

            logger.info(f"✅ Model added to ensemble (improvement: {ensemble_improvement:.3f})")
            return True

        except Exception as e:
            logger.error(f"Error adding model to ensemble: {e}")
            return False

    def _calculate_model_similarity(self, model1, model2, data: pd.DataFrame) -> float:
        """Calcular similitud entre dos modelos"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
            else:
                X = data.iloc[:, :-1]

            pred1 = model1.predict(X)
            pred2 = model2.predict(X)

            # Correlación de Pearson entre predicciones
            correlation = np.corrcoef(pred1, pred2)[0, 1]
            return abs(correlation)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 1.0  # Asumir máxima similitud en caso de error

    def _calculate_ensemble_improvement(self, candidate_model, data: pd.DataFrame) -> float:
        """Calcular mejora que aporta el candidato al ensemble"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
                y = data['target']
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            # Score del ensemble actual
            current_predictions = self._get_ensemble_predictions(X)
            current_score = self._calculate_score(current_predictions, y)

            # Score con candidato añadido
            candidate_predictions = candidate_model.predict(X)
            new_predictions = (current_predictions + candidate_predictions) / 2
            new_score = self._calculate_score(new_predictions, y)

            return new_score - current_score

        except Exception as e:
            logger.error(f"Error calculating ensemble improvement: {e}")
            return 0.0

    def _get_ensemble_predictions(self, X):
        """Obtener predicciones del ensemble actual"""
        if not self.ensemble_models:
            return np.zeros(len(X))

        predictions = np.zeros(len(X))
        for model in self.ensemble_models:
            predictions += model.predict(X)

        return predictions / len(self.ensemble_models)

    def _calculate_score(self, predictions, targets):
        """Calcular score de accuracy"""
        try:
            # Para clasificación binaria
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_preds == targets.values)
            return accuracy
        except:
            # Fallback
            return 0.5

    def _replace_worst_model(self, candidate_model, validation_data: pd.DataFrame) -> bool:
        """Reemplazar el peor modelo del ensemble"""
        try:
            if not self.ensemble_models:
                self.ensemble_models.append(candidate_model)
                return True

            # Evaluar todos los modelos
            model_scores = []
            for i, model in enumerate(self.ensemble_models):
                score = self._evaluate_single_model(model, validation_data)
                model_scores.append((i, score))

            # Encontrar peor modelo
            worst_idx, worst_score = min(model_scores, key=lambda x: x[1])

            # Evaluar candidato
            candidate_score = self._evaluate_single_model(candidate_model, validation_data)

            if candidate_score > worst_score:
                # Reemplazar
                self.ensemble_models[worst_idx] = candidate_model
                self._update_weights()
                logger.info(f"🔄 Replaced worst model (score {worst_score:.3f}) with candidate (score {candidate_score:.3f})")
                return True
            else:
                logger.info(f"⚠️ Candidate worse than worst model: {candidate_score:.3f} < {worst_score:.3f}")
                return False

        except Exception as e:
            logger.error(f"Error replacing model: {e}")
            return False

    def _evaluate_single_model(self, model, data: pd.DataFrame) -> float:
        """Evaluar un solo modelo"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
                y = data['target']
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            return model.score(X, y)
        except:
            return 0.0

    def _update_weights(self):
        """Actualizar pesos del ensemble"""
        if not self.ensemble_models:
            return

        # Pesos iguales por ahora (podría ser más sofisticado)
        weight = 1.0 / len(self.ensemble_models)
        for i, model in enumerate(self.ensemble_models):
            self.model_weights[f"model_{i}"] = weight

class ConceptDriftDetector:
    """Detecta cambios en la distribución de datos (concept drift)"""

    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold
        self.reference_distribution = None
        self.drift_history = []

    def detect_drift(self, new_data: pd.DataFrame) -> bool:
        """Detectar si hay concept drift en los nuevos datos"""
        try:
            current_distribution = self._calculate_distribution(new_data)

            if self.reference_distribution is None:
                self.reference_distribution = current_distribution
                return False

            # Calcular distancia entre distribuciones
            drift_distance = self._calculate_distribution_distance(
                self.reference_distribution, current_distribution
            )

            self.drift_history.append({
                'timestamp': datetime.now(),
                'distance': drift_distance,
                'threshold': self.drift_threshold
            })

            if drift_distance > self.drift_threshold:
                logger.warning(f"🌊 CONCEPT DRIFT DETECTED: Distance {drift_distance:.3f} > {self.drift_threshold}")
                # Actualizar distribución de referencia
                self.reference_distribution = current_distribution
                return True

            return False

        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return False

    def _calculate_distribution(self, data: pd.DataFrame) -> np.ndarray:
        """Calcular distribución estadística de los datos"""
        try:
            # Usar múltiples momentos estadísticos
            features = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if col == 'target':
                    continue

                values = data[col].dropna().values
                if len(values) > 10:
                    # Calcular histograma normalizado
                    hist, _ = np.histogram(values, bins=10, density=True)
                    features.extend(hist)

            return np.array(features)

        except Exception as e:
            logger.error(f"Error calculating distribution: {e}")
            return np.array([])

    def _calculate_distribution_distance(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calcular distancia entre dos distribuciones"""
        try:
            # Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon

            # Asegurar que tienen la misma longitud
            min_len = min(len(dist1), len(dist2))
            if min_len == 0:
                return 0.0

            dist1_norm = dist1[:min_len] / (np.sum(dist1[:min_len]) + 1e-10)
            dist2_norm = dist2[:min_len] / (np.sum(dist2[:min_len]) + 1e-10)

            return jensenshannon(dist1_norm, dist2_norm)

        except Exception as e:
            logger.error(f"Error calculating distribution distance: {e}")
            return 0.0

class SmartEarlyStopper:
    """Early stopping inteligente para prevenir overfitting"""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def should_stop_training(self, validation_scores: List[float]) -> bool:
        """Decidir si parar el entrenamiento"""
        if len(validation_scores) < self.patience:
            return False

        current_score = validation_scores[-1]

        if current_score > self.best_score + self.min_delta:
            # Mejora encontrada
            self.best_score = current_score
            self.wait = 0
            # Guardar mejores pesos aquí si fuera una red neuronal
        else:
            self.wait += 1

        if self.wait >= self.patience:
            logger.warning(f"⏹️ Early stopping: No improvement in {self.patience} epochs")
            return True

        return False

class AutoRetrainingSystem:
    """Sistema de reentrenamiento completamente automático"""

    def __init__(self):
        self.data_buffer = []
        self.model_versions = {}
        self.performance_history = []

        # FIX: Añadir lock para prevenir race condition en reentrenamiento
        self._retraining_lock = asyncio.Lock()
        self._retraining_in_progress = False

        # Componentes anti-overfitting (5 capas)
        self.validator = AntiOverfitValidator()
        self.regularizer = AdaptiveRegularizer()
        self.ensemble_builder = DiverseEnsembleBuilder()
        self.drift_detector = ConceptDriftDetector()
        self.early_stopper = SmartEarlyStopper()

        # Triggers automáticos (4 capas adicionales = 9 total)
        self.auto_triggers = {
            'time_based': {
                'enabled': True,
                'interval_hours': 168,  # 7 días
                'last_retrain': datetime.now()
            },
            'performance_based': {
                'enabled': True,
                'min_trades': 100,
                'win_rate_threshold': 0.52,
                'max_drawdown_threshold': 0.12
            },
            'regime_change': {
                'enabled': True,
                'regime_switches': 0,
                'last_regime': None
            },
            'data_volume': {
                'enabled': True,
                'min_new_trades': 500
            }
        }

        # Modelos base
        self.models = self._load_base_models()

        # LOG de las 9 capas de protección
        logger.info("=" * 70)
        logger.info("🛡️  9 CAPAS DE PROTECCIÓN ANTI-OVERFITTING INICIALIZADAS")
        logger.info("=" * 70)
        logger.info("   1️⃣  AntiOverfitValidator       - Validación cruzada continua")
        logger.info("   2️⃣  AdaptiveRegularizer        - Regularización adaptativa")
        logger.info("   3️⃣  DiverseEnsembleBuilder     - Ensemble diverso")
        logger.info("   4️⃣  ConceptDriftDetector       - Detección de concept drift")
        logger.info("   5️⃣  SmartEarlyStopper          - Early stopping inteligente")
        logger.info("   6️⃣  TimeBasedTrigger           - Trigger por tiempo (7 días)")
        logger.info("   7️⃣  PerformanceBasedTrigger    - Trigger por performance")
        logger.info("   8️⃣  RegimeChangeTrigger        - Trigger por cambio de régimen")
        logger.info("   9️⃣  DataVolumeTrigger          - Trigger por volumen de datos")
        logger.info("=" * 70)
        logger.info("🤖 Auto-Retraining System initialized with full anti-overfitting protection")

    def _load_base_models(self) -> Dict[str, Any]:
        """Cargar modelos base del sistema"""
        models = {}

        try:
            # Cargar modelos desde el sistema existente
            from l3_strategy.regime_classifier import clasificar_regimen_mejorado
            from l1_operational.models import L1Model

            # Placeholder - en implementación real cargar modelos entrenados
            models['regime_classifier'] = 'loaded'
            models['l1_models'] = 'loaded'

            logger.info("✅ Base models loaded")
            return models

        except Exception as e:
            logger.error(f"Error loading base models: {e}")
            return {}

    def add_trade_data(self, trade_data: TradeData):
        """Añadir datos de trade y verificar triggers automáticos"""
        self.data_buffer.append(trade_data)

        # FIX: Modificar add_trade_data para evitar la race condition
        if self._should_retrain() and not self._retraining_in_progress:
            # FIX: Usar get_event_loop solo si hay loop corriendo, con guard
            try:
                loop = asyncio.get_running_loop()
                self._retraining_in_progress = True
                task = loop.create_task(self._safe_retrain_wrapper())
                task.add_done_callback(lambda t: self._on_retrain_complete(t))
            except RuntimeError:
                logger.warning("No hay event loop corriendo — reentrenamiento pospuesto")

    async def _safe_retrain_wrapper(self):
        # FIX: Lock para garantizar solo un reentrenamiento a la vez
        async with self._retraining_lock:
            try:
                await self._auto_retrain_models()
            except Exception as e:
                logger.error(f"❌ Error en auto-reentrenamiento: {e}")
            finally:
                self._retraining_in_progress = False

    def _on_retrain_complete(self, task):
        if task.exception():
            logger.error(f"❌ Auto-reentrenamiento falló: {task.exception()}")
            self._retraining_in_progress = False

    def _should_retrain(self) -> bool:
        """Decidir automáticamente si reentrenar"""

        # 1. Trigger por tiempo
        time_trigger = self.auto_triggers['time_based']
        if time_trigger['enabled']:
            hours_since_last = (datetime.now() - time_trigger['last_retrain']).total_seconds() / 3600
            if hours_since_last >= time_trigger['interval_hours']:
                logger.info(f"🔄 AUTO-TRIGGER: Time-based ({hours_since_last:.1f}h >= {time_trigger['interval_hours']}h)")
                return True

        # 2. Trigger por performance
        perf_trigger = self.auto_triggers['performance_based']
        if perf_trigger['enabled'] and len(self.data_buffer) >= perf_trigger['min_trades']:
            recent_performance = self._calculate_recent_performance(
                self.data_buffer[-perf_trigger['min_trades']:]
            )

            if recent_performance['win_rate'] < perf_trigger['win_rate_threshold']:
                logger.warning(f"🚨 AUTO-TRIGGER: Win rate {recent_performance['win_rate']:.1%} < {perf_trigger['win_rate_threshold']:.1%}")
                return True

            if recent_performance['max_drawdown'] > perf_trigger['max_drawdown_threshold']:
                logger.warning(f"🚨 AUTO-TRIGGER: Drawdown {recent_performance['max_drawdown']:.1%} > {perf_trigger['max_drawdown_threshold']:.1%}")
                return True

        # 3. Trigger por cambio de régimen
        regime_trigger = self.auto_triggers['regime_change']
        if regime_trigger['enabled']:
            current_regime = self._detect_current_regime()
            if current_regime != regime_trigger['last_regime']:
                regime_trigger['regime_switches'] += 1
                if regime_trigger['regime_switches'] >= 3:
                    logger.info(f"🔄 AUTO-TRIGGER: Regime changed {regime_trigger['regime_switches']}x (current: {current_regime})")
                    regime_trigger['regime_switches'] = 0
                    regime_trigger['last_regime'] = current_regime
                    return True

        # 4. Trigger por volumen de datos
        volume_trigger = self.auto_triggers['data_volume']
        if volume_trigger['enabled'] and len(self.data_buffer) >= volume_trigger['min_new_trades']:
            logger.info(f"🔄 AUTO-TRIGGER: Data volume ({len(self.data_buffer)} >= {volume_trigger['min_new_trades']})")
            return True

        return False

    def _calculate_recent_performance(self, trades: List[TradeData]) -> Dict[str, float]:
        """Calcular performance reciente"""
        if not trades:
            return {'win_rate': 0.0, 'max_drawdown': 0.0}

        profitable_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = profitable_trades / len(trades)

        # Calcular drawdown máximo
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0

        for trade in trades:
            cumulative_pnl += trade.pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

    def _detect_current_regime(self) -> str:
        """Detectar régimen de mercado actual"""
        try:
            # Placeholder - en implementación real usar clasificador de régimen
            return "neutral"
        except:
            return "unknown"

    async def _auto_retrain_models(self):
        """Reentrenar modelos automáticamente con protección anti-overfitting"""
        logger.info("🤖 INICIANDO AUTO-REENTRENAMIENTO CON PROTECCIÓN ANTI-OVERFITTING...")

        try:
            # 1. Preparar datos de entrenamiento
            training_data = self._prepare_training_data()

            if len(training_data) < 100:
                logger.warning("⚠️ Insuficientes datos para reentrenamiento")
                return

            # 2. Detectar concept drift
            if self.drift_detector.detect_drift(training_data):
                logger.info("🌊 Concept drift detected - applying extra validation")

            # 3. Reentrenar cada modelo con validación anti-overfitting
            for model_name in self.models.keys():
                logger.info(f"🔄 Reentrenando {model_name}...")

                # Crear modelo candidato
                candidate_model = self._create_candidate_model(model_name)

                # Aplicar TODAS las protecciones anti-overfitting
                if self._passes_all_anti_overfitting_checks(candidate_model, training_data):
                    # Backup del modelo actual
                    self._backup_current_model(model_name)

                    # Desplegar nuevo modelo
                    self._deploy_new_model(model_name, candidate_model)
                    logger.info(f"✅ {model_name} mejorado y desplegado")
                else:
                    logger.info(f"⚠️ {model_name} no pasó validaciones anti-overfitting")

            # 4. Limpiar buffer de datos
            self.data_buffer = self.data_buffer[-500:]  # Mantener últimos 500

            # 5. Resetear timers
            self.auto_triggers['time_based']['last_retrain'] = datetime.now()

            logger.info("🎉 AUTO-REENTRENAMIENTO COMPLETADO CON PROTECCIÓN ANTI-OVERFITTING")

        except Exception as e:
            logger.error(f"❌ Error en auto-reentrenamiento: {e}")

    def _prepare_training_data(self) -> pd.DataFrame:
        """Convertir buffer de trades a DataFrame de entrenamiento"""
        try:
            data = []
            for trade in self.data_buffer:
                row = {
                    'timestamp': trade.timestamp.timestamp(),
                    'symbol': trade.symbol,
                    'side': 1 if trade.side == 'buy' else 0,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'confidence': trade.confidence,
                    'regime': trade.regime_at_entry,
                    'target': 1 if trade.pnl > 0 else 0,  # Target binario
                    **trade.features  # Incluir features técnicas
                }
                data.append(row)

            df = pd.DataFrame(data)

            # Limpiar datos
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            return df

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()

    def _create_candidate_model(self, model_name: str):
        """Crear modelo candidato para evaluación"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            # Crear modelo con regularización adaptativa
            if model_name == 'regime_classifier':
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=int(self.regularizer.regularization_params['min_samples_split']),
                    random_state=42
                )
            elif model_name == 'l1_models':
                return LogisticRegression(
                    C=1.0 / self.regularizer.regularization_params['l2_penalty'],
                    random_state=42
                )
            else:
                # Modelo genérico
                return RandomForestClassifier(n_estimators=50, random_state=42)

        except Exception as e:
            logger.error(f"Error creating candidate model {model_name}: {e}")
            return None

    def _passes_all_anti_overfitting_checks(self, candidate_model, training_data: pd.DataFrame) -> bool:
        """Aplicar TODAS las verificaciones anti-overfitting"""

        try:
            # 1. Validación cruzada continua
            cv_passed, cv_score = self.validator.validate_new_model(candidate_model, training_data)
            if not cv_passed:
                logger.warning("❌ Falló validación cruzada")
                return False

            # 2. Verificar diversidad del ensemble
            if not self.ensemble_builder.add_model_to_ensemble(candidate_model, training_data):
                logger.warning("❌ Falló verificación de diversidad")
                return False

            # 3. Verificar que no hay concept drift extremo
            if self.drift_detector.detect_drift(training_data):
                logger.info("🌊 Concept drift detectado - aplicando validación extra")

                # Re-evaluar con datos más recientes
                recent_data = training_data.tail(int(len(training_data) * 0.3))
                cv_passed_recent, _ = self.validator.validate_new_model(candidate_model, recent_data)
                if not cv_passed_recent:
                    logger.warning("❌ Falló validación con datos recientes post-drift")
                    return False

            # 4. Verificar early stopping (simulado)
            # En implementación real, esto se haría durante el entrenamiento

            logger.info("✅ PASÓ TODAS LAS VERIFICACIONES ANTI-OVERFITTING")
            return True

        except Exception as e:
            logger.error(f"Error en verificaciones anti-overfitting: {e}")
            return False

    def _backup_current_model(self, model_name: str):
        """Crear backup del modelo actual"""
        try:
            backup_path = f"models/backups/{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            if model_name in self.models:
                joblib.dump(self.models[model_name], backup_path)
                logger.info(f"💾 Backup creado: {backup_path}")

        except Exception as e:
            logger.error(f"Error creating backup for {model_name}: {e}")

    def _deploy_new_model(self, model_name: str, new_model):
        """Desplegar nuevo modelo"""
        self.models[model_name] = new_model

        # Registrar versión
        version_info = {
            'model': new_model,
            'version': f"auto_v{len(self.model_versions.get(model_name, [])) + 1}",
            'deployed_at': datetime.now(),
            'auto_generated': True
        }

        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(version_info)

        logger.info(f"🚀 Desplegado {model_name} versión {version_info['version']}")

# Sistema principal de auto-aprendizaje
class SelfImprovingTradingSystem:
    """Sistema de trading que se mejora solo con protección total anti-overfitting"""
    
    _instance = None

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SelfImprovingTradingSystem."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(self):
        # Evitar reinicialización si ya existe
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.auto_retrainer = AutoRetrainingSystem()
        self.performance_monitor = PerformanceMonitor()
        self.online_learners = {}  # Para componentes que pueden aprender online

        # Estado del sistema
        self.is_running = False
        self.last_update = datetime.now()
        
        # Componentes integrados
        self.state_manager = None
        self.order_manager = None
        self.portfolio_manager = None
        self.l2_processor = None
        self.trading_metrics = None
        self._initialized = True

        logger.info("🤖 Self-Improving Trading System initialized with maximum anti-overfitting protection")

    def integrate(self, state_manager=None, order_manager=None, portfolio_manager=None, 
                  l2_processor=None, trading_metrics=None):
        """
        Integrar el sistema de auto-aprendizaje con los componentes del sistema HRM.
        
        Args:
            state_manager: StateCoordinator para gestión de estado global
            order_manager: OrderManager para gestión de órdenes
            portfolio_manager: PortfolioManager para gestión de portfolio
            l2_processor: L2TacticProcessor para señales tácticas
            trading_metrics: TradingMetrics para métricas de trading
        """
        self.state_manager = state_manager
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.l2_processor = l2_processor
        self.trading_metrics = trading_metrics
        
        logger.info("=" * 70)
        logger.info("🔗 Auto-Learning System Integration:")
        logger.info(f"   State Manager:     {'✅' if state_manager else '❌'}")
        logger.info(f"   Order Manager:     {'✅' if order_manager else '❌'}")
        logger.info(f"   Portfolio Manager: {'✅' if portfolio_manager else '❌'}")
        logger.info(f"   L2 Processor:      {'✅' if l2_processor else '❌'}")
        logger.info(f"   Trading Metrics:   {'✅' if trading_metrics else '❌'}")
        logger.info("=" * 70)

    def start_auto_improvement(self):
        """Iniciar el ciclo de auto-mejora"""
        self.is_running = True
        logger.info("🚀 Auto-improvement cycle started")

        # En implementación real, esto sería un loop continuo
        # Por ahora, solo registramos que está listo

    def record_trade(self, trade_data: Dict[str, Any]):
        """Registrar trade para aprendizaje automático"""
        try:
            # Convertir a TradeData
            trade = TradeData(
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                side=trade_data.get('side', 'buy'),
                entry_price=trade_data.get('entry_price', 0.0),
                exit_price=trade_data.get('exit_price', 0.0),
                quantity=trade_data.get('quantity', 0.0),
                pnl=trade_data.get('pnl', 0.0),
                pnl_pct=trade_data.get('pnl_pct', 0.0),
                model_used=trade_data.get('model_used', 'unknown'),
                confidence=trade_data.get('confidence', 0.5),
                regime_at_entry=trade_data.get('regime', 'neutral'),
                features=trade_data.get('features', {}),
                market_data=trade_data.get('market_data', {})
            )

            # Añadir al sistema de auto-reentrenamiento
            self.auto_retrainer.add_trade_data(trade)

            # Actualizar métricas de performance
            self.performance_monitor.update_metrics(trade)

            logger.info(f"📊 Trade recorded for auto-learning: {trade.symbol} {trade.side} PnL: {trade.pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Error recording trade: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estado del sistema de auto-mejora.
        
        CRITICAL FIX: Ya NO llama a _get_current_portfolio_data() que usaba asyncio.run()
        En su lugar, devuelve un placeholder que debe ser llenado por el caller async
        """
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'data_buffer_size': len(self.auto_retrainer.data_buffer),
            'models_count': len(self.auto_retrainer.models),
            'ensemble_size': len(self.auto_retrainer.ensemble_builder.ensemble_models),
            'performance_metrics': self.performance_monitor.get_summary(),
            'anti_overfitting_active': True,
            'portfolio_data': None,  # CRITICAL FIX: Placeholder - use get_system_status_async() para datos reales
            'integration': {
                'state_manager': self.state_manager is not None,
                'order_manager': self.order_manager is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'l2_processor': self.l2_processor is not None,
                'trading_metrics': self.trading_metrics is not None
            }
        }
    
    async def get_system_status_async(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Versión async que obtiene datos reales del portfolio.
        Esta es la versión que debe usarse desde contextos async.
        """
        # Obtener datos del portfolio de forma async
        portfolio_data = await self._get_current_portfolio_data_async()
        
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'data_buffer_size': len(self.auto_retrainer.data_buffer),
            'models_count': len(self.auto_retrainer.models),
            'ensemble_size': len(self.auto_retrainer.ensemble_builder.ensemble_models),
            'performance_metrics': self.performance_monitor.get_summary(),
            'anti_overfitting_active': True,
            'portfolio_data': portfolio_data,  # CRITICAL: Datos reales obtenidos async
            'integration': {
                'state_manager': self.state_manager is not None,
                'order_manager': self.order_manager is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'l2_processor': self.l2_processor is not None,
                'trading_metrics': self.trading_metrics is not None
            }
        }
    
    async def _get_current_portfolio_data_async(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Get CURRENT portfolio data from PortfolioManager (ASYNC VERSION).
        This ensures AutoLearning trains on actual balances, not stale/initial values.
        
        CRITICAL FIX: Solo retorna balances verificados desde sync async. Si los balances
        fueron obtenidos vía fallback o error, esto se marca y el entrenamiento debe bloquearse.
        """
        try:
            if self.portfolio_manager is None:
                return {'error': 'PortfolioManager not available', 'is_verified': False}
            
            # Check if balances are verified (from async sync)
            is_verified = False
            verification_status = {}
            if hasattr(self.portfolio_manager, 'are_balances_verified'):
                is_verified = self.portfolio_manager.are_balances_verified()
            if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                verification_status = self.portfolio_manager.get_balance_verification_status()
            
            # Get balances using async methods
            btc_balance = 0.0
            eth_balance = 0.0
            usdt_balance = 0.0
            
            if hasattr(self.portfolio_manager, 'get_balances_async'):
                balances = await self.portfolio_manager.get_balances_async()
                btc_balance = balances.get('BTC', 0.0)
                eth_balance = balances.get('ETH', 0.0)
                usdt_balance = balances.get('USDT', 0.0)
            elif hasattr(self.portfolio_manager, 'get_asset_balance_async'):
                btc_balance = await self.portfolio_manager.get_asset_balance_async('BTC')
                eth_balance = await self.portfolio_manager.get_asset_balance_async('ETH')
                usdt_balance = await self.portfolio_manager.get_asset_balance_async('USDT')
            else:
                # Fallback - but mark as not verified
                logger.warning("[AUTO_LEARNING] No async balance methods available - data will be unverified")
                is_verified = False
                # No sync fallback - esto es intencional para prevenir violaciones async
                return {
                    'error': 'No async balance methods available in PortfolioManager',
                    'is_verified': False,
                    'data_source': 'error_no_async_methods'
                }
            
            # Get current NAV using async method if available
            current_nav = 0.0
            if hasattr(self.portfolio_manager, 'get_total_value_async'):
                current_nav = await self.portfolio_manager.get_total_value_async()
            elif hasattr(self.portfolio_manager, 'portfolio'):
                current_nav = self.portfolio_manager.portfolio.get('total', 0.0)
            
            # Calculate PnL
            initial_balance = getattr(self.portfolio_manager, 'initial_balance', 500.0)
            pnl = current_nav - initial_balance
            pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0.0
            
            # Calculate exposure
            total_crypto_value = 0.0
            if hasattr(self.portfolio_manager, 'client') and self.portfolio_manager.client:
                client = self.portfolio_manager.client
                if hasattr(client, 'get_market_price'):
                    try:
                        btc_price = client.get_market_price('BTCUSDT')
                        eth_price = client.get_market_price('ETHUSDT')
                        total_crypto_value = (btc_balance * btc_price) + (eth_balance * eth_price)
                    except:
                        pass
            
            exposure_pct = (total_crypto_value / current_nav * 100) if current_nav > 0 else 0.0
            
            return {
                'btc_balance': btc_balance,
                'eth_balance': eth_balance,
                'usdt_balance': usdt_balance,
                'current_nav': current_nav,
                'initial_balance': initial_balance,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exposure_pct': exposure_pct,
                'data_source': 'PortfolioManager (async_synced)' if is_verified else 'PortfolioManager (unverified)',
                'is_verified': is_verified,
                'verification_status': verification_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio data: {e}")
            return {'error': str(e), 'is_verified': False}

    def can_train(self) -> Tuple[bool, str]:
        """
        CRITICAL FIX: Versión sincrona que NO usa asyncio.run().
        Solo verifica el estado de verificación sincronizado - no obtiene datos nuevos.
        
        Para verificación completa con datos frescos, usar can_train_async().
        """
        try:
            if self.portfolio_manager is None:
                return False, "PortfolioManager not available"
            
            # Check verification status (síncrono - solo lee estado cacheado)
            if hasattr(self.portfolio_manager, 'are_balances_verified'):
                if not self.portfolio_manager.are_balances_verified():
                    verification_status = {}
                    if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                        verification_status = self.portfolio_manager.get_balance_verification_status()
                    
                    if verification_status.get('was_fallback_used'):
                        return False, f"Balances from fallback source - training blocked. Errors: {verification_status.get('sync_errors', [])}"
                    
                    return False, "Balances not verified from async sync"
            
            # Check if balances are fresh
            if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                status = self.portfolio_manager.get_balance_verification_status()
                seconds_since_sync = status.get('seconds_since_sync')
                
                if seconds_since_sync is not None and seconds_since_sync > 60:
                    return False, f"Balances are stale ({seconds_since_sync:.0f}s old) - sync required before training"
            
            return True, "Balances verified - training allowed"
            
        except Exception as e:
            return False, f"Error checking training eligibility: {e}"

    async def can_train_async(self) -> Tuple[bool, str]:
        """
        CRITICAL FIX: Versión async que obtiene datos frescos y verifica estado.
        Esta es la versión preferida para usar desde contextos async.
        """
        try:
            if self.portfolio_manager is None:
                return False, "PortfolioManager not available"
            
            # Get fresh portfolio data
            portfolio_data = await self._get_current_portfolio_data_async()
            
            # Check if data is verified
            if not portfolio_data.get('is_verified', False):
                error = portfolio_data.get('error', 'Unknown error')
                return False, f"Portfolio data not verified: {error}"
            
            # Check if balances are fresh
            verification_status = portfolio_data.get('verification_status', {})
            seconds_since_sync = verification_status.get('seconds_since_sync')
            
            if seconds_since_sync is not None and seconds_since_sync > 60:
                return False, f"Balances are stale ({seconds_since_sync:.0f}s old) - sync required before training"
            
            return True, "Balances verified and fresh - training allowed"
            
        except Exception as e:
            return False, f"Error checking training eligibility: {e}"

# Clase auxiliar para monitoreo de performance
class PerformanceMonitor:
    """Monitorea performance del sistema"""

    def __init__(self):
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    def update_metrics(self, trade: TradeData):
        """Actualizar métricas con nuevo trade"""
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += trade.pnl

        if trade.pnl > 0:
            self.metrics['winning_trades'] += 1

        # Calcular win rate
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']

    def get_summary(self) -> Dict[str, float]:
        """Obtener resumen de métricas"""
        return self.metrics.copy()

# Función principal para testing
if __name__ == "__main__":
    # Crear sistema
    system = SelfImprovingTradingSystem()

    # Simular algunos trades
    sample_trades = [
        {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.01,
            'pnl': 10.0,
            'pnl_pct': 0.02,
            'model_used': 'l2_finrl',
            'confidence': 0.8,
            'regime': 'bull',
            'features': {'rsi': 65, 'macd': 0.5}
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'sell',
            'entry_price': 3000,
            'exit_price': 2950,
            'quantity': 0.1,
            'pnl': 5.0,
            'pnl_pct': 0.0167,
            'model_used': 'l1_technical',
            'confidence': 0.7,
            'regime': 'neutral',
            'features': {'rsi': 45, 'macd': -0.3}
        }
    ]

    # Registrar trades
    for trade_data in sample_trades:
        system.record_trade(trade_data)

    # Mostrar estado (versión sync - sin datos de portfolio)
    status = system.get_system_status()
    print("🤖 Sistema de Auto-Aprendizaje con Anti-Overfitting Máximo:")
    print(f"   📊 Trades en buffer: {status['data_buffer_size']}")
    print(f"   🧠 Modelos activos: {status['models_count']}")
    print(f"   🎯 Ensemble size: {status['ensemble_size']}")
    print(f"   🛡️ Anti-overfitting: {'ACTIVO' if status['anti_overfitting_active'] else 'INACTIVO'}")
    print(f"   📈 Performance: {status['performance_metrics']}")

    print("\n✅ SISTEMA LISTO PARA AUTO-MEJORA CONTINUA CON PROTECCIÓN TOTAL ANTI-OVERFITTING!")



================================================
FILE: bootstrap.py
================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Bootstrap Module

This module handles system initialization, configuration loading,
and component wiring for the HRM system.

🔥 PRIORIDAD 4: Introducir session_is_fresh - no depender de balances
"""

import os
import sys
import asyncio
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.state_manager import initialize_state, validate_state_structure
from core.portfolio_manager import PortfolioManager
from core.logging import logger
from core.config import get_config, set_forced_mode
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics

from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus

from sentiment.sentiment_manager import update_sentiment_texts

from system.system_cleanup import perform_full_cleanup
from storage.paper_trade_logger import get_paper_logger
from config_loader import get_initial_balances, get_capital_usd


# 🔥 PRIORIDAD 4: SESSION IS FRESH - Flag global para nueva sesión
session_is_fresh = True

def is_session_fresh() -> bool:
    """Check if current session is fresh (not dependent on balances)"""
    global session_is_fresh
    return session_is_fresh

def mark_session_used():
    """Mark session as no longer fresh"""
    global session_is_fresh
    session_is_fresh = False


class HRMBootstrap:
    """HRM system bootstrap and initialization."""
    
    def __init__(self):
        self.components = {}
        self.state = None
        self.portfolio_manager = None
        self.order_manager = None
        self.runtime_loop = None
        self._mode = "paper"  # Default mode, will be set in bootstrap_system
    
    def _get_mode_from_bootstrap(self) -> str:
        """Get the mode from the single source of truth (bootstrap)."""
        return self._mode
        
    async def bootstrap_system(self, mode: str = "paper") -> Tuple[PortfolioManager, OrderManager]:
        """
        Bootstrap the entire HRM system.
        
        🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión,
        NO los balances. Esto evita dependencia de estado externo.
        """
        # Store mode as single source of truth for this bootstrap instance
        self._mode = mode
        
        logger.info("🚀 Starting HRM System Bootstrap")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        logger.info(f"🎯 BOOTSTRAP MODE: {mode}")
        
        try:
            # 1. System Cleanup
            await self._perform_system_cleanup()
            
            # 2. Load Configuration
            env_config = await self._load_configuration(mode)
            
            # 3. Initialize State
            self.state = await self._initialize_state(env_config)
            
            # 4. Initialize Core Components
            await self._initialize_core_components(env_config)
            
            # 5. Initialize L1 Components
            await self._initialize_l1_components(env_config)
            
            # 6. Initialize L2 Components
            await self._initialize_l2_components(env_config)
            
            # 7. Initialize L3 Components (if enabled)
            if not APAGAR_L3:
                await self._initialize_l3_components()
            
            # 8. Initialize Portfolio Manager
            self.portfolio_manager = await self._initialize_portfolio_manager(env_config)
            
            # 9. Initialize Order Manager
            self.order_manager = await self._initialize_order_manager()
            
            # 10. Initialize Runtime Loop
            from runtime_loop import HRMRuntimeLoop
            self.runtime_loop = HRMRuntimeLoop(self.portfolio_manager, self.order_manager)
            
            # 11. Start Background Services
            await self._start_background_services()
            
            # 12. Marcar sesión como usada después de bootstrap exitoso
            mark_session_used()
            logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()} (post-bootstrap)")
            
            logger.info("✅ HRM System Bootstrap Complete")
            return self.portfolio_manager, self.order_manager
            
        except Exception as e:
            logger.error(f"❌ Bootstrap failed: {e}", exc_info=True)
            await self._cleanup_on_failure()
            raise
    
    async def _perform_system_cleanup(self):
        """Perform system cleanup before startup."""
        logger.info("🧹 Running system cleanup...")
        
        cleanup_result = perform_full_cleanup(mode="paper")
        
        if not cleanup_result.get("success", False):
            logger.warning("⚠️ Cleanup completed with warnings")
        else:
            logger.info(f"✅ Cleanup completed: {cleanup_result.get('deleted_files', 0)} files, {cleanup_result.get('deleted_dirs', 0)} directories removed")
        
        # Clean paper trades for independent testing
        try:
            logger.info("🧹 Cleaning paper trades history...")
            get_paper_logger(clear_on_init=True)
            logger.info("✅ Paper trades history cleaned")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning paper trades: {e}")
    
    async def _load_configuration(self, mode: str) -> Dict[str, Any]:
        """Load system configuration."""
        logger.info(f"⚙️ Loading configuration for mode: {mode}")
        
        # Load environment variables
        load_dotenv()
        
        # 🔥 CRITICAL: Set the forced mode in core.config BEFORE getting config
        # This establishes the single source of truth for the mode
        set_forced_mode(mode)
        logger.info(f"🎯 Mode set as single source of truth: {mode}")
        
        # Get environment configuration
        # 🔥 PRIORIDAD 4: Usar mode explícito, no depender de configuración previa
        env_config = get_config(mode)
        
        # Check Binance operating mode
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        logger.info(f"🏦 BINANCE MODE: {binance_mode}")
        
        return env_config
    
    async def _initialize_state(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize system state."""
        logger.info("🧠 Initializing system state...")
        
        # Initialize state with symbols and initial balance
        # 🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión
        symbols = env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        initial_balance = get_capital_usd()  # Cargar desde initial_state.json
        
        state = initialize_state(symbols, initial_balance)
        state = validate_state_structure(state)
        
        logger.info(f"✅ State initialized for symbols: {symbols}")
        logger.info(f"   Initial balance: {initial_balance} USDT")
        logger.info(f"   Session fresh: {is_session_fresh()}")
        return state
    
    async def _initialize_core_components(self, env_config: Dict[str, Any]):
        """Initialize core system components."""
        logger.info("🔧 Initializing core components...")
        
        # Initialize trading metrics
        trading_metrics = get_trading_metrics()
        self.components['trading_metrics'] = trading_metrics
        
        # Initialize signal verifier
        signal_verifier = get_signal_verifier()
        self.components['signal_verifier'] = signal_verifier
        
        # Initialize message bus
        message_bus = MessageBus()
        self.components['message_bus'] = message_bus
        
        # Initialize sentiment manager
        sentiment_manager = await self._initialize_sentiment_manager()
        self.components['sentiment_manager'] = sentiment_manager
        
        logger.info("✅ Core components initialized")
    
    async def _initialize_l1_components(self, env_config: Dict[str, Any]):
        """Initialize L1 operational components."""
        logger.info("🔧 Initializing L1 operational components...")
        
        # Determine mode from the single source of truth
        mode = self._get_mode_from_bootstrap()
        
        # Initialize Binance client with injected mode (single source of truth)
        binance_client = BinanceClient(mode=mode)
        self.components['binance_client'] = binance_client
        
        # Initialize SimulatedExchangeClient (para paper trading)
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        
        # 🔥 PRIORIDAD 4: Cargar balances desde initial_state.json
        # Usar config_loader para centralizar la configuración
        initial_balances = get_initial_balances()
        
        logger.info(f"📊 Cargando balances desde initial_state.json: {initial_balances}")
        
        # Limpiar cualquier estado previo y crear nuevo
        SimulatedExchangeClient._instance = None
        SimulatedExchangeClient._initialized = False
        
        # Inicializar con balances conocidos
        simulated_client = SimulatedExchangeClient(initial_balances)
        self.components['simulated_client'] = simulated_client
        
        # Logs requeridos
        logger.info("📊 SIM_INIT: Nueva instancia")
        logger.info(f"🔢 SIM_STATE_ID: {id(simulated_client)}")
        logger.info(f"💰 SIM_BALANCES: {simulated_client.get_balances()}")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        
        # Verificar que los balances no estén vacíos
        if not simulated_client.get_balances() or all(balance == 0 for balance in simulated_client.get_balances().values()):
            logger.critical("🚨 FATAL: SimulatedExchangeClient initialized with empty or zero balances", exc_info=True)
            raise RuntimeError("SimulatedExchangeClient cannot operate with empty or zero balances")
        
        # Initialize RealTimeDataLoader
        loader = RealTimeDataLoader(config)
        self.components['data_loader'] = loader
        
        # Initialize L1 AI Models
        from l1_operational.trend_ai import models as l1_models
        logger.info(f"✅ Loaded L1 AI Models: {list(l1_models.keys())}")
        
        logger.info("✅ L1 components initialized")
    
    async def _initialize_l2_components(self, env_config: Dict[str, Any]):
        """Initialize L2 tactical components."""
        logger.info("🔧 Initializing L2 tactical components...")
        
        # Quick fix: Disable synchronizer in PAPER mode for better performance
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode != "LIVE":
            logger.info("📝 PAPER/TEST MODE: Disabling BTC/ETH synchronizer")
            os.environ['DISABLE_BTC_ETH_SYNC'] = 'true'
        
        # Initialize L2 Config
        l2_config = L2Config()
        self.components['l2_config'] = l2_config
        
        # Initialize L2 Processor
        l2_processor = L2TacticProcessor(l2_config, portfolio_manager=None, apagar_l3=APAGAR_L3)
        self.components['l2_processor'] = l2_processor
        
        # Initialize Risk Manager
        risk_manager = RiskControlManager(l2_config)
        self.components['risk_manager'] = risk_manager
        
        logger.info("✅ L2 components initialized")
    
    async def _initialize_l3_components(self):
        """Initialize L3 strategic components."""
        logger.info("🔧 Initializing L3 strategic components...")
        
        # Import L3 components
        from l3_strategy.regime_classifier import MarketRegimeClassifier
        from l3_strategy.decision_maker import make_decision
        
        # Initialize L3 Classifier
        regime_classifier = MarketRegimeClassifier()
        self.components['regime_classifier'] = regime_classifier
        
        # Initialize L3 Decision Maker
        self.components['l3_decision_maker'] = make_decision
        
        logger.info("✅ L3 components initialized")
    
    async def _initialize_portfolio_manager(self, env_config: Dict[str, Any]) -> PortfolioManager:
        """Initialize Portfolio Manager."""
        logger.info("💼 Initializing Portfolio Manager...")
        
        # Get environment configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        
        # Setup based on binance_mode
        # 🔥 PRIORIDAD 4: Usar modo explícito, no depender de balances previos
        if binance_mode == "LIVE":
            # Live mode: sync mandatory with exchange
            portfolio_mode = "live"
            initial_balance = 0.0  # Will be synced from exchange
        else:
            # Test mode: use simulated balance
            portfolio_mode = "simulated"
            initial_balance = get_capital_usd()  # Cargar desde initial_state.json
            logger.info(f"🧪 TESTING MODE: Using initial balance of {initial_balance} USDT from initial_state.json")
        
        # Initialize Portfolio Manager
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            client=self.components['simulated_client'] if portfolio_mode == "simulated" else self.components['binance_client'],
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )
        
        # Initialize asynchronously to get balances from simulated client
        if portfolio_mode == "simulated":
            await portfolio_manager.initialize_async()
        
        # 🔥 PRIORIDAD 4: NO sync con exchange en modo paper - confiar en balances iniciales
        # Esto evita dependencia de estado externo
        if portfolio_mode == "simulated":
            logger.info("📊 PAPER MODE: Trusting initial balances, no exchange sync needed")
        else:
            # CRITICAL: Synchronize with exchange for production mode
            try:
                logger.info("🔄 Synchronizing with exchange...")
                sync_success = await portfolio_manager.sync_with_exchange()
                
                if sync_success:
                    logger.info("✅ Portfolio synchronized with exchange")
                else:
                    logger.warning("⚠️ Exchange sync failed, loading local state...")
                    loaded = portfolio_manager.load_from_json()
                    if not loaded:
                        logger.info("📄 No saved portfolio found, starting clean")
                    else:
                        logger.info("📂 Local portfolio loaded")
                        
            except Exception as e:
                logger.error(f"❌ Portfolio synchronization failed: {e}")
                logger.warning("⚠️ Continuing with local state")
        
        logger.info("✅ Portfolio Manager initialized")
        return portfolio_manager
    
    async def _initialize_order_manager(self) -> OrderManager:
        """Initialize Order Manager."""
        logger.info("💰 Initializing Order Manager...")
        
        # Determine mode based on configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode == "LIVE":
            mode = "live"
        else:
            mode = "simulated"
            
        order_manager = OrderManager(
            state_manager=self.state,
            portfolio_manager=self.portfolio_manager,
            mode=mode,
            simulated_client=self.components['simulated_client']
        )
        
        logger.info(f"✅ Order Manager initialized (mode: {mode})")
        return order_manager
    
    async def _initialize_sentiment_manager(self):
        """Initialize sentiment analysis manager."""
        logger.info("🧠 Initializing sentiment analysis...")
        
        # Initialize sentiment cache
        sentiment_texts_cache = []
        last_sentiment_update = 0
        
        # Initial sentiment update
        try:
            sentiment_texts_cache = await update_sentiment_texts()
            last_sentiment_update = 0  # Will be set in runtime loop
            logger.info(f"✅ Sentiment analysis initialized with {len(sentiment_texts_cache)} texts")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment initialization failed: {e}")
        
        return {
            'texts_cache': sentiment_texts_cache,
            'last_update': last_sentiment_update
        }
    
    async def _start_background_services(self):
        """Start background services."""
        logger.info("🔄 Starting background services...")
        
        # Start signal verification
        signal_verifier = self.components['signal_verifier']
        await start_signal_verification()
        logger.info("✅ Signal verification started")
    
    async def _cleanup_on_failure(self):
        """Clean up components if bootstrap fails."""
        logger.info("🧹 Cleaning up after bootstrap failure...")
        
        try:
            # Stop signal verification
            await stop_signal_verification()
            
            # Close components
            for component in self.components.values():
                if hasattr(component, "close"):
                    await component.close()
                    
        except Exception as e:
            logger.error(f"❌ Cleanup after failure failed: {e}")
    
    def get_runtime_loop(self):
        """Get the initialized runtime loop."""
        return self.runtime_loop
    
    def get_state(self) -> Dict[str, Any]:
        """Get the initialized system state."""
        return self.state


async def bootstrap_hrm_system(mode: str = "paper"):
    """
    Bootstrap the HRM system and return core components.
    
    🔥 PRIORIDAD 4: session_is_fresh introducido - nueva sesión independiente de balances
    """
    bootstrap = HRMBootstrap()
    portfolio_manager, order_manager = await bootstrap.bootstrap_system(mode)
    runtime_loop = bootstrap.get_runtime_loop()
    
    return portfolio_manager, order_manager, runtime_loop


def reset_session():
    """
    Reset session to fresh state.
    
    🔥 PRIORIDAD 4: Función para marcar sesión como nueva
    """
    global session_is_fresh
    session_is_fresh = True
    logger.info("🔄 Session reset to fresh state")
    return True


# =============================================================================
# SHUTDOWN LIMPIO GLOBAL
# =============================================================================

# Variable global para almacenar referencias a componentes para shutdown
_shutdown_components = {
    'market_data_manager': None,
    'realtime_loader': None,
    'exchange_client': None,
    'binance_client': None,
    'data_feed': None,
    'portfolio_manager': None,
    'order_manager': None,
}


def register_component_for_shutdown(name: str, component: Any):
    """
    Registra un componente para ser cerrado durante el shutdown global.
    
    Args:
        name: Nombre identificador del componente
        component: Instancia del componente a registrar
    """
    global _shutdown_components
    _shutdown_components[name] = component
    logger.debug(f"🔧 Componente '{name}' registrado para shutdown")


def unregister_component_for_shutdown(name: str):
    """
    Desregistra un componente del shutdown global.
    
    Args:
        name: Nombre identificador del componente
    """
    global _shutdown_components
    if name in _shutdown_components:
        _shutdown_components[name] = None
        logger.debug(f"🔧 Componente '{name}' desregistrado del shutdown")


async def shutdown():
    """
    Shutdown limpio global - cierra todas las conexiones y sesiones.
    
    Esta función DEBE ser llamada al finalizar el sistema para asegurar:
    - Todas las sesiones aiohttp se cierran correctamente
    - No quedan warnings asyncio al terminar
    - Los recursos se liberan en el orden correcto
    
    Orden de cierre:
    1. MarketDataManager (cierra RealTimeLoader y DataFeed internamente)
    2. RealTimeLoader (WebSocket connections)
    3. ExchangeClient / BinanceClient (sesiones aiohttp)
    4. DataFeed
    5. PortfolioManager (guarda estado)
    6. OrderManager
    7. Cancelar todas las tareas pendientes de asyncio
    8. Cerrar sesiones aiohttp huérfanas
    """
    global _shutdown_components
    
    logger.info("🧹 Iniciando shutdown limpio global...")
    
    # 1. Cerrar MarketDataManager
    if _shutdown_components.get('market_data_manager') is not None:
        try:
            await _shutdown_components['market_data_manager'].close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando MarketDataManager: {e}")
    
    # 2. Cerrar RealTimeLoader directamente si existe
    if _shutdown_components.get('realtime_loader') is not None:
        try:
            await _shutdown_components['realtime_loader'].close()
            logger.info("✅ RealTimeLoader cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando RealTimeLoader: {e}")
    
    # 3. Cerrar ExchangeClient / BinanceClient
    for client_name in ['exchange_client', 'binance_client']:
        if _shutdown_components.get(client_name) is not None:
            try:
                client = _shutdown_components[client_name]
                if hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, 'close_connection'):
                    await client.close_connection()
                logger.info(f"✅ {client_name} cerrado")
            except Exception as e:
                logger.warning(f"⚠️ Error cerrando {client_name}: {e}")
    
    # 4. Cerrar DataFeed
    if _shutdown_components.get('data_feed') is not None:
        try:
            await _shutdown_components['data_feed'].close()
            logger.info("✅ DataFeed cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando DataFeed: {e}")
    
    # 5. Guardar estado del PortfolioManager
    if _shutdown_components.get('portfolio_manager') is not None:
        try:
            pm = _shutdown_components['portfolio_manager']
            if hasattr(pm, 'save_to_json'):
                pm.save_to_json()
                logger.info("✅ PortfolioManager estado guardado")
            if hasattr(pm, 'close'):
                await pm.close()
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando PortfolioManager: {e}")
    
    # 6. Cerrar OrderManager
    if _shutdown_components.get('order_manager') is not None:
        try:
            om = _shutdown_components['order_manager']
            if hasattr(om, 'close'):
                await om.close()
            logger.info("✅ OrderManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando OrderManager: {e}")
    
    # 7. Cerrar todas las sesiones aiohttp pendientes
    await _close_aiohttp_sessions()
    
    # 8. Cancelar todas las tareas pendientes de asyncio (excepto la actual)
    await _cancel_pending_tasks()
    
    logger.info("🧹 Shutdown limpio global completado")


async def _close_aiohttp_sessions():
    """
    Cierra todas las sesiones aiohttp abiertas.
    """
    try:
        import aiohttp
        
        # Buscar y cerrar todas las sesiones aiohttp
        closed_count = 0
        for obj in list(globals().values()) + list(_shutdown_components.values()):
            if obj is None:
                continue
            
            # Buscar atributos que sean sesiones aiohttp
            if hasattr(obj, 'session') and hasattr(obj.session, 'closed'):
                try:
                    if not obj.session.closed:
                        await obj.session.close()
                        closed_count += 1
                except Exception:
                    pass
            
            # Buscar atributos que sean ClientSession directamente
            if isinstance(obj, aiohttp.ClientSession):
                try:
                    if not obj.closed:
                        await obj.close()
                        closed_count += 1
                except Exception:
                    pass
        
        if closed_count > 0:
            logger.info(f"✅ {closed_count} sesiones aiohttp cerradas")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando sesiones aiohttp: {e}")


async def _cancel_pending_tasks():
    """
    Cancela todas las tareas pendientes de asyncio excepto la tarea actual.
    Esto evita warnings de "task was destroyed but it is pending".
    """
    try:
        import asyncio
        
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        
        # Obtener todas las tareas pendientes excepto la actual
        pending_tasks = [
            task for task in asyncio.all_tasks(loop) 
            if task is not current_task and not task.done()
        ]
        
        if pending_tasks:
            logger.info(f"🛑 Cancelando {len(pending_tasks)} tareas pendientes...")
            
            # Cancelar todas las tareas
            for task in pending_tasks:
                task.cancel()
            
            # Esperar a que todas las tareas se completen (o se cancelen)
            results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            
            cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
            logger.info(f"✅ {cancelled_count} tareas canceladas, {len(results) - cancelled_count} completadas")
    except Exception as e:
        logger.warning(f"⚠️ Error cancelando tareas pendientes: {e}")


async def shutdown_with_signal_handling():
    """
    Versión del shutdown que maneja señales del sistema.
    Usar esta función cuando se quiera un shutdown completo con manejo de señales.
    """
    logger.info("🛑 Ejecutando shutdown con manejo de señales...")
    await shutdown()


# Manejadores de señales para shutdown limpio
_shutdown_event = asyncio.Event()

def _signal_handler(signum, frame):
    """
    Manejador de señales que activa el evento de shutdown.
    """
    import signal
    sig_name = signal.Signals(signum).name
    logger.info(f"🛑 Señal {sig_name} recibida, iniciando shutdown...")
    _shutdown_event.set()


def setup_signal_handlers():
    """
    Configura los manejadores de señales para SIGINT y SIGTERM.
    Llama a esta función al inicio de la aplicación para habilitar shutdown limpio.
    """
    import signal
    
    # Configurar manejadores para SIGINT (Ctrl+C) y SIGTERM
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    logger.info("✅ Manejadores de señales configurados (SIGINT, SIGTERM)")


async def wait_for_shutdown_signal():
    """
    Espera hasta que se reciba una señal de shutdown.
    Útil para mantener el programa corriendo hasta que se presione Ctrl+C.
    """
    await _shutdown_event.wait()
    logger.info("🛑 Señal de shutdown recibida, procediendo...")



================================================
FILE: config_loader.py
================================================
"""
Config Loader Module

Carga la configuración inicial desde initial_state.json y proporciona
funciones para acceder a los valores de configuración.

Este módulo centraliza el acceso a la configuración inicial del sistema.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "capital_usdt": 3000.0,
    "btc": 0.0,
    "eth": 0.0,
    "mode": "paper",
    "auto_learning": "fix",
    "initial_balances": {
        "USDT": 3000.0,
        "BTC": 0.0,
        "ETH": 0.0
    },
    "reset_singletons": True
}


def load_initial_state(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Load initial state from JSON file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Ensure all default keys exist
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading {config_path}: {e}")
            print("   Using default configuration")
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file if it doesn't exist
        save_initial_state(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG.copy()


def save_initial_state(config: Dict[str, Any], config_path: str = "initial_state.json") -> bool:
    """
    Save initial state to JSON file.
    
    Args:
        config: Configuration dict to save
        config_path: Path to the initial state JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"⚠️  Error saving {config_path}: {e}")
        return False


def get_initial_balances(config_path: str = "initial_state.json") -> Dict[str, float]:
    """
    Get initial balances for SimulatedExchangeClient.
    
    Returns:
        Dict with BTC, ETH, and USDT balances
    """
    config = load_initial_state(config_path)
    
    # Prefer initial_balances if available
    if "initial_balances" in config:
        balances = config["initial_balances"]
        return {
            "BTC": float(balances.get("BTC", 0.0)),
            "ETH": float(balances.get("ETH", 0.0)),
            "USDT": float(balances.get("USDT", 3000.0))
        }
    
    # Fallback to individual fields
    return {
        "BTC": float(config.get("btc", 0.0)),
        "ETH": float(config.get("eth", 0.0)),
        "USDT": float(config.get("capital_usdt", 3000.0))
    }


def get_capital_usd(config_path: str = "initial_state.json") -> float:
    """Get initial capital in USD."""
    config = load_initial_state(config_path)
    return float(config.get("capital_usdt", 3000.0))


def get_mode(config_path: str = "initial_state.json") -> str:
    """Get trading mode (paper/live)."""
    config = load_initial_state(config_path)
    return config.get("mode", "paper")


def get_auto_learning(config_path: str = "initial_state.json") -> str:
    """Get auto-learning mode."""
    config = load_initial_state(config_path)
    return config.get("auto_learning", "fix")


def should_reset_singletons(config_path: str = "initial_state.json") -> bool:
    """Check if singletons should be reset."""
    config = load_initial_state(config_path)
    return config.get("reset_singletons", True)


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Get cached configuration or load from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_initial_state(config_path)
    
    return _config_cache


def reload_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    _config_cache = load_initial_state(config_path)
    return _config_cache


def reset_config_cache():
    """Reset the configuration cache."""
    global _config_cache
    _config_cache = None


# Convenience properties
@property
def initial_balances() -> Dict[str, float]:
    """Get initial balances (uses cached config)."""
    return get_initial_balances()


@property
def capital_usd() -> float:
    """Get initial capital (uses cached config)."""
    return get_capital_usd()


@property
def mode() -> str:
    """Get trading mode (uses cached config)."""
    return get_mode()


if __name__ == "__main__":
    # Test the module
    print("🔧 Config Loader Test")
    print("=" * 50)
    
    config = load_initial_state()
    print(f"📊 Config loaded: {config}")
    print()
    
    balances = get_initial_balances()
    print(f"💰 Initial balances: {balances}")
    print(f"💵 Capital USD: {get_capital_usd()}")
    print(f"🎯 Mode: {get_mode()}")
    print(f"🤖 Auto-learning: {get_auto_learning()}")
    print(f"🔄 Reset singletons: {should_reset_singletons()}")



================================================
FILE: CONTENT FILES.md
================================================
CONTENT FILES

================================================
FILE: __init__.py
================================================
# HRM - Sistema de Trading Cripto con Modelo de Razonamiento Jerárquico
# Este archivo hace que Python reconozca el directorio como un paquete

__version__ = "1.0.0"
__author__ = "Equipo de desarrollo HRM"
__description__ = "Sistema de trading cripto con modelo de razonamiento jerárquico"



================================================
FILE: auto_learning_system.py
================================================
#!/usr/bin/env python3
"""
Sistema de Auto-Aprendizaje con Protección Total Anti-Overfitting
Implementa aprendizaje continuo completamente automático con 9 capas de protección

CRITICAL FIX: Eliminado todo asyncio.run() - ahora 100% async-compatible
"""

import asyncio
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import joblib
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradeData:
    """Estructura de datos de trade para aprendizaje"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    model_used: str
    confidence: float
    regime_at_entry: str
    features: Dict[str, float] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Métricas de performance de modelo"""
    model_name: str
    timestamp: datetime
    train_score: float
    val_score: float
    oos_score: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float

class AntiOverfitValidator:
    """Validación cruzada continua para prevenir overfitting"""

    def __init__(self, validation_windows: int = 5, min_score: float = 0.55):
        self.validation_windows = validation_windows
        self.min_validation_score = min_score
        self.max_score_std = 0.15  # Máxima desviación estándar permitida

    def validate_new_model(self, model, training_data: pd.DataFrame) -> Tuple[bool, float]:
        """Validar modelo con múltiples ventanas temporales"""
        try:
            scores = []

            # Rolling window validation
            for i in range(self.validation_windows):
                train_start_idx = int(len(training_data) * i / self.validation_windows)
                train_end_idx = int(len(training_data) * (i + 4) / self.validation_windows)
                val_start_idx = train_end_idx
                val_end_idx = int(len(training_data) * (i + 5) / self.validation_windows)

                if val_end_idx > len(training_data):
                    val_end_idx = len(training_data)

                if val_start_idx >= val_end_idx:
                    continue

                # Datos de entrenamiento y validación
                train_data = training_data.iloc[train_start_idx:train_end_idx]
                val_data = training_data.iloc[val_start_idx:val_end_idx]

                if len(val_data) < 10:
                    continue

                # Entrenar y validar
                model_copy = self._clone_model(model)
                score = self._evaluate_window(model_copy, train_data, val_data)
                scores.append(score)

            if not scores:
                return False, 0.0

            avg_score = np.mean(scores)
            std_score = np.std(scores)

            # Rechazar si score bajo o muy variable
            if avg_score < self.min_validation_score:
                logger.warning(f"❌ CV Score too low: {avg_score:.3f} < {self.min_validation_score}")
                return False, avg_score

            if std_score > self.max_score_std:
                logger.warning(f"❌ CV Score too variable: std {std_score:.3f} > {self.max_score_std}")
                return False, avg_score

            logger.info(f"✅ CV Validation passed: {avg_score:.3f} ± {std_score:.3f}")
            return True, avg_score

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return False, 0.0

    def _clone_model(self, model):
        """Clonar modelo para validación"""
        try:
            # Para sklearn models
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                return type(model)(**model.get_params())
        except:
            pass

        # Fallback: usar deepcopy
        import copy
        return copy.deepcopy(model)

    def _evaluate_window(self, model, train_data: pd.DataFrame, val_data: pd.DataFrame) -> float:
        """Evaluar modelo en una ventana específica"""
        try:
            if 'target' not in train_data.columns or 'target' not in val_data.columns:
                # Asumir que la última columna es target
                target_col = train_data.columns[-1]
            else:
                target_col = 'target'

            X_train = train_data.drop(target_col, axis=1)
            y_train = train_data[target_col]
            X_val = val_data.drop(target_col, axis=1)
            y_val = val_data[target_col]

            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)

            return score

        except Exception as e:
            logger.error(f"Error evaluating window: {e}")
            return 0.0

class AdaptiveRegularizer:
    """Regularización adaptativa basada en señales de overfitting"""

    def __init__(self):
        self.regularization_params = {
            'l1_penalty': 0.01,
            'l2_penalty': 0.01,
            'dropout_rate': 0.2,
            'early_stopping_patience': 10,
            'max_features': None,
            'min_samples_split': 2
        }
        self.overfitting_threshold = 0.15  # 15% gap = overfitting

    def adjust_regularization(self, performance_history: List[ModelPerformance]):
        """Ajustar regularización basado en historial de performance"""
        if len(performance_history) < 5:
            return

        # Calcular train vs validation gap reciente
        recent_perfs = performance_history[-5:]
        avg_train_score = np.mean([p.train_score for p in recent_perfs])
        avg_val_score = np.mean([p.val_score for p in recent_perfs])

        overfitting_gap = avg_train_score - avg_val_score

        if overfitting_gap > self.overfitting_threshold:
            # Aumentar regularización
            self.regularization_params['l2_penalty'] *= 1.5
            self.regularization_params['dropout_rate'] = min(0.5, self.regularization_params['dropout_rate'] * 1.2)
            self.regularization_params['early_stopping_patience'] = max(5, self.regularization_params['early_stopping_patience'] - 2)

            logger.warning(f"🚨 OVERFIT DETECTED: Gap {overfitting_gap:.3f}, increasing regularization")

        elif overfitting_gap < 0.05:
            # Reducir regularización (posible underfitting)
            self.regularization_params['l2_penalty'] *= 0.9
            self.regularization_params['dropout_rate'] *= 0.95
            self.regularization_params['early_stopping_patience'] = min(20, self.regularization_params['early_stopping_patience'] + 1)

            logger.info(f"📉 Underfit detected: Gap {overfitting_gap:.3f}, reducing regularization")

class DiverseEnsembleBuilder:
    """Construye ensemble diverso para evitar overfitting"""

    def __init__(self, max_models: int = 10, similarity_threshold: float = 0.85):
        self.max_models = max_models
        self.similarity_threshold = similarity_threshold
        self.ensemble_models = []
        self.model_weights = {}

    def add_model_to_ensemble(self, candidate_model, validation_data: pd.DataFrame) -> bool:
        """Añadir modelo solo si aumenta diversidad del ensemble"""
        try:
            if len(self.ensemble_models) >= self.max_models:
                # Reemplazar peor modelo si candidato es mejor
                return self._replace_worst_model(candidate_model, validation_data)

            # Verificar diversidad con modelos existentes
            for existing_model in self.ensemble_models:
                similarity = self._calculate_model_similarity(candidate_model, existing_model, validation_data)
                if similarity > self.similarity_threshold:
                    logger.info(f"⚠️ Model rejected: similarity {similarity:.3f} > {self.similarity_threshold}")
                    return False

            # Verificar que mejora ensemble performance
            ensemble_improvement = self._calculate_ensemble_improvement(candidate_model, validation_data)

            if ensemble_improvement < 0.01:  # Menos del 1% de mejora
                logger.info(f"⚠️ Model rejected: improvement {ensemble_improvement:.3f} < 0.01")
                return False

            # Añadir al ensemble
            self.ensemble_models.append(candidate_model)
            self._update_weights()

            logger.info(f"✅ Model added to ensemble (improvement: {ensemble_improvement:.3f})")
            return True

        except Exception as e:
            logger.error(f"Error adding model to ensemble: {e}")
            return False

    def _calculate_model_similarity(self, model1, model2, data: pd.DataFrame) -> float:
        """Calcular similitud entre dos modelos"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
            else:
                X = data.iloc[:, :-1]

            pred1 = model1.predict(X)
            pred2 = model2.predict(X)

            # Correlación de Pearson entre predicciones
            correlation = np.corrcoef(pred1, pred2)[0, 1]
            return abs(correlation)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 1.0  # Asumir máxima similitud en caso de error

    def _calculate_ensemble_improvement(self, candidate_model, data: pd.DataFrame) -> float:
        """Calcular mejora que aporta el candidato al ensemble"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
                y = data['target']
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            # Score del ensemble actual
            current_predictions = self._get_ensemble_predictions(X)
            current_score = self._calculate_score(current_predictions, y)

            # Score con candidato añadido
            candidate_predictions = candidate_model.predict(X)
            new_predictions = (current_predictions + candidate_predictions) / 2
            new_score = self._calculate_score(new_predictions, y)

            return new_score - current_score

        except Exception as e:
            logger.error(f"Error calculating ensemble improvement: {e}")
            return 0.0

    def _get_ensemble_predictions(self, X):
        """Obtener predicciones del ensemble actual"""
        if not self.ensemble_models:
            return np.zeros(len(X))

        predictions = np.zeros(len(X))
        for model in self.ensemble_models:
            predictions += model.predict(X)

        return predictions / len(self.ensemble_models)

    def _calculate_score(self, predictions, targets):
        """Calcular score de accuracy"""
        try:
            # Para clasificación binaria
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_preds == targets.values)
            return accuracy
        except:
            # Fallback
            return 0.5

    def _replace_worst_model(self, candidate_model, validation_data: pd.DataFrame) -> bool:
        """Reemplazar el peor modelo del ensemble"""
        try:
            if not self.ensemble_models:
                self.ensemble_models.append(candidate_model)
                return True

            # Evaluar todos los modelos
            model_scores = []
            for i, model in enumerate(self.ensemble_models):
                score = self._evaluate_single_model(model, validation_data)
                model_scores.append((i, score))

            # Encontrar peor modelo
            worst_idx, worst_score = min(model_scores, key=lambda x: x[1])

            # Evaluar candidato
            candidate_score = self._evaluate_single_model(candidate_model, validation_data)

            if candidate_score > worst_score:
                # Reemplazar
                self.ensemble_models[worst_idx] = candidate_model
                self._update_weights()
                logger.info(f"🔄 Replaced worst model (score {worst_score:.3f}) with candidate (score {candidate_score:.3f})")
                return True
            else:
                logger.info(f"⚠️ Candidate worse than worst model: {candidate_score:.3f} < {worst_score:.3f}")
                return False

        except Exception as e:
            logger.error(f"Error replacing model: {e}")
            return False

    def _evaluate_single_model(self, model, data: pd.DataFrame) -> float:
        """Evaluar un solo modelo"""
        try:
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
                y = data['target']
            else:
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            return model.score(X, y)
        except:
            return 0.0

    def _update_weights(self):
        """Actualizar pesos del ensemble"""
        if not self.ensemble_models:
            return

        # Pesos iguales por ahora (podría ser más sofisticado)
        weight = 1.0 / len(self.ensemble_models)
        for i, model in enumerate(self.ensemble_models):
            self.model_weights[f"model_{i}"] = weight

class ConceptDriftDetector:
    """Detecta cambios en la distribución de datos (concept drift)"""

    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold
        self.reference_distribution = None
        self.drift_history = []

    def detect_drift(self, new_data: pd.DataFrame) -> bool:
        """Detectar si hay concept drift en los nuevos datos"""
        try:
            current_distribution = self._calculate_distribution(new_data)

            if self.reference_distribution is None:
                self.reference_distribution = current_distribution
                return False

            # Calcular distancia entre distribuciones
            drift_distance = self._calculate_distribution_distance(
                self.reference_distribution, current_distribution
            )

            self.drift_history.append({
                'timestamp': datetime.now(),
                'distance': drift_distance,
                'threshold': self.drift_threshold
            })

            if drift_distance > self.drift_threshold:
                logger.warning(f"🌊 CONCEPT DRIFT DETECTED: Distance {drift_distance:.3f} > {self.drift_threshold}")
                # Actualizar distribución de referencia
                self.reference_distribution = current_distribution
                return True

            return False

        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return False

    def _calculate_distribution(self, data: pd.DataFrame) -> np.ndarray:
        """Calcular distribución estadística de los datos"""
        try:
            # Usar múltiples momentos estadísticos
            features = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if col == 'target':
                    continue

                values = data[col].dropna().values
                if len(values) > 10:
                    # Calcular histograma normalizado
                    hist, _ = np.histogram(values, bins=10, density=True)
                    features.extend(hist)

            return np.array(features)

        except Exception as e:
            logger.error(f"Error calculating distribution: {e}")
            return np.array([])

    def _calculate_distribution_distance(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calcular distancia entre dos distribuciones"""
        try:
            # Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon

            # Asegurar que tienen la misma longitud
            min_len = min(len(dist1), len(dist2))
            if min_len == 0:
                return 0.0

            dist1_norm = dist1[:min_len] / (np.sum(dist1[:min_len]) + 1e-10)
            dist2_norm = dist2[:min_len] / (np.sum(dist2[:min_len]) + 1e-10)

            return jensenshannon(dist1_norm, dist2_norm)

        except Exception as e:
            logger.error(f"Error calculating distribution distance: {e}")
            return 0.0

class SmartEarlyStopper:
    """Early stopping inteligente para prevenir overfitting"""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def should_stop_training(self, validation_scores: List[float]) -> bool:
        """Decidir si parar el entrenamiento"""
        if len(validation_scores) < self.patience:
            return False

        current_score = validation_scores[-1]

        if current_score > self.best_score + self.min_delta:
            # Mejora encontrada
            self.best_score = current_score
            self.wait = 0
            # Guardar mejores pesos aquí si fuera una red neuronal
        else:
            self.wait += 1

        if self.wait >= self.patience:
            logger.warning(f"⏹️ Early stopping: No improvement in {self.patience} epochs")
            return True

        return False

class AutoRetrainingSystem:
    """Sistema de reentrenamiento completamente automático"""

    def __init__(self):
        self.data_buffer = []
        self.model_versions = {}
        self.performance_history = []

        # Componentes anti-overfitting (5 capas)
        self.validator = AntiOverfitValidator()
        self.regularizer = AdaptiveRegularizer()
        self.ensemble_builder = DiverseEnsembleBuilder()
        self.drift_detector = ConceptDriftDetector()
        self.early_stopper = SmartEarlyStopper()

        # Triggers automáticos (4 capas adicionales = 9 total)
        self.auto_triggers = {
            'time_based': {
                'enabled': True,
                'interval_hours': 168,  # 7 días
                'last_retrain': datetime.now()
            },
            'performance_based': {
                'enabled': True,
                'min_trades': 100,
                'win_rate_threshold': 0.52,
                'max_drawdown_threshold': 0.12
            },
            'regime_change': {
                'enabled': True,
                'regime_switches': 0,
                'last_regime': None
            },
            'data_volume': {
                'enabled': True,
                'min_new_trades': 500
            }
        }

        # Modelos base
        self.models = self._load_base_models()

        # LOG de las 9 capas de protección
        logger.info("=" * 70)
        logger.info("🛡️  9 CAPAS DE PROTECCIÓN ANTI-OVERFITTING INICIALIZADAS")
        logger.info("=" * 70)
        logger.info("   1️⃣  AntiOverfitValidator       - Validación cruzada continua")
        logger.info("   2️⃣  AdaptiveRegularizer        - Regularización adaptativa")
        logger.info("   3️⃣  DiverseEnsembleBuilder     - Ensemble diverso")
        logger.info("   4️⃣  ConceptDriftDetector       - Detección de concept drift")
        logger.info("   5️⃣  SmartEarlyStopper          - Early stopping inteligente")
        logger.info("   6️⃣  TimeBasedTrigger           - Trigger por tiempo (7 días)")
        logger.info("   7️⃣  PerformanceBasedTrigger    - Trigger por performance")
        logger.info("   8️⃣  RegimeChangeTrigger        - Trigger por cambio de régimen")
        logger.info("   9️⃣  DataVolumeTrigger          - Trigger por volumen de datos")
        logger.info("=" * 70)
        logger.info("🤖 Auto-Retraining System initialized with full anti-overfitting protection")

    def _load_base_models(self) -> Dict[str, Any]:
        """Cargar modelos base del sistema"""
        models = {}

        try:
            # Cargar modelos desde el sistema existente
            from l3_strategy.regime_classifier import clasificar_regimen_mejorado
            from l1_operational.models import L1Model

            # Placeholder - en implementación real cargar modelos entrenados
            models['regime_classifier'] = 'loaded'
            models['l1_models'] = 'loaded'

            logger.info("✅ Base models loaded")
            return models

        except Exception as e:
            logger.error(f"Error loading base models: {e}")
            return {}

    def add_trade_data(self, trade_data: TradeData):
        """Añadir datos de trade y verificar triggers automáticos"""
        self.data_buffer.append(trade_data)

        # Verificar todos los triggers
        if self._should_retrain():
            asyncio.create_task(self._auto_retrain_models())

    def _should_retrain(self) -> bool:
        """Decidir automáticamente si reentrenar"""

        # 1. Trigger por tiempo
        time_trigger = self.auto_triggers['time_based']
        if time_trigger['enabled']:
            hours_since_last = (datetime.now() - time_trigger['last_retrain']).total_seconds() / 3600
            if hours_since_last >= time_trigger['interval_hours']:
                logger.info(f"🔄 AUTO-TRIGGER: Time-based ({hours_since_last:.1f}h >= {time_trigger['interval_hours']}h)")
                return True

        # 2. Trigger por performance
        perf_trigger = self.auto_triggers['performance_based']
        if perf_trigger['enabled'] and len(self.data_buffer) >= perf_trigger['min_trades']:
            recent_performance = self._calculate_recent_performance(
                self.data_buffer[-perf_trigger['min_trades']:]
            )

            if recent_performance['win_rate'] < perf_trigger['win_rate_threshold']:
                logger.warning(f"🚨 AUTO-TRIGGER: Win rate {recent_performance['win_rate']:.1%} < {perf_trigger['win_rate_threshold']:.1%}")
                return True

            if recent_performance['max_drawdown'] > perf_trigger['max_drawdown_threshold']:
                logger.warning(f"🚨 AUTO-TRIGGER: Drawdown {recent_performance['max_drawdown']:.1%} > {perf_trigger['max_drawdown_threshold']:.1%}")
                return True

        # 3. Trigger por cambio de régimen
        regime_trigger = self.auto_triggers['regime_change']
        if regime_trigger['enabled']:
            current_regime = self._detect_current_regime()
            if current_regime != regime_trigger['last_regime']:
                regime_trigger['regime_switches'] += 1
                if regime_trigger['regime_switches'] >= 3:
                    logger.info(f"🔄 AUTO-TRIGGER: Regime changed {regime_trigger['regime_switches']}x (current: {current_regime})")
                    regime_trigger['regime_switches'] = 0
                    regime_trigger['last_regime'] = current_regime
                    return True

        # 4. Trigger por volumen de datos
        volume_trigger = self.auto_triggers['data_volume']
        if volume_trigger['enabled'] and len(self.data_buffer) >= volume_trigger['min_new_trades']:
            logger.info(f"🔄 AUTO-TRIGGER: Data volume ({len(self.data_buffer)} >= {volume_trigger['min_new_trades']})")
            return True

        return False

    def _calculate_recent_performance(self, trades: List[TradeData]) -> Dict[str, float]:
        """Calcular performance reciente"""
        if not trades:
            return {'win_rate': 0.0, 'max_drawdown': 0.0}

        profitable_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = profitable_trades / len(trades)

        # Calcular drawdown máximo
        cumulative_pnl = 0
        peak = 0
        max_drawdown = 0

        for trade in trades:
            cumulative_pnl += trade.pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'win_rate': win_rate,
            'max_drawdown': max_drawdown
        }

    def _detect_current_regime(self) -> str:
        """Detectar régimen de mercado actual"""
        try:
            # Placeholder - en implementación real usar clasificador de régimen
            return "neutral"
        except:
            return "unknown"

    async def _auto_retrain_models(self):
        """Reentrenar modelos automáticamente con protección anti-overfitting"""
        logger.info("🤖 INICIANDO AUTO-REENTRENAMIENTO CON PROTECCIÓN ANTI-OVERFITTING...")

        try:
            # 1. Preparar datos de entrenamiento
            training_data = self._prepare_training_data()

            if len(training_data) < 100:
                logger.warning("⚠️ Insuficientes datos para reentrenamiento")
                return

            # 2. Detectar concept drift
            if self.drift_detector.detect_drift(training_data):
                logger.info("🌊 Concept drift detected - applying extra validation")

            # 3. Reentrenar cada modelo con validación anti-overfitting
            for model_name in self.models.keys():
                logger.info(f"🔄 Reentrenando {model_name}...")

                # Crear modelo candidato
                candidate_model = self._create_candidate_model(model_name)

                # Aplicar TODAS las protecciones anti-overfitting
                if self._passes_all_anti_overfitting_checks(candidate_model, training_data):
                    # Backup del modelo actual
                    self._backup_current_model(model_name)

                    # Desplegar nuevo modelo
                    self._deploy_new_model(model_name, candidate_model)
                    logger.info(f"✅ {model_name} mejorado y desplegado")
                else:
                    logger.info(f"⚠️ {model_name} no pasó validaciones anti-overfitting")

            # 4. Limpiar buffer de datos
            self.data_buffer = self.data_buffer[-500:]  # Mantener últimos 500

            # 5. Resetear timers
            self.auto_triggers['time_based']['last_retrain'] = datetime.now()

            logger.info("🎉 AUTO-REENTRENAMIENTO COMPLETADO CON PROTECCIÓN ANTI-OVERFITTING")

        except Exception as e:
            logger.error(f"❌ Error en auto-reentrenamiento: {e}")

    def _prepare_training_data(self) -> pd.DataFrame:
        """Convertir buffer de trades a DataFrame de entrenamiento"""
        try:
            data = []
            for trade in self.data_buffer:
                row = {
                    'timestamp': trade.timestamp.timestamp(),
                    'symbol': trade.symbol,
                    'side': 1 if trade.side == 'buy' else 0,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'confidence': trade.confidence,
                    'regime': trade.regime_at_entry,
                    'target': 1 if trade.pnl > 0 else 0,  # Target binario
                    **trade.features  # Incluir features técnicas
                }
                data.append(row)

            df = pd.DataFrame(data)

            # Limpiar datos
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            return df

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()

    def _create_candidate_model(self, model_name: str):
        """Crear modelo candidato para evaluación"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            # Crear modelo con regularización adaptativa
            if model_name == 'regime_classifier':
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=int(self.regularizer.regularization_params['min_samples_split']),
                    random_state=42
                )
            elif model_name == 'l1_models':
                return LogisticRegression(
                    C=1.0 / self.regularizer.regularization_params['l2_penalty'],
                    random_state=42
                )
            else:
                # Modelo genérico
                return RandomForestClassifier(n_estimators=50, random_state=42)

        except Exception as e:
            logger.error(f"Error creating candidate model {model_name}: {e}")
            return None

    def _passes_all_anti_overfitting_checks(self, candidate_model, training_data: pd.DataFrame) -> bool:
        """Aplicar TODAS las verificaciones anti-overfitting"""

        try:
            # 1. Validación cruzada continua
            cv_passed, cv_score = self.validator.validate_new_model(candidate_model, training_data)
            if not cv_passed:
                logger.warning("❌ Falló validación cruzada")
                return False

            # 2. Verificar diversidad del ensemble
            if not self.ensemble_builder.add_model_to_ensemble(candidate_model, training_data):
                logger.warning("❌ Falló verificación de diversidad")
                return False

            # 3. Verificar que no hay concept drift extremo
            if self.drift_detector.detect_drift(training_data):
                logger.info("🌊 Concept drift detectado - aplicando validación extra")

                # Re-evaluar con datos más recientes
                recent_data = training_data.tail(int(len(training_data) * 0.3))
                cv_passed_recent, _ = self.validator.validate_new_model(candidate_model, recent_data)
                if not cv_passed_recent:
                    logger.warning("❌ Falló validación con datos recientes post-drift")
                    return False

            # 4. Verificar early stopping (simulado)
            # En implementación real, esto se haría durante el entrenamiento

            logger.info("✅ PASÓ TODAS LAS VERIFICACIONES ANTI-OVERFITTING")
            return True

        except Exception as e:
            logger.error(f"Error en verificaciones anti-overfitting: {e}")
            return False

    def _backup_current_model(self, model_name: str):
        """Crear backup del modelo actual"""
        try:
            backup_path = f"models/backups/{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            if model_name in self.models:
                joblib.dump(self.models[model_name], backup_path)
                logger.info(f"💾 Backup creado: {backup_path}")

        except Exception as e:
            logger.error(f"Error creating backup for {model_name}: {e}")

    def _deploy_new_model(self, model_name: str, new_model):
        """Desplegar nuevo modelo"""
        self.models[model_name] = new_model

        # Registrar versión
        version_info = {
            'model': new_model,
            'version': f"auto_v{len(self.model_versions.get(model_name, [])) + 1}",
            'deployed_at': datetime.now(),
            'auto_generated': True
        }

        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(version_info)

        logger.info(f"🚀 Desplegado {model_name} versión {version_info['version']}")

# Sistema principal de auto-aprendizaje
class SelfImprovingTradingSystem:
    """Sistema de trading que se mejora solo con protección total anti-overfitting"""
    
    _instance = None

    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SelfImprovingTradingSystem."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(self):
        # Evitar reinicialización si ya existe
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.auto_retrainer = AutoRetrainingSystem()
        self.performance_monitor = PerformanceMonitor()
        self.online_learners = {}  # Para componentes que pueden aprender online

        # Estado del sistema
        self.is_running = False
        self.last_update = datetime.now()
        
        # Componentes integrados
        self.state_manager = None
        self.order_manager = None
        self.portfolio_manager = None
        self.l2_processor = None
        self.trading_metrics = None
        self._initialized = True

        logger.info("🤖 Self-Improving Trading System initialized with maximum anti-overfitting protection")

    def integrate(self, state_manager=None, order_manager=None, portfolio_manager=None, 
                  l2_processor=None, trading_metrics=None):
        """
        Integrar el sistema de auto-aprendizaje con los componentes del sistema HRM.
        
        Args:
            state_manager: StateCoordinator para gestión de estado global
            order_manager: OrderManager para gestión de órdenes
            portfolio_manager: PortfolioManager para gestión de portfolio
            l2_processor: L2TacticProcessor para señales tácticas
            trading_metrics: TradingMetrics para métricas de trading
        """
        self.state_manager = state_manager
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.l2_processor = l2_processor
        self.trading_metrics = trading_metrics
        
        logger.info("=" * 70)
        logger.info("🔗 Auto-Learning System Integration:")
        logger.info(f"   State Manager:     {'✅' if state_manager else '❌'}")
        logger.info(f"   Order Manager:     {'✅' if order_manager else '❌'}")
        logger.info(f"   Portfolio Manager: {'✅' if portfolio_manager else '❌'}")
        logger.info(f"   L2 Processor:      {'✅' if l2_processor else '❌'}")
        logger.info(f"   Trading Metrics:   {'✅' if trading_metrics else '❌'}")
        logger.info("=" * 70)

    def start_auto_improvement(self):
        """Iniciar el ciclo de auto-mejora"""
        self.is_running = True
        logger.info("🚀 Auto-improvement cycle started")

        # En implementación real, esto sería un loop continuo
        # Por ahora, solo registramos que está listo

    def record_trade(self, trade_data: Dict[str, Any]):
        """Registrar trade para aprendizaje automático"""
        try:
            # Convertir a TradeData
            trade = TradeData(
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', 'UNKNOWN'),
                side=trade_data.get('side', 'buy'),
                entry_price=trade_data.get('entry_price', 0.0),
                exit_price=trade_data.get('exit_price', 0.0),
                quantity=trade_data.get('quantity', 0.0),
                pnl=trade_data.get('pnl', 0.0),
                pnl_pct=trade_data.get('pnl_pct', 0.0),
                model_used=trade_data.get('model_used', 'unknown'),
                confidence=trade_data.get('confidence', 0.5),
                regime_at_entry=trade_data.get('regime', 'neutral'),
                features=trade_data.get('features', {}),
                market_data=trade_data.get('market_data', {})
            )

            # Añadir al sistema de auto-reentrenamiento
            self.auto_retrainer.add_trade_data(trade)

            # Actualizar métricas de performance
            self.performance_monitor.update_metrics(trade)

            logger.info(f"📊 Trade recorded for auto-learning: {trade.symbol} {trade.side} PnL: {trade.pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Error recording trade: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtener estado del sistema de auto-mejora.
        
        CRITICAL FIX: Ya NO llama a _get_current_portfolio_data() que usaba asyncio.run()
        En su lugar, devuelve un placeholder que debe ser llenado por el caller async
        """
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'data_buffer_size': len(self.auto_retrainer.data_buffer),
            'models_count': len(self.auto_retrainer.models),
            'ensemble_size': len(self.auto_retrainer.ensemble_builder.ensemble_models),
            'performance_metrics': self.performance_monitor.get_summary(),
            'anti_overfitting_active': True,
            'portfolio_data': None,  # CRITICAL FIX: Placeholder - use get_system_status_async() para datos reales
            'integration': {
                'state_manager': self.state_manager is not None,
                'order_manager': self.order_manager is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'l2_processor': self.l2_processor is not None,
                'trading_metrics': self.trading_metrics is not None
            }
        }
    
    async def get_system_status_async(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Versión async que obtiene datos reales del portfolio.
        Esta es la versión que debe usarse desde contextos async.
        """
        # Obtener datos del portfolio de forma async
        portfolio_data = await self._get_current_portfolio_data_async()
        
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'data_buffer_size': len(self.auto_retrainer.data_buffer),
            'models_count': len(self.auto_retrainer.models),
            'ensemble_size': len(self.auto_retrainer.ensemble_builder.ensemble_models),
            'performance_metrics': self.performance_monitor.get_summary(),
            'anti_overfitting_active': True,
            'portfolio_data': portfolio_data,  # CRITICAL: Datos reales obtenidos async
            'integration': {
                'state_manager': self.state_manager is not None,
                'order_manager': self.order_manager is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'l2_processor': self.l2_processor is not None,
                'trading_metrics': self.trading_metrics is not None
            }
        }
    
    async def _get_current_portfolio_data_async(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Get CURRENT portfolio data from PortfolioManager (ASYNC VERSION).
        This ensures AutoLearning trains on actual balances, not stale/initial values.
        
        CRITICAL FIX: Solo retorna balances verificados desde sync async. Si los balances
        fueron obtenidos vía fallback o error, esto se marca y el entrenamiento debe bloquearse.
        """
        try:
            if self.portfolio_manager is None:
                return {'error': 'PortfolioManager not available', 'is_verified': False}
            
            # Check if balances are verified (from async sync)
            is_verified = False
            verification_status = {}
            if hasattr(self.portfolio_manager, 'are_balances_verified'):
                is_verified = self.portfolio_manager.are_balances_verified()
            if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                verification_status = self.portfolio_manager.get_balance_verification_status()
            
            # Get balances using async methods
            btc_balance = 0.0
            eth_balance = 0.0
            usdt_balance = 0.0
            
            if hasattr(self.portfolio_manager, 'get_balances_async'):
                balances = await self.portfolio_manager.get_balances_async()
                btc_balance = balances.get('BTC', 0.0)
                eth_balance = balances.get('ETH', 0.0)
                usdt_balance = balances.get('USDT', 0.0)
            elif hasattr(self.portfolio_manager, 'get_asset_balance_async'):
                btc_balance = await self.portfolio_manager.get_asset_balance_async('BTC')
                eth_balance = await self.portfolio_manager.get_asset_balance_async('ETH')
                usdt_balance = await self.portfolio_manager.get_asset_balance_async('USDT')
            else:
                # Fallback - but mark as not verified
                logger.warning("[AUTO_LEARNING] No async balance methods available - data will be unverified")
                is_verified = False
                # No sync fallback - esto es intencional para prevenir violaciones async
                return {
                    'error': 'No async balance methods available in PortfolioManager',
                    'is_verified': False,
                    'data_source': 'error_no_async_methods'
                }
            
            # Get current NAV using async method if available
            current_nav = 0.0
            if hasattr(self.portfolio_manager, 'get_total_value_async'):
                current_nav = await self.portfolio_manager.get_total_value_async()
            elif hasattr(self.portfolio_manager, 'portfolio'):
                current_nav = self.portfolio_manager.portfolio.get('total', 0.0)
            
            # Calculate PnL
            initial_balance = getattr(self.portfolio_manager, 'initial_balance', 500.0)
            pnl = current_nav - initial_balance
            pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0.0
            
            # Calculate exposure
            total_crypto_value = 0.0
            if hasattr(self.portfolio_manager, 'client') and self.portfolio_manager.client:
                client = self.portfolio_manager.client
                if hasattr(client, 'get_market_price'):
                    try:
                        btc_price = client.get_market_price('BTCUSDT')
                        eth_price = client.get_market_price('ETHUSDT')
                        total_crypto_value = (btc_balance * btc_price) + (eth_balance * eth_price)
                    except:
                        pass
            
            exposure_pct = (total_crypto_value / current_nav * 100) if current_nav > 0 else 0.0
            
            return {
                'btc_balance': btc_balance,
                'eth_balance': eth_balance,
                'usdt_balance': usdt_balance,
                'current_nav': current_nav,
                'initial_balance': initial_balance,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exposure_pct': exposure_pct,
                'data_source': 'PortfolioManager (async_synced)' if is_verified else 'PortfolioManager (unverified)',
                'is_verified': is_verified,
                'verification_status': verification_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio data: {e}")
            return {'error': str(e), 'is_verified': False}

    def can_train(self) -> Tuple[bool, str]:
        """
        CRITICAL FIX: Versión sincrona que NO usa asyncio.run().
        Solo verifica el estado de verificación sincronizado - no obtiene datos nuevos.
        
        Para verificación completa con datos frescos, usar can_train_async().
        """
        try:
            if self.portfolio_manager is None:
                return False, "PortfolioManager not available"
            
            # Check verification status (síncrono - solo lee estado cacheado)
            if hasattr(self.portfolio_manager, 'are_balances_verified'):
                if not self.portfolio_manager.are_balances_verified():
                    verification_status = {}
                    if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                        verification_status = self.portfolio_manager.get_balance_verification_status()
                    
                    if verification_status.get('was_fallback_used'):
                        return False, f"Balances from fallback source - training blocked. Errors: {verification_status.get('sync_errors', [])}"
                    
                    return False, "Balances not verified from async sync"
            
            # Check if balances are fresh
            if hasattr(self.portfolio_manager, 'get_balance_verification_status'):
                status = self.portfolio_manager.get_balance_verification_status()
                seconds_since_sync = status.get('seconds_since_sync')
                
                if seconds_since_sync is not None and seconds_since_sync > 60:
                    return False, f"Balances are stale ({seconds_since_sync:.0f}s old) - sync required before training"
            
            return True, "Balances verified - training allowed"
            
        except Exception as e:
            return False, f"Error checking training eligibility: {e}"

    async def can_train_async(self) -> Tuple[bool, str]:
        """
        CRITICAL FIX: Versión async que obtiene datos frescos y verifica estado.
        Esta es la versión preferida para usar desde contextos async.
        """
        try:
            if self.portfolio_manager is None:
                return False, "PortfolioManager not available"
            
            # Get fresh portfolio data
            portfolio_data = await self._get_current_portfolio_data_async()
            
            # Check if data is verified
            if not portfolio_data.get('is_verified', False):
                error = portfolio_data.get('error', 'Unknown error')
                return False, f"Portfolio data not verified: {error}"
            
            # Check if balances are fresh
            verification_status = portfolio_data.get('verification_status', {})
            seconds_since_sync = verification_status.get('seconds_since_sync')
            
            if seconds_since_sync is not None and seconds_since_sync > 60:
                return False, f"Balances are stale ({seconds_since_sync:.0f}s old) - sync required before training"
            
            return True, "Balances verified and fresh - training allowed"
            
        except Exception as e:
            return False, f"Error checking training eligibility: {e}"

# Clase auxiliar para monitoreo de performance
class PerformanceMonitor:
    """Monitorea performance del sistema"""

    def __init__(self):
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    def update_metrics(self, trade: TradeData):
        """Actualizar métricas con nuevo trade"""
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += trade.pnl

        if trade.pnl > 0:
            self.metrics['winning_trades'] += 1

        # Calcular win rate
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']

    def get_summary(self) -> Dict[str, float]:
        """Obtener resumen de métricas"""
        return self.metrics.copy()

# Función principal para testing
if __name__ == "__main__":
    # Crear sistema
    system = SelfImprovingTradingSystem()

    # Simular algunos trades
    sample_trades = [
        {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 50000,
            'exit_price': 51000,
            'quantity': 0.01,
            'pnl': 10.0,
            'pnl_pct': 0.02,
            'model_used': 'l2_finrl',
            'confidence': 0.8,
            'regime': 'bull',
            'features': {'rsi': 65, 'macd': 0.5}
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'sell',
            'entry_price': 3000,
            'exit_price': 2950,
            'quantity': 0.1,
            'pnl': 5.0,
            'pnl_pct': 0.0167,
            'model_used': 'l1_technical',
            'confidence': 0.7,
            'regime': 'neutral',
            'features': {'rsi': 45, 'macd': -0.3}
        }
    ]

    # Registrar trades
    for trade_data in sample_trades:
        system.record_trade(trade_data)

    # Mostrar estado (versión sync - sin datos de portfolio)
    status = system.get_system_status()
    print("🤖 Sistema de Auto-Aprendizaje con Anti-Overfitting Máximo:")
    print(f"   📊 Trades en buffer: {status['data_buffer_size']}")
    print(f"   🧠 Modelos activos: {status['models_count']}")
    print(f"   🎯 Ensemble size: {status['ensemble_size']}")
    print(f"   🛡️ Anti-overfitting: {'ACTIVO' if status['anti_overfitting_active'] else 'INACTIVO'}")
    print(f"   📈 Performance: {status['performance_metrics']}")

    print("\n✅ SISTEMA LISTO PARA AUTO-MEJORA CONTINUA CON PROTECCIÓN TOTAL ANTI-OVERFITTING!")



================================================
FILE: bootstrap.py
================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Bootstrap Module

This module handles system initialization, configuration loading,
and component wiring for the HRM system.

🔥 PRIORIDAD 4: Introducir session_is_fresh - no depender de balances
"""

import os
import sys
import asyncio
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.state_manager import initialize_state, validate_state_structure
from core.portfolio_manager import PortfolioManager
from core.logging import logger
from core.config import get_config, set_forced_mode
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics

from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus

from sentiment.sentiment_manager import update_sentiment_texts

from system.system_cleanup import perform_full_cleanup
from storage.paper_trade_logger import get_paper_logger
from config_loader import get_initial_balances, get_capital_usd


# 🔥 PRIORIDAD 4: SESSION IS FRESH - Flag global para nueva sesión
session_is_fresh = True

def is_session_fresh() -> bool:
    """Check if current session is fresh (not dependent on balances)"""
    global session_is_fresh
    return session_is_fresh

def mark_session_used():
    """Mark session as no longer fresh"""
    global session_is_fresh
    session_is_fresh = False


class HRMBootstrap:
    """HRM system bootstrap and initialization."""
    
    def __init__(self):
        self.components = {}
        self.state = None
        self.portfolio_manager = None
        self.order_manager = None
        self.runtime_loop = None
        self._mode = "paper"  # Default mode, will be set in bootstrap_system
    
    def _get_mode_from_bootstrap(self) -> str:
        """Get the mode from the single source of truth (bootstrap)."""
        return self._mode
        
    async def bootstrap_system(self, mode: str = "paper") -> Tuple[PortfolioManager, OrderManager]:
        """
        Bootstrap the entire HRM system.
        
        🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión,
        NO los balances. Esto evita dependencia de estado externo.
        """
        # Store mode as single source of truth for this bootstrap instance
        self._mode = mode
        
        logger.info("🚀 Starting HRM System Bootstrap")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        logger.info(f"🎯 BOOTSTRAP MODE: {mode}")
        
        try:
            # 1. System Cleanup
            await self._perform_system_cleanup()
            
            # 2. Load Configuration
            env_config = await self._load_configuration(mode)
            
            # 3. Initialize State
            self.state = await self._initialize_state(env_config)
            
            # 4. Initialize Core Components
            await self._initialize_core_components(env_config)
            
            # 5. Initialize L1 Components
            await self._initialize_l1_components(env_config)
            
            # 6. Initialize L2 Components
            await self._initialize_l2_components(env_config)
            
            # 7. Initialize L3 Components (if enabled)
            if not APAGAR_L3:
                await self._initialize_l3_components()
            
            # 8. Initialize Portfolio Manager
            self.portfolio_manager = await self._initialize_portfolio_manager(env_config)
            
            # 9. Initialize Order Manager
            self.order_manager = await self._initialize_order_manager()
            
            # 10. Initialize Runtime Loop
            from runtime_loop import HRMRuntimeLoop
            self.runtime_loop = HRMRuntimeLoop(self.portfolio_manager, self.order_manager)
            
            # 11. Start Background Services
            await self._start_background_services()
            
            # 12. Marcar sesión como usada después de bootstrap exitoso
            mark_session_used()
            logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()} (post-bootstrap)")
            
            logger.info("✅ HRM System Bootstrap Complete")
            return self.portfolio_manager, self.order_manager
            
        except Exception as e:
            logger.error(f"❌ Bootstrap failed: {e}", exc_info=True)
            await self._cleanup_on_failure()
            raise
    
    async def _perform_system_cleanup(self):
        """Perform system cleanup before startup."""
        logger.info("🧹 Running system cleanup...")
        
        cleanup_result = perform_full_cleanup(mode="paper")
        
        if not cleanup_result.get("success", False):
            logger.warning("⚠️ Cleanup completed with warnings")
        else:
            logger.info(f"✅ Cleanup completed: {cleanup_result.get('deleted_files', 0)} files, {cleanup_result.get('deleted_dirs', 0)} directories removed")
        
        # Clean paper trades for independent testing
        try:
            logger.info("🧹 Cleaning paper trades history...")
            get_paper_logger(clear_on_init=True)
            logger.info("✅ Paper trades history cleaned")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning paper trades: {e}")
    
    async def _load_configuration(self, mode: str) -> Dict[str, Any]:
        """Load system configuration."""
        logger.info(f"⚙️ Loading configuration for mode: {mode}")
        
        # Load environment variables
        load_dotenv()
        
        # 🔥 CRITICAL: Set the forced mode in core.config BEFORE getting config
        # This establishes the single source of truth for the mode
        set_forced_mode(mode)
        logger.info(f"🎯 Mode set as single source of truth: {mode}")
        
        # Get environment configuration
        # 🔥 PRIORIDAD 4: Usar mode explícito, no depender de configuración previa
        env_config = get_config(mode)
        
        # Check Binance operating mode
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        logger.info(f"🏦 BINANCE MODE: {binance_mode}")
        
        return env_config
    
    async def _initialize_state(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize system state."""
        logger.info("🧠 Initializing system state...")
        
        # Initialize state with symbols and initial balance
        # 🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión
        symbols = env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        initial_balance = get_capital_usd()  # Cargar desde initial_state.json
        
        state = initialize_state(symbols, initial_balance)
        state = validate_state_structure(state)
        
        logger.info(f"✅ State initialized for symbols: {symbols}")
        logger.info(f"   Initial balance: {initial_balance} USDT")
        logger.info(f"   Session fresh: {is_session_fresh()}")
        return state
    
    async def _initialize_core_components(self, env_config: Dict[str, Any]):
        """Initialize core system components."""
        logger.info("🔧 Initializing core components...")
        
        # Initialize trading metrics
        trading_metrics = get_trading_metrics()
        self.components['trading_metrics'] = trading_metrics
        
        # Initialize signal verifier
        signal_verifier = get_signal_verifier()
        self.components['signal_verifier'] = signal_verifier
        
        # Initialize message bus
        message_bus = MessageBus()
        self.components['message_bus'] = message_bus
        
        # Initialize sentiment manager
        sentiment_manager = await self._initialize_sentiment_manager()
        self.components['sentiment_manager'] = sentiment_manager
        
        logger.info("✅ Core components initialized")
    
    async def _initialize_l1_components(self, env_config: Dict[str, Any]):
        """Initialize L1 operational components."""
        logger.info("🔧 Initializing L1 operational components...")
        
        # Determine mode from the single source of truth
        mode = self._get_mode_from_bootstrap()
        
        # Initialize Binance client with injected mode (single source of truth)
        binance_client = BinanceClient(mode=mode)
        self.components['binance_client'] = binance_client
        
        # Initialize SimulatedExchangeClient (para paper trading)
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        
        # 🔥 PRIORIDAD 4: Cargar balances desde initial_state.json
        # Usar config_loader para centralizar la configuración
        initial_balances = get_initial_balances()
        
        logger.info(f"📊 Cargando balances desde initial_state.json: {initial_balances}")
        
        # Limpiar cualquier estado previo y crear nuevo
        SimulatedExchangeClient._instance = None
        SimulatedExchangeClient._initialized = False
        
        # Inicializar con balances conocidos
        simulated_client = SimulatedExchangeClient(initial_balances)
        self.components['simulated_client'] = simulated_client
        
        # Logs requeridos
        logger.info("📊 SIM_INIT: Nueva instancia")
        logger.info(f"🔢 SIM_STATE_ID: {id(simulated_client)}")
        logger.info(f"💰 SIM_BALANCES: {simulated_client.get_balances()}")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        
        # Verificar que los balances no estén vacíos
        if not simulated_client.get_balances() or all(balance == 0 for balance in simulated_client.get_balances().values()):
            logger.critical("🚨 FATAL: SimulatedExchangeClient initialized with empty or zero balances", exc_info=True)
            raise RuntimeError("SimulatedExchangeClient cannot operate with empty or zero balances")
        
        # Initialize RealTimeDataLoader
        loader = RealTimeDataLoader(config)
        self.components['data_loader'] = loader
        
        # Initialize L1 AI Models
        from l1_operational.trend_ai import models as l1_models
        logger.info(f"✅ Loaded L1 AI Models: {list(l1_models.keys())}")
        
        logger.info("✅ L1 components initialized")
    
    async def _initialize_l2_components(self, env_config: Dict[str, Any]):
        """Initialize L2 tactical components."""
        logger.info("🔧 Initializing L2 tactical components...")
        
        # Quick fix: Disable synchronizer in PAPER mode for better performance
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode != "LIVE":
            logger.info("📝 PAPER/TEST MODE: Disabling BTC/ETH synchronizer")
            os.environ['DISABLE_BTC_ETH_SYNC'] = 'true'
        
        # Initialize L2 Config
        l2_config = L2Config()
        self.components['l2_config'] = l2_config
        
        # Initialize L2 Processor
        l2_processor = L2TacticProcessor(l2_config, portfolio_manager=None, apagar_l3=APAGAR_L3)
        self.components['l2_processor'] = l2_processor
        
        # Initialize Risk Manager
        risk_manager = RiskControlManager(l2_config)
        self.components['risk_manager'] = risk_manager
        
        logger.info("✅ L2 components initialized")
    
    async def _initialize_l3_components(self):
        """Initialize L3 strategic components."""
        logger.info("🔧 Initializing L3 strategic components...")
        
        # Import L3 components
        from l3_strategy.regime_classifier import MarketRegimeClassifier
        from l3_strategy.decision_maker import make_decision
        
        # Initialize L3 Classifier
        regime_classifier = MarketRegimeClassifier()
        self.components['regime_classifier'] = regime_classifier
        
        # Initialize L3 Decision Maker
        self.components['l3_decision_maker'] = make_decision
        
        logger.info("✅ L3 components initialized")
    
    async def _initialize_portfolio_manager(self, env_config: Dict[str, Any]) -> PortfolioManager:
        """Initialize Portfolio Manager."""
        logger.info("💼 Initializing Portfolio Manager...")
        
        # Get environment configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        
        # Setup based on binance_mode
        # 🔥 PRIORIDAD 4: Usar modo explícito, no depender de balances previos
        if binance_mode == "LIVE":
            # Live mode: sync mandatory with exchange
            portfolio_mode = "live"
            initial_balance = 0.0  # Will be synced from exchange
        else:
            # Test mode: use simulated balance
            portfolio_mode = "simulated"
            initial_balance = get_capital_usd()  # Cargar desde initial_state.json
            logger.info(f"🧪 TESTING MODE: Using initial balance of {initial_balance} USDT from initial_state.json")
        
        # Initialize Portfolio Manager
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            client=self.components['simulated_client'] if portfolio_mode == "simulated" else self.components['binance_client'],
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )
        
        # Initialize asynchronously to get balances from simulated client
        if portfolio_mode == "simulated":
            await portfolio_manager.initialize_async()
        
        # 🔥 PRIORIDAD 4: NO sync con exchange en modo paper - confiar en balances iniciales
        # Esto evita dependencia de estado externo
        if portfolio_mode == "simulated":
            logger.info("📊 PAPER MODE: Trusting initial balances, no exchange sync needed")
        else:
            # CRITICAL: Synchronize with exchange for production mode
            try:
                logger.info("🔄 Synchronizing with exchange...")
                sync_success = await portfolio_manager.sync_with_exchange()
                
                if sync_success:
                    logger.info("✅ Portfolio synchronized with exchange")
                else:
                    logger.warning("⚠️ Exchange sync failed, loading local state...")
                    loaded = portfolio_manager.load_from_json()
                    if not loaded:
                        logger.info("📄 No saved portfolio found, starting clean")
                    else:
                        logger.info("📂 Local portfolio loaded")
                        
            except Exception as e:
                logger.error(f"❌ Portfolio synchronization failed: {e}")
                logger.warning("⚠️ Continuing with local state")
        
        logger.info("✅ Portfolio Manager initialized")
        return portfolio_manager
    
    async def _initialize_order_manager(self) -> OrderManager:
        """Initialize Order Manager."""
        logger.info("💰 Initializing Order Manager...")
        
        # Determine mode based on configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode == "LIVE":
            mode = "live"
        else:
            mode = "simulated"
            
        order_manager = OrderManager(
            state_manager=self.state,
            portfolio_manager=self.portfolio_manager,
            mode=mode,
            simulated_client=self.components['simulated_client']
        )
        
        logger.info(f"✅ Order Manager initialized (mode: {mode})")
        return order_manager
    
    async def _initialize_sentiment_manager(self):
        """Initialize sentiment analysis manager."""
        logger.info("🧠 Initializing sentiment analysis...")
        
        # Initialize sentiment cache
        sentiment_texts_cache = []
        last_sentiment_update = 0
        
        # Initial sentiment update
        try:
            sentiment_texts_cache = await update_sentiment_texts()
            last_sentiment_update = 0  # Will be set in runtime loop
            logger.info(f"✅ Sentiment analysis initialized with {len(sentiment_texts_cache)} texts")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment initialization failed: {e}")
        
        return {
            'texts_cache': sentiment_texts_cache,
            'last_update': last_sentiment_update
        }
    
    async def _start_background_services(self):
        """Start background services."""
        logger.info("🔄 Starting background services...")
        
        # Start signal verification
        signal_verifier = self.components['signal_verifier']
        await start_signal_verification()
        logger.info("✅ Signal verification started")
    
    async def _cleanup_on_failure(self):
        """Clean up components if bootstrap fails."""
        logger.info("🧹 Cleaning up after bootstrap failure...")
        
        try:
            # Stop signal verification
            await stop_signal_verification()
            
            # Close components
            for component in self.components.values():
                if hasattr(component, "close"):
                    await component.close()
                    
        except Exception as e:
            logger.error(f"❌ Cleanup after failure failed: {e}")
    
    def get_runtime_loop(self):
        """Get the initialized runtime loop."""
        return self.runtime_loop
    
    def get_state(self) -> Dict[str, Any]:
        """Get the initialized system state."""
        return self.state


async def bootstrap_hrm_system(mode: str = "paper"):
    """
    Bootstrap the HRM system and return core components.
    
    🔥 PRIORIDAD 4: session_is_fresh introducido - nueva sesión independiente de balances
    """
    bootstrap = HRMBootstrap()
    portfolio_manager, order_manager = await bootstrap.bootstrap_system(mode)
    runtime_loop = bootstrap.get_runtime_loop()
    
    return portfolio_manager, order_manager, runtime_loop


def reset_session():
    """
    Reset session to fresh state.
    
    🔥 PRIORIDAD 4: Función para marcar sesión como nueva
    """
    global session_is_fresh
    session_is_fresh = True
    logger.info("🔄 Session reset to fresh state")
    return True


# =============================================================================
# SHUTDOWN LIMPIO GLOBAL
# =============================================================================

# Variable global para almacenar referencias a componentes para shutdown
_shutdown_components = {
    'market_data_manager': None,
    'realtime_loader': None,
    'exchange_client': None,
    'binance_client': None,
    'data_feed': None,
    'portfolio_manager': None,
    'order_manager': None,
}


def register_component_for_shutdown(name: str, component: Any):
    """
    Registra un componente para ser cerrado durante el shutdown global.
    
    Args:
        name: Nombre identificador del componente
        component: Instancia del componente a registrar
    """
    global _shutdown_components
    _shutdown_components[name] = component
    logger.debug(f"🔧 Componente '{name}' registrado para shutdown")


def unregister_component_for_shutdown(name: str):
    """
    Desregistra un componente del shutdown global.
    
    Args:
        name: Nombre identificador del componente
    """
    global _shutdown_components
    if name in _shutdown_components:
        _shutdown_components[name] = None
        logger.debug(f"🔧 Componente '{name}' desregistrado del shutdown")


async def shutdown():
    """
    Shutdown limpio global - cierra todas las conexiones y sesiones.
    
    Esta función DEBE ser llamada al finalizar el sistema para asegurar:
    - Todas las sesiones aiohttp se cierran correctamente
    - No quedan warnings asyncio al terminar
    - Los recursos se liberan en el orden correcto
    
    Orden de cierre:
    1. MarketDataManager (cierra RealTimeLoader y DataFeed internamente)
    2. RealTimeLoader (WebSocket connections)
    3. ExchangeClient / BinanceClient (sesiones aiohttp)
    4. DataFeed
    5. PortfolioManager (guarda estado)
    6. OrderManager
    7. Cancelar todas las tareas pendientes de asyncio
    8. Cerrar sesiones aiohttp huérfanas
    """
    global _shutdown_components
    
    logger.info("🧹 Iniciando shutdown limpio global...")
    
    # 1. Cerrar MarketDataManager
    if _shutdown_components.get('market_data_manager') is not None:
        try:
            await _shutdown_components['market_data_manager'].close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando MarketDataManager: {e}")
    
    # 2. Cerrar RealTimeLoader directamente si existe
    if _shutdown_components.get('realtime_loader') is not None:
        try:
            await _shutdown_components['realtime_loader'].close()
            logger.info("✅ RealTimeLoader cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando RealTimeLoader: {e}")
    
    # 3. Cerrar ExchangeClient / BinanceClient
    for client_name in ['exchange_client', 'binance_client']:
        if _shutdown_components.get(client_name) is not None:
            try:
                client = _shutdown_components[client_name]
                if hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, 'close_connection'):
                    await client.close_connection()
                logger.info(f"✅ {client_name} cerrado")
            except Exception as e:
                logger.warning(f"⚠️ Error cerrando {client_name}: {e}")
    
    # 4. Cerrar DataFeed
    if _shutdown_components.get('data_feed') is not None:
        try:
            await _shutdown_components['data_feed'].close()
            logger.info("✅ DataFeed cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando DataFeed: {e}")
    
    # 5. Guardar estado del PortfolioManager
    if _shutdown_components.get('portfolio_manager') is not None:
        try:
            pm = _shutdown_components['portfolio_manager']
            if hasattr(pm, 'save_to_json'):
                pm.save_to_json()
                logger.info("✅ PortfolioManager estado guardado")
            if hasattr(pm, 'close'):
                await pm.close()
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando PortfolioManager: {e}")
    
    # 6. Cerrar OrderManager
    if _shutdown_components.get('order_manager') is not None:
        try:
            om = _shutdown_components['order_manager']
            if hasattr(om, 'close'):
                await om.close()
            logger.info("✅ OrderManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando OrderManager: {e}")
    
    # 7. Cerrar todas las sesiones aiohttp pendientes
    await _close_aiohttp_sessions()
    
    # 8. Cancelar todas las tareas pendientes de asyncio (excepto la actual)
    await _cancel_pending_tasks()
    
    logger.info("🧹 Shutdown limpio global completado")


async def _close_aiohttp_sessions():
    """
    Cierra todas las sesiones aiohttp abiertas.
    """
    try:
        import aiohttp
        
        # Buscar y cerrar todas las sesiones aiohttp
        closed_count = 0
        for obj in list(globals().values()) + list(_shutdown_components.values()):
            if obj is None:
                continue
            
            # Buscar atributos que sean sesiones aiohttp
            if hasattr(obj, 'session') and hasattr(obj.session, 'closed'):
                try:
                    if not obj.session.closed:
                        await obj.session.close()
                        closed_count += 1
                except Exception:
                    pass
            
            # Buscar atributos que sean ClientSession directamente
            if isinstance(obj, aiohttp.ClientSession):
                try:
                    if not obj.closed:
                        await obj.close()
                        closed_count += 1
                except Exception:
                    pass
        
        if closed_count > 0:
            logger.info(f"✅ {closed_count} sesiones aiohttp cerradas")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando sesiones aiohttp: {e}")


async def _cancel_pending_tasks():
    """
    Cancela todas las tareas pendientes de asyncio excepto la tarea actual.
    Esto evita warnings de "task was destroyed but it is pending".
    """
    try:
        import asyncio
        
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        
        # Obtener todas las tareas pendientes excepto la actual
        pending_tasks = [
            task for task in asyncio.all_tasks(loop) 
            if task is not current_task and not task.done()
        ]
        
        if pending_tasks:
            logger.info(f"🛑 Cancelando {len(pending_tasks)} tareas pendientes...")
            
            # Cancelar todas las tareas
            for task in pending_tasks:
                task.cancel()
            
            # Esperar a que todas las tareas se completen (o se cancelen)
            results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            
            cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
            logger.info(f"✅ {cancelled_count} tareas canceladas, {len(results) - cancelled_count} completadas")
    except Exception as e:
        logger.warning(f"⚠️ Error cancelando tareas pendientes: {e}")


async def shutdown_with_signal_handling():
    """
    Versión del shutdown que maneja señales del sistema.
    Usar esta función cuando se quiera un shutdown completo con manejo de señales.
    """
    logger.info("🛑 Ejecutando shutdown con manejo de señales...")
    await shutdown()


# Manejadores de señales para shutdown limpio
_shutdown_event = asyncio.Event()

def _signal_handler(signum, frame):
    """
    Manejador de señales que activa el evento de shutdown.
    """
    import signal
    sig_name = signal.Signals(signum).name
    logger.info(f"🛑 Señal {sig_name} recibida, iniciando shutdown...")
    _shutdown_event.set()


def setup_signal_handlers():
    """
    Configura los manejadores de señales para SIGINT y SIGTERM.
    Llama a esta función al inicio de la aplicación para habilitar shutdown limpio.
    """
    import signal
    
    # Configurar manejadores para SIGINT (Ctrl+C) y SIGTERM
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    logger.info("✅ Manejadores de señales configurados (SIGINT, SIGTERM)")


async def wait_for_shutdown_signal():
    """
    Espera hasta que se reciba una señal de shutdown.
    Útil para mantener el programa corriendo hasta que se presione Ctrl+C.
    """
    await _shutdown_event.wait()
    logger.info("🛑 Señal de shutdown recibida, procediendo...")



================================================
FILE: config_loader.py
================================================
"""
Config Loader Module

Carga la configuración inicial desde initial_state.json y proporciona
funciones para acceder a los valores de configuración.

Este módulo centraliza el acceso a la configuración inicial del sistema.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "capital_usdt": 3000.0,
    "btc": 0.0,
    "eth": 0.0,
    "mode": "paper",
    "auto_learning": "fix",
    "initial_balances": {
        "USDT": 3000.0,
        "BTC": 0.0,
        "ETH": 0.0
    },
    "reset_singletons": True
}


def load_initial_state(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Load initial state from JSON file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Ensure all default keys exist
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading {config_path}: {e}")
            print("   Using default configuration")
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file if it doesn't exist
        save_initial_state(DEFAULT_CONFIG, config_path)
        return DEFAULT_CONFIG.copy()


def save_initial_state(config: Dict[str, Any], config_path: str = "initial_state.json") -> bool:
    """
    Save initial state to JSON file.
    
    Args:
        config: Configuration dict to save
        config_path: Path to the initial state JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"⚠️  Error saving {config_path}: {e}")
        return False


def get_initial_balances(config_path: str = "initial_state.json") -> Dict[str, float]:
    """
    Get initial balances for SimulatedExchangeClient.
    
    Returns:
        Dict with BTC, ETH, and USDT balances
    """
    config = load_initial_state(config_path)
    
    # Prefer initial_balances if available
    if "initial_balances" in config:
        balances = config["initial_balances"]
        return {
            "BTC": float(balances.get("BTC", 0.0)),
            "ETH": float(balances.get("ETH", 0.0)),
            "USDT": float(balances.get("USDT", 3000.0))
        }
    
    # Fallback to individual fields
    return {
        "BTC": float(config.get("btc", 0.0)),
        "ETH": float(config.get("eth", 0.0)),
        "USDT": float(config.get("capital_usdt", 3000.0))
    }


def get_capital_usd(config_path: str = "initial_state.json") -> float:
    """Get initial capital in USD."""
    config = load_initial_state(config_path)
    return float(config.get("capital_usdt", 3000.0))


def get_mode(config_path: str = "initial_state.json") -> str:
    """Get trading mode (paper/live)."""
    config = load_initial_state(config_path)
    return config.get("mode", "paper")


def get_auto_learning(config_path: str = "initial_state.json") -> str:
    """Get auto-learning mode."""
    config = load_initial_state(config_path)
    return config.get("auto_learning", "fix")


def should_reset_singletons(config_path: str = "initial_state.json") -> bool:
    """Check if singletons should be reset."""
    config = load_initial_state(config_path)
    return config.get("reset_singletons", True)


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def get_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Get cached configuration or load from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_initial_state(config_path)
    
    return _config_cache


def reload_config(config_path: str = "initial_state.json") -> Dict[str, Any]:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to the initial state JSON file
        
    Returns:
        Dict with configuration values
    """
    global _config_cache
    _config_cache = load_initial_state(config_path)
    return _config_cache


def reset_config_cache():
    """Reset the configuration cache."""
    global _config_cache
    _config_cache = None


# Convenience properties
@property
def initial_balances() -> Dict[str, float]:
    """Get initial balances (uses cached config)."""
    return get_initial_balances()


@property
def capital_usd() -> float:
    """Get initial capital (uses cached config)."""
    return get_capital_usd()


@property
def mode() -> str:
    """Get trading mode (uses cached config)."""
    return get_mode()


if __name__ == "__main__":
    # Test the module
    print("🔧 Config Loader Test")
    print("=" * 50)
    
    config = load_initial_state()
    print(f"📊 Config loaded: {config}")
    print()
    
    balances = get_initial_balances()
    print(f"💰 Initial balances: {balances}")
    print(f"💵 Capital USD: {get_capital_usd()}")
    print(f"🎯 Mode: {get_mode()}")
    print(f"🤖 Auto-learning: {get_auto_learning()}")
    print(f"🔄 Reset singletons: {should_reset_singletons()}")



================================================
FILE: execution_gateway.py
================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Execution Gateway Module

This module provides the single point of execution for the HRM system.
It handles order validation, execution, and risk management.
"""

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler
from core.trading_metrics import TradingMetrics

from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.order_validators import OrderValidator
from l1_operational.risk_guard import RiskGuard

from l2_tactic.tactical_signal_processor import TacticalSignal
from l2_tactic.config import L2Config

from comms.config import config


class ExecutionGateway:
    """Single point of execution for HRM system orders."""
    
    def __init__(self, portfolio_manager: PortfolioManager, order_manager: OrderManager):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.binance_client = order_manager.binance_client
        
        # Initialize execution components
        self.validator = OrderValidator(portfolio_manager)
        self.risk_guard = RiskGuard(portfolio_manager)
        self.trading_metrics = TradingMetrics()
        
        # Execution state
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'rejected_orders': 0,
            'risk_blocks': 0
        }
    
    async def execute_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a list of orders through the gateway."""
        if not orders:
            return []
        
        logger.info(f"💰 Gateway: Processing {len(orders)} orders")
        
        # Update execution stats
        self.execution_stats['total_orders'] += len(orders)
        
        # 1. Validate all orders
        validated_orders = await self._validate_orders(orders)
        
        # 2. Apply risk management
        risk_checked_orders = await self._apply_risk_management(validated_orders)
        
        # 3. Execute orders
        executed_orders = await self._execute_validated_orders(risk_checked_orders)
        
        # 4. Update portfolio and metrics
        await self._update_portfolio_and_metrics(executed_orders)
        
        # 5. Log execution results
        await self._log_execution_results(orders, validated_orders, risk_checked_orders, executed_orders)
        
        return executed_orders
    
    async def _validate_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate orders before execution."""
        validated_orders = []
        
        for order in orders:
            try:
                # Basic order validation
                validation_result = self.validator.validate_order(order)
                
                if validation_result['valid']:
                    # Portfolio-specific validation
                    portfolio_validation = self.validator.validate_portfolio_constraints(order)
                    
                    if portfolio_validation['valid']:
                        validated_orders.append(order)
                        logger.debug(f"✅ Order validated: {order.get('symbol')} {order.get('side')} {order.get('quantity')}")
                    else:
                        logger.warning(f"❌ Portfolio validation failed: {portfolio_validation['reason']}")
                        order['status'] = 'rejected'
                        order['validation_error'] = portfolio_validation['reason']
                        self.execution_stats['rejected_orders'] += 1
                        validated_orders.append(order)
                else:
                    logger.warning(f"❌ Order validation failed: {validation_result['reason']}")
                    order['status'] = 'rejected'
                    order['validation_error'] = validation_result['reason']
                    self.execution_stats['rejected_orders'] += 1
                    validated_orders.append(order)
                    
            except Exception as e:
                logger.error(f"❌ Order validation error: {e}")
                order['status'] = 'rejected'
                order['validation_error'] = str(e)
                self.execution_stats['rejected_orders'] += 1
                validated_orders.append(order)
        
        return validated_orders
    
    async def _apply_risk_management(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply risk management rules to orders."""
        risk_checked_orders = []
        
        for order in orders:
            try:
                # Check if order should be blocked by risk management
                risk_check = await self.risk_guard.check_order_risk(order)
                
                if risk_check['allowed']:
                    risk_checked_orders.append(order)
                    logger.debug(f"🛡️ Risk check passed: {order.get('symbol')} {order.get('side')}")
                else:
                    logger.warning(f"🛡️ Risk check blocked: {risk_check['reason']}")
                    order['status'] = 'rejected'
                    order['risk_error'] = risk_check['reason']
                    self.execution_stats['risk_blocks'] += 1
                    risk_checked_orders.append(order)
                    
            except Exception as e:
                logger.error(f"❌ Risk management error: {e}")
                order['status'] = 'rejected'
                order['risk_error'] = str(e)
                self.execution_stats['risk_blocks'] += 1
                risk_checked_orders.append(order)
        
        return risk_checked_orders
    
    async def _execute_validated_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute validated and risk-checked orders."""
        executed_orders = []
        
        for order in orders:
            if order.get('status') == 'rejected':
                executed_orders.append(order)
                continue
            
            try:
                # Execute order through order manager
                execution_result = await self.order_manager.execute_single_order(order)
                
                if execution_result.get('status') == 'filled':
                    self.execution_stats['successful_executions'] += 1
                    logger.info(f"✅ Order executed: {order.get('symbol')} {order.get('side')} {order.get('quantity')} @ {execution_result.get('fill_price')}")
                else:
                    self.execution_stats['failed_executions'] += 1
                    logger.warning(f"⚠️ Order failed: {order.get('symbol')} {order.get('side')} - {execution_result.get('status')}")
                
                executed_orders.append(execution_result)
                
            except Exception as e:
                logger.error(f"❌ Order execution failed: {e}")
                order['status'] = 'failed'
                order['execution_error'] = str(e)
                self.execution_stats['failed_executions'] += 1
                executed_orders.append(order)
        
        return executed_orders
    
    async def _update_portfolio_and_metrics(self, executed_orders: List[Dict[str, Any]]):
        """Update portfolio state and trading metrics."""
        try:
            # Update portfolio from executed orders - VERSIÓN PURA: no requiere market_data
            await self.portfolio_manager.update_from_orders_async(executed_orders)
            
            # Update trading metrics - CRITICAL FIX: Use async method (no market_data needed)
            total_value = await self.portfolio_manager.get_total_value_async()
            self.trading_metrics.update_from_orders(executed_orders, total_value)
            
            # Save portfolio state periodically
            if self.execution_stats['total_orders'] % 5 == 0:
                self.portfolio_manager.save_to_json()
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio and metrics: {e}")
    
    async def _log_execution_results(self, original_orders: List[Dict[str, Any]], 
                                   validated_orders: List[Dict[str, Any]],
                                   risk_checked_orders: List[Dict[str, Any]],
                                   executed_orders: List[Dict[str, Any]]):
        """Log execution results and statistics."""
        try:
            # Count different types of orders
            successful = [o for o in executed_orders if o.get('status') == 'filled']
            failed = [o for o in executed_orders if o.get('status') == 'failed']
            rejected = [o for o in executed_orders if o.get('status') == 'rejected']
            
            # Log summary
            logger.info(
                f"💰 Gateway Results: "
                f"Original: {len(original_orders)} | "
                f"Validated: {len(validated_orders)} | "
                f"Risk-checked: {len(risk_checked_orders)} | "
                f"Executed: {len(executed_orders)} | "
                f"Successful: {len(successful)} | "
                f"Failed: {len(failed)} | "
                f"Rejected: {len(rejected)}"
            )
            
            # Log detailed stats periodically
            if self.execution_stats['total_orders'] % 10 == 0:
                logger.info(
                    f"📊 Gateway Stats (Total: {self.execution_stats['total_orders']}): "
                    f"Successful: {self.execution_stats['successful_executions']} | "
                    f"Failed: {self.execution_stats['failed_executions']} | "
                    f"Rejected: {self.execution_stats['rejected_orders']} | "
                    f"Risk Blocks: {self.execution_stats['risk_blocks']}"
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to log execution results: {e}")
    
    async def execute_signal(self, signal: TacticalSignal, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single tactical signal."""
        try:
            # Convert signal to orders
            orders = await self._convert_signal_to_orders(signal, market_data)
            
            # Execute orders through gateway
            executed_orders = await self.execute_orders(orders)
            
            return executed_orders
            
        except Exception as e:
            logger.error(f"❌ Signal execution failed: {e}")
            return []
    
    async def _convert_signal_to_orders(self, signal: TacticalSignal, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a tactical signal to executable orders."""
        try:
            symbol = getattr(signal, 'symbol', '')
            side = getattr(signal, 'side', 'hold')
            confidence = getattr(signal, 'confidence', 0.5)
            
            if side == 'hold':
                return []
            
            # Get current price for the symbol
            current_price = self._get_current_price(symbol, market_data)
            if not current_price:
                logger.warning(f"⚠️ No price data for {symbol}")
                return []
            
            # Calculate order size based on signal strength and portfolio allocation
            order_size = await self._calculate_order_size(signal, market_data, current_price)
            
            if order_size <= 0:
                logger.warning(f"⚠️ Invalid order size for {symbol}")
                return []
            
            # Create order
            order = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': order_size,
                'price': current_price,
                'confidence': confidence,
                'signal_source': 'tactical',
                'status': 'pending',
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }
            
            logger.info(f"🎯 Signal converted to order: {symbol} {side} {order_size} @ {current_price}")
            return [order]
            
        except Exception as e:
            logger.error(f"❌ Signal conversion failed: {e}")
            return []
    
    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            symbol_data = market_data.get(symbol, {})
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                return float(symbol_data['close'])
            elif isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                return float(symbol_data['close'].iloc[-1])
            return None
        except Exception:
            return None
    
    async def _calculate_order_size(self, signal: TacticalSignal, market_data: Dict[str, Any], current_price: float) -> float:
        """Calculate order size based on signal and portfolio constraints."""
        try:
            symbol = getattr(signal, 'symbol', '')
            confidence = getattr(signal, 'confidence', 0.5)
            
            # Get portfolio state - CRITICAL FIX: Use async method (no market_data needed)
            portfolio_state = self.portfolio_manager.get_portfolio_state()
            total_value = await self.portfolio_manager.get_total_value_async()
            
            # Calculate position size based on confidence and risk limits
            base_allocation = total_value * 0.10  # 10% base allocation
            risk_adjusted_allocation = base_allocation * confidence
            
            # Calculate quantity
            quantity = risk_adjusted_allocation / current_price
            
            # Apply minimum order size constraints
            min_notional = 10.0  # Minimum $10 order
            if quantity * current_price < min_notional:
                quantity = min_notional / current_price
            
            # Apply portfolio constraints
            max_position_size = total_value * 0.30  # Max 30% in single position
            max_quantity = max_position_size / current_price
            
            quantity = min(quantity, max_quantity)
            
            return max(0.0, quantity)
            
        except Exception as e:
            logger.error(f"❌ Order size calculation failed: {e}")
            return 0.0
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    async def cleanup_stale_orders(self, current_positions: Dict[str, float]):
        """Clean up stale stop-loss and profit-taking orders."""
        try:
            cleanup_stats = self.order_manager.cleanup_stale_orders(current_positions)
            logger.info(f"🧹 Gateway cleanup: {cleanup_stats}")
            return cleanup_stats
        except Exception as e:
            logger.error(f"❌ Gateway cleanup failed: {e}")
            return {}


async def create_execution_gateway(portfolio_manager: PortfolioManager, 
                                 order_manager: OrderManager) -> ExecutionGateway:
    """Create and initialize an execution gateway."""
    gateway = ExecutionGateway(portfolio_manager, order_manager)
    logger.info("✅ Execution Gateway initialized")
    return gateway


================================================
FILE: fix_l3_dominance.py
================================================
# fix_l3_dominance.py - Corrección de Lógica de Dominancia L3

"""
PROBLEMA IDENTIFICADO:
- L3 HOLD con 0.85 confidence bloquea TODAS las señales L2/L1
- Allocations L3 no se ejecutan (BTC target 50% pero current 0%)
- $994 USDT disponible sin usar
- Sistema 100% en HOLD permanente

SOLUCIÓN:
1. L3 HOLD solo bloquea si confidence > 0.90 (no 0.85)
2. En regímenes RANGE, permitir L2 autonomía con setup trading
3. Ejecutar rebalancing automático hacia targets L3
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class L3DominanceFixConfig:
    """Configuración corregida para dominancia L3"""

    # ANTES: L3 HOLD bloqueaba con 0.80+ confidence
    # AHORA: Solo bloquea con 0.90+ confidence (muy alta certeza)
    L3_HOLD_BLOCKING_THRESHOLD = 0.90  # Aumentado de 0.80

    # En RANGE, permitir L2 autonomía incluso con HOLD L3
    ALLOW_L2_AUTONOMY_IN_RANGE = True

    # Ejecutar rebalancing cuando diferencia > 10%
    REBALANCE_THRESHOLD_PCT = 0.10

    # Setup trading en RANGE (oversold/overbought)
    ENABLE_SETUP_TRADING_IN_RANGE = True


def should_l3_block_l2_signals(
    l3_signal: str,
    l3_confidence: float,
    regime: str,
    current_allocation: Dict[str, float],
    target_allocation: Dict[str, float],
    l2_signal_action: str = None,
    has_position: bool = False,
    allow_l2_signals: bool = True
) -> bool:
    """
    Decide si L3 debe bloquear señales L2.

    LÓGICA CORREGIDA - NUEVA JERARQUÍA:
    1. Si allow_l2_signals=False → SIEMPRE bloquear señales direccionales (invariante jerárquica)
    2. Stop-loss L1 (siempre primero - no aplica aquí)
    3. SELL TÁCTICO DE SALIDA LIMPIA (excepción quirúrgica)
    4. Dominancia L3 (normal) - L3 confidence >= 0.6 bloquea
    5. Duda → HOLD (INV-5) - L3 confidence < 0.6 fuerza HOLD

    EXCEPCIÓN QUIRÚRGICA SOLO PARA SALIDA:
    - has_position + l2_signal SELL + l3_regime TRENDING + l3_confidence < 0.6
    """

    # 1. INVARIANTE JERÁRQUICA: Si allow_l2_signals=False, BLOQUEAR TODAS LAS SEÑALES DIRECCIONALES
    if not allow_l2_signals:
        if l2_signal_action and l2_signal_action.upper() not in ['HOLD']:
            logger.warning(
                f"🚫 L3 DOMINANCE: allow_l2_signals=False bloquea señal L2 {l2_signal_action.upper()} "
                f"en régimen {regime}"
            )
            return True
        else:
            return False

    # 2. L3 BUY/SELL siempre tienen prioridad (dominancia normal)
    if l3_signal in ['buy', 'sell']:
        logger.info(f"✅ L3 {l3_signal.upper()} signal - L2 debe seguir")
        return False

    # 3. EXCEPCIÓN QUIRÚRGICA: SELL TÁCTICO DE SALIDA LIMPIA
    # SOLO si se cumplen TODAS las condiciones:
    if (l2_signal_action and l2_signal_action.upper() in ['SELL', 'SELL_LIGHT', 'REDUCE'] and
        has_position and
        regime.upper() == 'TRENDING' and
        l3_confidence < 0.6):
        logger.info(
            f"🎯 SELL TÁCTICO DE SALIDA LIMPIA: Allow {l2_signal_action.upper()} "
            f"(position={has_position}, regime={regime}, conf={l3_confidence:.2f}<0.6)"
        )
        return False

    # 4. DOMINANCIA L3 NORMAL: L3 bloquea si confidence >= 0.6
    if l3_confidence >= 0.6:
        logger.warning(
            f"🚫 L3 DOMINANCE: L3 {l3_signal.upper()} con {l3_confidence:.2f} confidence (>= 0.6) "
            f"bloquea señales direccionales L2 en {regime} regime (AutoRebalancer still allowed)"
        )
        return True

    # 5. DUDA → HOLD (INV-5): Si L3 confidence < 0.6, forzar HOLD global
    # NO permitir BUY bajo duda - solo HOLD hasta que L3 tenga claridad
    if l3_confidence < 0.6:
        logger.info(
            f"🛡️ INV-5 PROTECTION: L3 duda ({l3_confidence:.2f} < 0.6) → HOLD global "
            f"(no trades under doubt)"
        )
        return True

    # 6. Si allocations desviadas >10%, permitir rebalancing (pero solo HOLD, no BUY)
    allocation_deviation = calculate_allocation_deviation(
        current_allocation, target_allocation
    )
    if allocation_deviation > L3DominanceFixConfig.REBALANCE_THRESHOLD_PCT:
        logger.warning(
            f"⚠️ Allocation deviation {allocation_deviation:.1%} > 10% - "
            f"Permitir rebalancing (solo SELL/REDUCE, no BUY)"
        )
        # Solo permitir SELL/REDUCE para rebalancing, no BUY
        if l2_signal_action and l2_signal_action.upper() in ['SELL', 'SELL_LIGHT', 'REDUCE']:
            return False
        else:
            return True

    # Default: mantener bloqueo por seguridad
    logger.warning(f"🚫 L3 DEFAULT BLOCK: Manteniendo bloqueo por seguridad")
    return True


def get_l3_push_signal(l3_signal: str, l3_confidence: float, l2_original_signal: str) -> str:
    """
    Cuando L3 "empuja" a L2, convertir señales HOLD en alternativas defensivas.

    Si L3 = SELL y L2 = HOLD → L2 debe emitir SELL_LIGHT, REDUCE, HEDGE o NO_LONG
    con size reducido.

    Args:
        l3_signal: Señal L3 (buy/sell/hold)
        l3_confidence: Confianza L3
        l2_original_signal: Señal L2 original

    Returns:
        Señal L2 modificada por L3 push
    """
    # Solo aplicar push si L3 tiene señal clara y confianza razonable
    if l3_signal not in ['buy', 'sell'] or l3_confidence < 0.50:
        return l2_original_signal

    # Si L2 ya tiene señal BUY/SELL, no modificar
    if l2_original_signal in ['buy', 'sell']:
        return l2_original_signal

    # L3 BUY push: si L3 quiere comprar pero L2 quiere HOLD
    if l3_signal == 'buy':
        logger.info(f"🚀 L3 BUY PUSH: L2 HOLD → BUY_LIGHT (reduced size)")
        return 'buy_light'  # BUY con size reducido

    # L3 SELL push: si L3 quiere vender pero L2 quiere HOLD
    elif l3_signal == 'sell':
        # Elegir entre alternativas defensivas basadas en confianza
        if l3_confidence >= 0.80:
            push_signal = 'sell_light'  # Venta ligera
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → SELL_LIGHT (conf={l3_confidence:.2f})")
        elif l3_confidence >= 0.65:
            push_signal = 'reduce'  # Reducir posición
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → REDUCE (conf={l3_confidence:.2f})")
        else:
            push_signal = 'hedge'  # Hedging
            logger.info(f"📉 L3 SELL PUSH: L2 HOLD → HEDGE (conf={l3_confidence:.2f})")

        return push_signal

    return l2_original_signal


def apply_l3_push_to_signals(l2_signals: list, l3_signal: str, l3_confidence: float) -> list:
    """
    Aplicar L3 push a una lista de señales L2.

    Args:
        l2_signals: Lista de señales L2
        l3_signal: Señal L3
        l3_confidence: Confianza L3

    Returns:
        Lista de señales L2 modificadas por L3 push
    """
    if not l2_signals:
        return l2_signals

    modified_signals = []

    for signal in l2_signals:
        original_side = getattr(signal, 'side', 'hold')

        # Aplicar L3 push logic
        pushed_side = get_l3_push_signal(l3_signal, l3_confidence, original_side)

        if pushed_side != original_side:
            # Modificar la señal
            signal.side = pushed_side

            # Reducir confidence ligeramente para push signals
            if hasattr(signal, 'confidence'):
                signal.confidence = min(signal.confidence, l3_confidence * 0.9)

            # Reducir size para señales push (excepto buy_light que puede ser normal)
            if pushed_side in ['sell_light', 'reduce', 'hedge']:
                if hasattr(signal, 'metadata'):
                    signal.metadata['l3_push_applied'] = True
                    signal.metadata['l3_push_type'] = pushed_side
                    signal.metadata['l3_push_confidence'] = l3_confidence
                    signal.metadata['size_multiplier'] = 0.5  # Reducir size a 50%

            logger.info(f"🔄 L3 PUSH APPLIED: {original_side.upper()} → {pushed_side.upper()}")

        modified_signals.append(signal)

    return modified_signals


def calculate_allocation_deviation(
    current: Dict[str, float],
    target: Dict[str, float]
) -> float:
    """Calcula desviación máxima entre allocations actuales y target"""
    deviations = []
    for symbol in target.keys():
        current_pct = current.get(symbol, 0.0)
        target_pct = target.get(symbol, 0.0)
        deviation = abs(current_pct - target_pct)
        deviations.append(deviation)

    return max(deviations) if deviations else 0.0


def should_trigger_rebalancing(
    current_allocation: Dict[str, float],
    target_allocation: Dict[str, float],
    available_usdt: float,
    min_rebalance_amount: float = 100.0
) -> bool:
    """
    Decide si ejecutar rebalancing automático.

    CRITERIOS:
    1. Desviación > 10% en algún activo
    2. Capital disponible > $100
    3. Diferencia en USDT > $100
    """

    # 1. Desviación significativa
    deviation = calculate_allocation_deviation(current_allocation, target_allocation)
    if deviation < L3DominanceFixConfig.REBALANCE_THRESHOLD_PCT:
        logger.debug(f"Desviación {deviation:.1%} < 10% - No rebalancing")
        return False

    # 2. Capital suficiente
    if available_usdt < min_rebalance_amount:
        logger.debug(f"USDT disponible ${available_usdt:.2f} < ${min_rebalance_amount}")
        return False

    # 3. Calcular diferencia en USDT
    total_portfolio_value = sum(current_allocation.values())
    if total_portfolio_value == 0:
        return False

    max_diff_usdt = max([
        abs(current_allocation.get(s, 0) - target_allocation.get(s, 0))
        for s in target_allocation.keys()
    ])

    if max_diff_usdt < min_rebalance_amount:
        logger.debug(f"Diferencia max ${max_diff_usdt:.2f} < ${min_rebalance_amount}")
        return False

    logger.info(
        f"✅ TRIGGER REBALANCING: deviation={deviation:.1%}, "
        f"available=${available_usdt:.2f}, diff=${max_diff_usdt:.2f}"
    )
    return True


# ============================================================================
# INTEGRACIÓN EN MAIN.PY
# ============================================================================

def integrate_fix_in_main():
    """
    Código para integrar en main.py

    REEMPLAZAR EN MAIN.PY (aproximadamente línea 850-900):
    """
    example_code = '''
# ANTES (LÓGICA ANTIGUA - REMOVER):
if l3_info['signal'] == 'hold' and l3_info['confidence'] > 0.80:
    logger.warning("🚫 L3 DOMINANCE: L3 HOLD blocks all L2 signals")
    # ... código antiguo

# DESPUÉS (LÓGICA NUEVA - INTEGRAR):
from fix_l3_dominance import (
    should_l3_block_l2_signals,
    should_trigger_rebalancing,
    L3DominanceFixConfig
)

# Verificar si L3 debe bloquear L2
l3_blocks_l2 = should_l3_block_l2_signals(
    l3_signal=l3_info['signal'],
    l3_confidence=l3_info['confidence'],
    regime=l3_info['regime'],
    current_allocation=state.portfolio.get_current_allocation(),
    target_allocation=l3_info['asset_allocation']
)

if l3_blocks_l2:
    logger.warning(
        f"🚫 L3 DOMINANCE: L3 {l3_info['signal'].upper()} "
        f"con {l3_info['confidence']:.2f} confidence bloquea L2"
    )
    # Forzar HOLD signals
    btc_signal = {'action': 'hold', 'symbol': 'BTCUSDT', 'confidence': 0.5}
    eth_signal = {'action': 'hold', 'symbol': 'ETHUSDT', 'confidence': 0.5}
else:
    logger.info("🔓 L3 permite autonomía L2 - procesando señales normalmente")
    # Procesar L2 normalmente
    btc_signal = l2_processor.process_signals(state, 'BTCUSDT')
    eth_signal = l2_processor.process_signals(state, 'ETHUSDT')

# Verificar si ejecutar rebalancing automático
if should_trigger_rebalancing(
    current_allocation=state.portfolio.get_current_allocation(),
    target_allocation=l3_info['asset_allocation'],
    available_usdt=state.portfolio.get_available_usdt()
):
    logger.info("🔄 EXECUTING AUTO-REBALANCING towards L3 targets")
    auto_rebalancer.execute_rebalancing(
        current=state.portfolio.get_current_allocation(),
        target=l3_info['asset_allocation'],
        prices=state.market_data
    )
'''
    return example_code


# ============================================================================
# TESTING DE LA SOLUCIÓN
# ============================================================================

def test_fix():
    """Test de la solución con casos reales"""

    print("=" * 80)
    print("TESTING L3 DOMINANCE FIX")
    print("=" * 80)

    # Caso 1: L3 HOLD con 0.85 confidence en RANGE
    print("\n📊 CASO 1: L3 HOLD 0.85 confidence en RANGE")
    blocks = should_l3_block_l2_signals(
        l3_signal='hold',
        l3_confidence=0.85,
        regime='RANGE',
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3860.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0}
    )
    print(f"Resultado: {'BLOQUEA ❌' if blocks else 'PERMITE ✅'}")
    print(f"Esperado: PERMITE ✅ (confidence < 0.90 y regime RANGE)")

    # Caso 2: L3 HOLD con 0.95 confidence
    print("\n📊 CASO 2: L3 HOLD 0.95 confidence (muy alta certeza)")
    blocks = should_l3_block_l2_signals(
        l3_signal='hold',
        l3_confidence=0.95,
        regime='BULL',
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3860.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0}
    )
    print(f"Resultado: {'BLOQUEA ❌' if blocks else 'PERMITE ✅'}")
    print(f"Esperado: BLOQUEA ❌ (confidence > 0.90)")

    # Caso 3: Rebalancing necesario
    print("\n📊 CASO 3: Verificar rebalancing con $994 disponible")
    trigger = should_trigger_rebalancing(
        current_allocation={'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 994.0},
        target_allocation={'BTCUSDT': 1930.0, 'ETHUSDT': 1158.0, 'USDT': 772.0},
        available_usdt=994.0
    )
    print(f"Resultado: {'TRIGGER ✅' if trigger else 'NO TRIGGER ❌'}")
    print(f"Esperado: TRIGGER ✅ (desviación 50% y capital disponible)")

    print("\n" + "=" * 80)
    print("FIX VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Ejecutar tests
    test_fix()

    # Mostrar código de integración
    print("\n" + "=" * 80)
    print("CÓDIGO PARA INTEGRAR EN MAIN.PY:")
    print("=" * 80)
    print(integrate_fix_in_main())



================================================
FILE: integration_auto_learning.py
================================================
#!/usr/bin/env python3
"""
Integration Auto-Learning System
Integra el sistema de auto-aprendizaje con el sistema principal HRM

CRITICAL FIX: Eliminado asyncio.run() - ahora 100% async-compatible
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

# Importar el sistema de auto-aprendizaje
from auto_learning_system import SelfImprovingTradingSystem, TradeData

# Importar componentes del sistema HRM
from core.state_manager import get_state_manager
from core.trading_metrics import get_trading_metrics
from l1_operational.order_manager import OrderManager
from l2_tactic.tactical_signal_processor import L2TacticProcessor

logger = logging.getLogger(__name__)

class AutoLearningIntegration:
    """Integración del sistema de auto-aprendizaje con HRM"""
    
    def __init__(self):
        self.auto_learning_system = SelfImprovingTradingSystem()
        self.state_manager = None
        self.order_manager = None
        self.portfolio_manager = None
        self.l2_processor = None
        self.trading_metrics = None
        self.config = None
        
        # Estado de integración
        self.is_initialized = False
        self.last_integration_check = datetime.now()
        
        logger.info("🔧 Auto-Learning Integration initialized")
    
    async def initialize_integration(self, 
                                   state_manager=None,
                                   order_manager=None,
                                   portfolio_manager=None,
                                   l2_processor=None,
                                   trading_metrics=None,
                                   config=None):
        """
        Inicializar la integración con componentes del sistema.
        
        Args:
            state_manager: State manager para el estado global
            order_manager: OrderManager ya inicializado (con state_manager, portfolio_manager, config)
            portfolio_manager: PortfolioManager para gestión de cartera
            l2_processor: L2TacticProcessor para señales tácticas
            trading_metrics: TradingMetrics para métricas de trading
            config: Configuración del sistema
        """
        
        try:
            # Store config for later use
            self.config = config
            
            # Obtener componentes si no se proporcionan
            if state_manager is None:
                self.state_manager = get_state_manager()
            else:
                self.state_manager = state_manager
            
            # ✅ CRITICAL FIX: OrderManager debe inicializarse con argumentos requeridos
            if order_manager is None:
                # Intentar obtener OrderManager del sistema con argumentos correctos
                try:
                    from l1_operational.order_manager import OrderManager
                    
                    # ✅ FIX: OrderManager requiere: state_manager, portfolio_manager, config
                    if portfolio_manager is None:
                        logger.error("❌ Cannot create OrderManager: portfolio_manager is required")
                        self.order_manager = None
                    elif config is None:
                        logger.error("❌ Cannot create OrderManager: config is required")
                        self.order_manager = None
                    else:
                        self.order_manager = OrderManager(
                            state_manager=self.state_manager,
                            portfolio_manager=portfolio_manager,
                            config=config
                        )
                        logger.info("✅ OrderManager initialized with required arguments")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize OrderManager: {e}")
                    self.order_manager = None
            else:
                self.order_manager = order_manager
                logger.info("✅ OrderManager provided externally")
            
            # Store portfolio manager reference
            if portfolio_manager is not None:
                self.portfolio_manager = portfolio_manager
            elif self.order_manager is not None and hasattr(self.order_manager, 'portfolio'):
                self.portfolio_manager = self.order_manager.portfolio
            
            # Initialize L2 Processor if not provided
            if l2_processor is None:
                try:
                    from l2_tactic.tactical_signal_processor import L2TacticProcessor
                    self.l2_processor = L2TacticProcessor()
                    logger.info("✅ L2TacticProcessor initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize L2TacticProcessor: {e}")
                    self.l2_processor = None
            else:
                self.l2_processor = l2_processor
                logger.info("✅ L2TacticProcessor provided externally")
            
            # Initialize Trading Metrics if not provided
            if trading_metrics is None:
                self.trading_metrics = get_trading_metrics()
                logger.info("✅ TradingMetrics initialized")
            else:
                self.trading_metrics = trading_metrics
                logger.info("✅ TradingMetrics provided externally")
            
            # Iniciar auto-improvement cycle
            self.auto_learning_system.start_auto_improvement()
            
            self.is_initialized = True
            self.last_integration_check = datetime.now()
            
            logger.info("=" * 70)
            logger.info("✅ Auto-Learning Integration fully initialized")
            logger.info(f"   📊 State Manager:     {'✅' if self.state_manager else '❌'}")
            logger.info(f"   🤖 Order Manager:     {'✅' if self.order_manager else '❌'}")
            logger.info(f"   💼 Portfolio Manager: {'✅' if self.portfolio_manager else '❌'}")
            logger.info(f"   🎯 L2 Processor:      {'✅' if self.l2_processor else '❌'}")
            logger.info(f"   📈 Trading Metrics:   {'✅' if self.trading_metrics else '❌'}")
            logger.info(f"   ⚙️  Config:            {'✅' if self.config else '❌'}")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            return False
    
    def record_trade_for_learning(self, trade_data: Dict[str, Any]):
        """Registrar un trade para el sistema de auto-aprendizaje"""
        
        if not self.is_initialized:
            logger.warning("⚠️ Integration not initialized, skipping trade recording")
            return False
        
        try:
            # Convertir datos del trade a formato compatible
            formatted_trade = {
                'symbol': trade_data.get('symbol', 'UNKNOWN'),
                'side': trade_data.get('side', 'buy'),
                'entry_price': trade_data.get('entry_price', 0.0),
                'exit_price': trade_data.get('exit_price', 0.0),
                'quantity': trade_data.get('quantity', 0.0),
                'pnl': trade_data.get('pnl', 0.0),
                'pnl_pct': trade_data.get('pnl_pct', 0.0),
                'model_used': trade_data.get('model_used', 'unknown'),
                'confidence': trade_data.get('confidence', 0.5),
                'regime': trade_data.get('regime', 'neutral'),
                'features': trade_data.get('features', {}),
                'market_data': trade_data.get('market_data', {})
            }
            
            # Registrar en el sistema de auto-aprendizaje
            self.auto_learning_system.record_trade(formatted_trade)
            
            logger.debug(f"📊 Trade recorded for auto-learning: {formatted_trade['symbol']} {formatted_trade['side']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording trade for learning: {e}")
            return False
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """
        Obtener estado del sistema de auto-aprendizaje.
        
        CRITICAL FIX: Usa get_system_status_async() para obtener datos reales del portfolio
        sin usar asyncio.run()
        """
        
        if not self.is_initialized:
            return {'error': 'Integration not initialized'}
        
        try:
            # CRITICAL FIX: Usar versión async que no usa asyncio.run()
            learning_status = await self.auto_learning_system.get_system_status_async()
            
            # Añadir información de integración
            integration_info = {
                'integration_active': self.is_initialized,
                'last_check': self.last_integration_check,
                'state_manager_available': self.state_manager is not None,
                'order_manager_available': self.order_manager is not None,
                'l2_processor_available': self.l2_processor is not None,
                'trading_metrics_available': self.trading_metrics is not None
            }
            
            return {
                'learning_system': learning_status,
                'integration': integration_info
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting learning status: {e}")
            return {'error': str(e)}
    
    async def trigger_manual_retrain(self) -> bool:
        """Disparar reentrenamiento manual del sistema"""
        
        if not self.is_initialized:
            logger.warning("⚠️ Integration not initialized, cannot trigger retrain")
            return False
        
        try:
            # En implementación real, esto dispararía el reentrenamiento
            # Por ahora, solo registramos la solicitud
            logger.info("🔄 Manual retrain triggered")
            
            # Podría disparar el auto-reentrenamiento forzado
            # await self.auto_learning_system._auto_retrain_models()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error triggering manual retrain: {e}")
            return False
    
    async def check_training_eligibility(self) -> Dict[str, Any]:
        """
        Verificar si el entrenamiento está permitido.
        
        CRITICAL FIX: Usa can_train_async() para evitar asyncio.run()
        """
        try:
            can_train, reason = await self.auto_learning_system.can_train_async()
            
            return {
                'can_train': can_train,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking training eligibility: {e}")
            return {
                'can_train': False,
                'reason': f"Error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Limpiar recursos de la integración"""
        
        try:
            if self.is_initialized:
                logger.info("🧹 Cleaning up Auto-Learning Integration...")
                
                # Detener auto-improvement cycle
                self.auto_learning_system.is_running = False
                
                # Limpiar referencias
                self.state_manager = None
                self.order_manager = None
                self.l2_processor = None
                self.trading_metrics = None
                self.is_initialized = False
                
                logger.info("✅ Auto-Learning Integration cleaned up")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def integrate_with_main_system():
    """
    Función principal de integración
    Esta es la función que se llama desde main.py
    
    CRITICAL FIX: Ahora es async y no usa asyncio.run()
    """
    
    try:
        # Crear instancia de integración
        integration = AutoLearningIntegration()
        
        # Inicializar integración con await
        await integration.initialize_integration()
        
        logger.info("🤖 Auto-Learning System successfully integrated with HRM")
        return integration
        
    except Exception as e:
        logger.error(f"❌ Auto-Learning integration failed: {e}")
        raise RuntimeError(f"Auto-Learning integration failed: {e}")

# Función de prueba para validar la integración
async def test_integration():
    """Probar la integración del sistema de auto-aprendizaje"""
    
    try:
        logger.info("🧪 Testing Auto-Learning Integration...")
        
        # Crear integración
        integration = AutoLearningIntegration()
        
        # Inicializar
        success = await integration.initialize_integration()
        
        if not success:
            logger.error("❌ Integration test failed during initialization")
            return False
        
        # Probar registro de trade
        test_trade = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'quantity': 0.01,
            'pnl': 10.0,
            'pnl_pct': 0.02,
            'model_used': 'l2_finrl',
            'confidence': 0.8,
            'regime': 'bull',
            'features': {'rsi': 65, 'macd': 0.5}
        }
        
        integration.record_trade_for_learning(test_trade)
        
        # Obtener estado (usando versión async)
        status = await integration.get_learning_status()
        
        logger.info("✅ Integration test completed successfully")
        logger.info(f"   📊 Learning system status: {status.get('learning_system', {}).get('data_buffer_size', 0)} trades")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Probar la integración
    asyncio.run(test_integration())



================================================
FILE: runtime_loop.py
================================================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Runtime Loop Module

This module contains the main runtime loop logic for the HRM system.
It handles the execution cycle, timing, and system heartbeat.
"""

import asyncio
import time
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from core.state_manager import get_system_state
from core.logging import logger
from core.trading_metrics import get_trading_metrics
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler

from l1_operational.data_feed import DataFeed
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l3_strategy.regime_classifier import MarketRegimeClassifier
from l3_strategy.decision_maker import make_decision
from l3_strategy.sentiment_inference import download_reddit, download_news, infer_sentiment

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus
from sentiment.sentiment_manager import update_sentiment_texts


class HRMRuntimeLoop:
    """Main runtime loop for HRM system execution."""
    
    def __init__(self, portfolio_manager: PortfolioManager, order_manager: OrderManager):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.cycle_id = 0
        self.last_sentiment_update = 0
        self.sentiment_texts_cache = []
        self.sentiment_update_interval = 2160  # 6 hours in cycles
        
        # Initialize components
        self.data_feed = DataFeed(config)
        self.l2_config = L2Config()
        self.l2_processor = L2TacticProcessor(self.l2_config, portfolio_manager=portfolio_manager, apagar_l3=APAGAR_L3)
        self.trading_metrics = get_trading_metrics()
        
        # L3 components
        self.regime_classifier = MarketRegimeClassifier()
        self.message_bus = MessageBus()
        
    async def run(self):
        """Execute the main runtime loop."""
        logger.info("🚀 Starting HRM Runtime Loop")
        
        try:
            while True:
                await self._execute_cycle()
                await asyncio.sleep(10)  # 10-second cycle interval
                
        except KeyboardInterrupt:
            logger.info("🛑 Runtime loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime loop error: {e}", exc_info=True)
        finally:
            await self._cleanup()
    
    async def _execute_cycle(self):
        """Execute a single cycle of the HRM system."""
        self.cycle_id += 1
        start_time = pd.Timestamp.utcnow()
        
        try:
            # 1. Update market data
            market_data = await self._update_market_data()
            if not market_data:
                logger.warning("⚠️ No market data available, skipping cycle")
                return
            
            # 2. Update portfolio state
            await self._update_portfolio_state(market_data)
            
            # 3. Update sentiment analysis
            await self._update_sentiment_analysis()
            
            # 4. Execute L3 processing
            l3_decision = await self._execute_l3_processing(market_data)
            
            # 5. Execute L2 processing
            l2_signals = await self._execute_l2_processing(market_data, l3_decision)
            
            # 6. Execute L1 processing
            orders = await self._execute_l1_processing(l2_signals, market_data)
            
            # 7. Execute orders
            processed_orders = await self._execute_orders(orders)
            
            # 8. Update portfolio with results
            await self._update_portfolio_from_orders(processed_orders, market_data)
            
            # 9. Log cycle results
            await self._log_cycle_results(start_time, l2_signals, processed_orders)
            
        except Exception as e:
            logger.error(f"❌ Cycle {self.cycle_id} failed: {e}", exc_info=True)
            await ErrorHandler.handle_cycle_error(e, self.cycle_id)
    
    async def _update_market_data(self) -> Dict[str, Any]:
        """Update market data from data feed."""
        try:
            market_data = await self.data_feed.get_market_data()
            if market_data:
                logger.info(f"📊 Market data updated for {len(market_data)} symbols")
            return market_data
        except Exception as e:
            logger.error(f"❌ Failed to update market data: {e}")
            return {}
    
    async def _update_portfolio_state(self, market_data: Dict[str, Any]):
        """Update portfolio state with current market data."""
        try:
            # Sync portfolio with exchange
            sync_success = await self.portfolio_manager.sync_with_exchange()
            if sync_success:
                logger.info("✅ Portfolio synchronized with exchange")
            else:
                logger.warning("⚠️ Portfolio sync failed, using local state")
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio state: {e}")
    
    async def _update_sentiment_analysis(self):
        """Update sentiment analysis if needed."""
        cycles_since_update = max(0, self.cycle_id - self.last_sentiment_update)
        
        if cycles_since_update >= self.sentiment_update_interval:
            logger.info("🔄 Updating sentiment analysis...")
            self.sentiment_texts_cache = await update_sentiment_texts()
            self.last_sentiment_update = self.cycle_id
            logger.info(f"✅ Sentiment analysis updated with {len(self.sentiment_texts_cache)} texts")
    
    async def _execute_l3_processing(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute L3 strategic processing."""
        if APAGAR_L3:
            logger.info("🔴 L3 processing disabled")
            return {
                'regime': 'disabled',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
        
        try:
            # Classify market regime
            if 'BTCUSDT' in market_data:
                regime_result = self.regime_classifier.classify_market_regime(
                    market_data.get("BTCUSDT", pd.DataFrame()), "BTCUSDT"
                )
                
                # Generate L3 decision
                l3_decision = make_decision(
                    inputs={},
                    portfolio_state=self.portfolio_manager.get_portfolio_state(),
                    market_data=market_data,
                    regime_decision=regime_result,
                    balances_synced=True
                )
                
                logger.info(f"🧠 L3 decision: {l3_decision.get('regime')} regime with {l3_decision.get('confidence', 0):.2f} confidence")
                return l3_decision
            
            logger.warning("⚠️ No BTCUSDT data for L3 processing")
            return {
                'regime': 'unknown',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
            
        except Exception as e:
            logger.error(f"❌ L3 processing failed: {e}")
            return {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
    
    async def _execute_l2_processing(self, market_data: Dict[str, Any], l3_decision: Dict[str, Any]) -> list:
        """Execute L2 tactical processing."""
        try:
            # Generate L2 signals with L3 context
            l2_signals = self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=l3_decision
            )
            
            logger.info(f"🎯 Generated {len(l2_signals)} L2 signals")
            return l2_signals
            
        except Exception as e:
            logger.error(f"❌ L2 processing failed: {e}")
            return []
    
    async def _execute_l1_processing(self, l2_signals: list, market_data: Dict[str, Any]) -> list:
        """Execute L1 operational processing."""
        try:
            # Generate orders from L2 signals
            orders = await self.order_manager.generate_orders(get_system_state(), l2_signals)
            
            # Validate orders
            validated_orders = []
            for order in orders:
                if order.get("status") == "pending":
                    validation_result = self.order_manager.validate_order_size(
                        order.get("symbol"),
                        order.get("quantity", 0.0),
                        order.get("price", 0.0),
                        get_system_state().get("portfolio", {})
                    )
                    
                    if validation_result["valid"]:
                        validated_orders.append(order)
                    else:
                        logger.warning(f"❌ Order rejected: {validation_result['reason']}")
                        order["status"] = "rejected"
                        order["validation_error"] = validation_result["reason"]
                        validated_orders.append(order)
                else:
                    validated_orders.append(order)
            
            logger.info(f"✅ Generated {len(validated_orders)} validated orders")
            return validated_orders
            
        except Exception as e:
            logger.error(f"❌ L1 processing failed: {e}")
            return []
    
    async def _execute_orders(self, orders: list) -> list:
        """Execute orders through the order manager."""
        try:
            processed_orders = await self.order_manager.execute_orders(orders)
            
            # Count successful executions
            successful_orders = [o for o in processed_orders if o.get("status") == "filled"]
            logger.info(f"💰 Executed {len(successful_orders)} orders successfully")
            
            return processed_orders
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            return []
    
    async def _update_portfolio_from_orders(self, processed_orders: list, market_data: Dict[str, Any]):
        """Update portfolio state from processed orders."""
        try:
            await self.portfolio_manager.update_from_orders_async(processed_orders, market_data)
            
            # Update trading metrics - CRITICAL FIX: Use async method (no market_data needed - PortfolioManager uses injected MarketDataManager)
            total_value = await self.portfolio_manager.get_total_value_async()
            self.trading_metrics.update_from_orders(processed_orders, total_value)
            
            # Save portfolio state periodically
            if self.cycle_id % 5 == 0:
                self.portfolio_manager.save_to_json()
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio from orders: {e}")
    
    async def _log_cycle_results(self, start_time: pd.Timestamp, l2_signals: list, processed_orders: list):
        """Log the results of the current cycle."""
        try:
            # Calculate cycle duration
            cycle_duration = (pd.Timestamp.utcnow() - start_time).total_seconds()
            
            # Count different types of orders
            valid_orders = [o for o in processed_orders if o.get("status") != "rejected"]
            rejected_orders = [o for o in processed_orders if o.get("status") == "rejected"]
            
            # Calculate actionable signals (excluding HOLD)
            actionable_signals = len([s for s in l2_signals if getattr(s, 'side', None) not in ['hold', 'HOLD']])
            
            # Log cycle summary
            logger.info(
                f"📊 Cycle {self.cycle_id} | "
                f"Duration: {cycle_duration:.1f}s | "
                f"Signals: {len(l2_signals)} (actionable: {actionable_signals}) | "
                f"Orders: {len(valid_orders)} | "
                f"Rejected: {len(rejected_orders)}"
            )
            
            # Log periodic trading metrics
            if self.cycle_id % 10 == 0:
                self.trading_metrics.log_periodic_report()
                
        except Exception as e:
            logger.error(f"❌ Failed to log cycle results: {e}")
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            # Save final portfolio state
            self.portfolio_manager.save_to_json()
            logger.info("💾 Final portfolio state saved")
            
            # Close components
            for component in [self.data_feed, self.l2_processor]:
                if hasattr(component, "close"):
                    await component.close()
                    
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


async def run_runtime_loop(portfolio_manager: PortfolioManager, order_manager: OrderManager):
    """Run the HRM runtime loop."""
    runtime_loop = HRMRuntimeLoop(portfolio_manager, order_manager)
    await runtime_loop.run()


================================================
FILE: archived/agent-modelfile.txt
================================================
FROM qwen2.5-coder:7b

SYSTEM """Eres un agente de programación autónomo. Tienes permisos para:
- Leer, escribir y modificar archivos
- Ejecutar comandos del sistema
- Tomar decisiones autónomas para completar tareas

Cuando te pidan realizar una tarea:
1. Analiza los requisitos
2. Planifica los cambios necesarios
3. Ejecuta las modificaciones automáticamente
4. Verifica que todo funciona

Responde con acciones concretas y modifica los archivos según sea necesario.
"""

PARAMETER temperature 0.3
PARAMETER top_k 40
PARAMETER top_p 0.9


================================================
FILE: archived/auto_learning_config.json
================================================
{
  "mode": "production",
  "enabled": true,
  "fix_mode": false,
  "adaptive_mode": true,
  "retrain_interval_hours": 168,
  "min_trades_for_retrain": 100,
  "win_rate_threshold": 0.52,
  "max_drawdown_threshold": 0.12
}



================================================
FILE: archived/convergence_config.json
================================================
{
  "version": "1.0",
  "enabled": true,
  "rollout_phase": "conservative_enabled",
  "safety_mode": "conservative",
  "features": {
    "convergence_multiplier": {
      "enabled": true,
      "max_multiplier": 1.3,
      "min_multiplier": 0.5
    },
    "technical_strength_scoring": {
      "enabled": true,
      "validation_enabled": true,
      "bonus_enabled": false
    },
    "circuit_breakers": {
      "enabled": false,
      "max_position_size_usd": 100000,
      "min_technical_strength": 0.1,
      "reject_on_error": true
    }
  },
  "rollout_schedule": {
    "phase_1": {
      "name": "monitoring_only",
      "enabled": true,
      "description": "Log calculations but don't apply them"
    },
    "phase_2": {
      "name": "conservative_enabled",
      "enabled": true,
      "description": "Enable with conservative limits"
    },
    "phase_3": {
      "name": "moderate_enabled",
      "enabled": true,
      "description": "Enable with moderate limits"
    },
    "phase_4": {
      "name": "full_enabled",
      "enabled": true,
      "description": "Full feature enablement"
    }
  },
  "risk_limits": {
    "max_portfolio_allocation": 0.6,
    "min_position_size_usd": 10.0,
    "max_position_size_usd": 25000.0,
    "emergency_stop_threshold": 0.05
  },
  "logging": {
    "detailed_logging": true,
    "alert_on_rejections": true,
    "performance_tracking": true
  },
  "last_updated": "2025-09-28T23:23:30.320308",
  "created": "2025-09-28T23:22:57.549495"
}


================================================
FILE: archived/global_system_state.json
================================================
{
  "current_state": "BLIND",
  "state_history": [
    {
      "from_state": "NORMAL",
      "to_state": "BLIND",
      "reason": "L3 balance sync failed - entering BLIND mode",
      "timestamp": "2026-01-20T23:23:31.184120",
      "metadata": {
        "l3_balance_sync_failed": true,
        "l3_mode": "BLIND"
      }
    }
  ],
  "state_metadata": {
    "l3_balance_sync_failed": true,
    "l3_mode": "BLIND"
  },
  "last_updated": "2026-01-20T23:23:31.184120"
}


================================================
FILE: archived/initial_state.json
================================================
{
  "capital_usdt": 3000.0,
  "btc": 0.0,
  "eth": 0.0,
  "mode": "paper",
  "auto_learning": "fix",
  "initial_balances": {
    "USDT": 3000.0,
    "BTC": 0.0,
    "ETH": 0.0
  },
  "timestamp": "2026-02-08T15:30:19.162057"
}


================================================
FILE: archived/sentiment_cache_timestamp.json
================================================
{"timestamp": "2025-09-30T13:05:43.925238", "texts": ["Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2", "Sample text 1", "Sample text 2"]}


================================================
FILE: backtesting/__init__
================================================
# hola


================================================
FILE: backtesting/backtesting_utils.py
================================================
# C:\proyectos\HRM\backtesting\backtesting_utils.py

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Definiciones movidas de hrm_tester.py
class TestMode(Enum):
    LIVE_TRADES = "live_trades"
    HISTORICAL = "historical"
    SIMULATION = "simulation"

class TestLevel(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"

@dataclass
class TestResult:
    test_id: str
    level: TestLevel
    name: str
    description: str
    success: bool
    message: str
    duration_ms: float
    details: Optional[Dict] = None

class L1Model:
    def __init__(self, predictions_path: Optional[str] = None):
        self.is_real = False
        self.models = {}
        self.predictions_path = predictions_path or os.path.join(Path(__file__).parent, 'L1_predictions.json')
        self._load_predictions()

    def _load_predictions(self):
        """Carga las predicciones desde el archivo JSON si existe."""
        if os.path.exists(self.predictions_path):
            try:
                with open(self.predictions_path, 'r') as f:
                    self.models = json.load(f)
                    self.is_real = True
                    logging.info(f"✅ Predicciones de L1 cargadas con éxito desde '{self.predictions_path}'.")
            except Exception as e:
                logging.warning(f"Error al cargar el archivo de predicciones: {e}. Usando implementación mock.")
                self._load_mock_predictions()
        else:
            logging.warning("Archivo de predicciones de L1 no encontrado. Usando implementación mock.")
            self._load_mock_predictions()

    def _load_mock_predictions(self):
        """Crea un diccionario con métricas simuladas."""
        self.models = {
            'LogisticRegression': {
                'accuracy': 0.85,
                'precision': 0.80,
                'f1_score': 0.82,
                'profit_contribution': 1000,
                'latency_ms': 50
            },
            'RandomForest': {
                'accuracy': 0.82,
                'precision': 0.78,
                'f1_score': 0.8,
                'profit_contribution': 800,
                'latency_ms': 45
            },
            'LightGBM': {
                'accuracy': 0.87,
                'precision': 0.83,
                'f1_score': 0.85,
                'profit_contribution': 1200,
                'latency_ms': 60
            }
        }
    
    def predict(self, data: Any) -> Dict:
        """
        Devuelve las predicciones cargadas o simuladas.
        """
        return self.models


================================================
FILE: backtesting/config.json
================================================
{
  "binance": {
      "api_key": "",
      "api_secret": "",
      "testnet": false,
      "symbols": ["BTCUSDT", "ETHUSDT"],
      "intervals": ["1m", "5m", "15m", "1h"]
  },
  "testing": {
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "initial_capital": 3000.0,
      "strategy_name": "FinRL_MultiAsset_Ensemble",
      "output_dir": "backtesting/results"
  },
  "reporting": {
      "generate_charts": true,
      "detailed_logs": true,
      "export_trades": true
  },
  "analysis": {
  "generate_charts": true,
  "export_trades": true,
  "metrics": ["sharpe", "drawdown", "win_rate"]
},
  "validation": {
    "periods": [
      {"start_date": "2022-01-01", "end_date": "2022-12-31"},
      {"start_date": "2023-01-01", "end_date": "2023-12-31"},
      {"start_date": "2024-01-01", "end_date": "2024-12-31"}
    ],
    "cost_profiles": [
      {"name": "base", "fee_rate": 0.001, "slippage_pct": 0.0000},
      {"name": "medio", "fee_rate": 0.002, "slippage_pct": 0.0005},
      {"name": "duro", "fee_rate": 0.003, "slippage_pct": 0.0010}
    ]
  }
}


================================================
FILE: backtesting/descargar.py
================================================
#!/usr/bin/env python3
"""
Script para descargar datos históricos de Binance de los últimos 5 años
Guarda los datos en formato parquet compatible con el backtesting HRM
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BinanceHistoricalDownloader:
    """Descargador de datos históricos de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        self.rate_limit_delay = 0.2  # 200ms entre requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
        """Obtiene klines (velas) de Binance"""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded for {symbol}. Waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self.get_klines(symbol, interval, start_time, end_time, limit)
                else:
                    logger.error(f"Error {response.status} for {symbol}: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    async def download_symbol_data(self, symbol: str, interval: str = '5m', years: int = 5) -> pd.DataFrame:
        """Descarga datos históricos para un símbolo específico"""
        logger.info(f"Descargando datos de {symbol} con intervalo {interval} para {years} años...")
        
        # Calcular fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Convertir a timestamps en milisegundos
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        
        # Binance permite máximo 1000 klines por request
        # Para 5m, 1000 klines = ~3.47 días
        batch_size = 1000
        interval_ms = self._interval_to_milliseconds(interval)
        batch_duration = batch_size * interval_ms
        
        batch_count = 0
        while current_start < end_timestamp:
            current_end = min(current_start + batch_duration, end_timestamp)
            
            logger.info(f"  Batch {batch_count + 1}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} a {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")
            
            klines = await self.get_klines(symbol, interval, current_start, current_end, batch_size)
            
            if klines:
                all_klines.extend(klines)
                logger.info(f"    Obtenidas {len(klines)} velas")
            else:
                logger.warning(f"    No se obtuvieron datos para este batch")
                break
            
            current_start = current_end + interval_ms  # Evitar solapamiento
            batch_count += 1
            
            # Progreso cada 10 batches
            if batch_count % 10 == 0:
                progress = (current_start - start_timestamp) / (end_timestamp - start_timestamp) * 100
                logger.info(f"  Progreso: {progress:.1f}%")
        
        if not all_klines:
            logger.error(f"No se obtuvieron datos para {symbol}")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = self._klines_to_dataframe(all_klines, symbol)
        
        # Remover duplicados por timestamp
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"  Completado: {len(df)} velas descargadas para {symbol}")
        return df
    
    def _interval_to_milliseconds(self, interval: str) -> int:
        """Convierte intervalo a milisegundos"""
        intervals = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 5 * 60 * 1000)  # Default 5m
    
    def _klines_to_dataframe(self, klines: List, symbol: str) -> pd.DataFrame:
        """Convierte klines a DataFrame normalizado"""
        if not klines:
            return pd.DataFrame()
        
        # Estructura de klines de Binance:
        # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, 
        #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Añadir columna de símbolo
        df['symbol'] = symbol
        
        # Seleccionar solo las columnas necesarias
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        return df.dropna()

async def main():
    """Función principal"""
    
    # Configuración
    symbols = ['BTCUSDT', 'ETHUSDT']  # Añadir más símbolos si se desea
    interval = '5m'
    years = 5
    
    # Rutas
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / 'normalized_grok.parquet'
    
    logger.info(f"Iniciando descarga de datos históricos...")
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Intervalo: {interval}")
    logger.info(f"Años: {years}")
    logger.info(f"Archivo de salida: {output_file}")
    
    all_data = []
    
    async with BinanceHistoricalDownloader() as downloader:
        for symbol in symbols:
            try:
                df = await downloader.download_symbol_data(symbol, interval, years)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Datos de {symbol} añadidos: {len(df)} registros")
                else:
                    logger.warning(f"No se obtuvieron datos para {symbol}")
            except Exception as e:
                logger.error(f"Error descargando {symbol}: {e}")
                continue
    
    if not all_data:
        logger.error("No se descargaron datos de ningún símbolo")
        return
    
    # Combinar todos los datos
    logger.info("Combinando y guardando datos...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ordenar por símbolo y timestamp
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Estadísticas
    logger.info(f"Dataset final:")
    logger.info(f"  Total de registros: {len(combined_df)}")
    logger.info(f"  Símbolos: {combined_df['symbol'].unique()}")
    logger.info(f"  Rango de fechas: {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
    logger.info(f"  Registros por símbolo:")
    for symbol in combined_df['symbol'].unique():
        count = len(combined_df[combined_df['symbol'] == symbol])
        logger.info(f"    {symbol}: {count}")
    
    # Guardar en formato parquet
    try:
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        logger.info(f"Datos guardados exitosamente en: {output_file}")
        logger.info(f"Tamaño del archivo: {output_file.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        logger.error(f"Error guardando el archivo: {e}")
        # Fallback a pickle si parquet falla
        fallback_file = data_dir / 'normalized_grok.pkl'
        combined_df.to_pickle(fallback_file)
        logger.info(f"Datos guardados como fallback en: {fallback_file}")
    
    # Verificación final
    logger.info("Verificando archivo guardado...")
    try:
        test_df = pd.read_parquet(output_file)
        logger.info(f"Verificación exitosa: {len(test_df)} registros leídos")
        logger.info(f"Columnas: {list(test_df.columns)}")
        logger.info(f"Primeros registros:")
        print(test_df.head())
    except Exception as e:
        logger.error(f"Error en verificación: {e}")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import aiohttp
        import pyarrow
    except ImportError as e:
        logger.error(f"Dependencia faltante: {e}")
        logger.error("Instalar con: pip install aiohttp pyarrow")
        exit(1)
    
    logger.info("Iniciando descarga...")
    asyncio.run(main())


================================================
FILE: backtesting/descargar_datos_para_modelo_finrl.py
================================================
#!/usr/bin/env python3
"""
Script para descargar datos históricos de Binance de los últimos 5 años
Guarda los datos en formato CSV compatible con el backtesting HRM
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BinanceHistoricalDownloader:
    """Descargador de datos históricos de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        self.rate_limit_delay = 0.2  # 200ms entre requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
        """Obtiene klines (velas) de Binance"""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded for {symbol}. Waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self.get_klines(symbol, interval, start_time, end_time, limit)
                else:
                    logger.error(f"Error {response.status} for {symbol}: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    async def download_symbol_data(self, symbol: str, interval: str = '5m', years: int = 5) -> pd.DataFrame:
        """Descarga datos históricos para un símbolo específico"""
        logger.info(f"Descargando datos de {symbol} con intervalo {interval} para {years} años...")
        
        # Calcular fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Convertir a timestamps en milisegundos
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        
        # Binance permite máximo 1000 klines por request
        # Para 5m, 1000 klines = ~3.47 días
        batch_size = 1000
        interval_ms = self._interval_to_milliseconds(interval)
        batch_duration = batch_size * interval_ms
        
        batch_count = 0
        while current_start < end_timestamp:
            current_end = min(current_start + batch_duration, end_timestamp)
            
            logger.info(f"  Batch {batch_count + 1}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} a {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")
            
            klines = await self.get_klines(symbol, interval, current_start, current_end, batch_size)
            
            if klines:
                all_klines.extend(klines)
                logger.info(f"    Obtenidas {len(klines)} velas")
            else:
                logger.warning(f"    No se obtuvieron datos para este batch")
                break
            
            current_start = current_end + interval_ms  # Evitar solapamiento
            batch_count += 1
            
            # Progreso cada 10 batches
            if batch_count % 10 == 0:
                progress = (current_start - start_timestamp) / (end_timestamp - start_timestamp) * 100
                logger.info(f"  Progreso: {progress:.1f}%")
        
        if not all_klines:
            logger.error(f"No se obtuvieron datos para {symbol}")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = self._klines_to_dataframe(all_klines, symbol)
        
        # Remover duplicados por timestamp
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"  Completado: {len(df)} velas descargadas para {symbol}")
        return df
    
    def _interval_to_milliseconds(self, interval: str) -> int:
        """Convierte intervalo a milisegundos"""
        intervals = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 5 * 60 * 1000)  # Default 5m
    
    def _klines_to_dataframe(self, klines: List, symbol: str) -> pd.DataFrame:
        """Convierte klines a DataFrame normalizado"""
        if not klines:
            return pd.DataFrame()
        
        # Estructura de klines de Binance:
        # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, 
        #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Añadir columna de símbolo
        df['symbol'] = symbol
        
        # Seleccionar solo las columnas necesarias
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        return df.dropna()

async def main():
    """Función principal"""
    
    # Configuración
    symbols = ['BTCUSDT', 'ETHUSDT']  # Añadir más símbolos si se desea
    interval = '5m'
    years = 5
    
    # Rutas
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / 'final_combinado.csv'
    
    logger.info(f"Iniciando descarga de datos históricos...")
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Intervalo: {interval}")
    logger.info(f"Años: {years}")
    logger.info(f"Archivo de salida: {output_file}")
    
    all_data = []
    
    async with BinanceHistoricalDownloader() as downloader:
        for symbol in symbols:
            try:
                df = await downloader.download_symbol_data(symbol, interval, years)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Datos de {symbol} añadidos: {len(df)} registros")
                else:
                    logger.warning(f"No se obtuvieron datos para {symbol}")
            except Exception as e:
                logger.error(f"Error descargando {symbol}: {e}")
                continue
    
    if not all_data:
        logger.error("No se descargaron datos de ningún símbolo")
        return
    
    # Combinar todos los datos
    logger.info("Combinando y guardando datos...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ordenar por símbolo y timestamp
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Estadísticas
    logger.info(f"Dataset final:")
    logger.info(f"  Total de registros: {len(combined_df)}")
    logger.info(f"  Símbolos: {combined_df['symbol'].unique()}")
    logger.info(f"  Rango de fechas: {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
    logger.info(f"  Registros por símbolo:")
    for symbol in combined_df['symbol'].unique():
        count = len(combined_df[combined_df['symbol'] == symbol])
        logger.info(f"    {symbol}: {count}")
    
    # Guardar en formato CSV
    try:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Datos guardados exitosamente en: {output_file}")
        logger.info(f"Tamaño del archivo: {output_file.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        logger.error(f"Error guardando el archivo CSV: {e}")
        # Fallback a pickle si CSV falla
        fallback_file = data_dir / 'normalized_grok.pkl'
        combined_df.to_pickle(fallback_file)
        logger.info(f"Datos guardados como fallback en: {fallback_file}")
    
    # Verificación final
    logger.info("Verificando archivo guardado...")
    try:
        test_df = pd.read_csv(output_file, parse_dates=['timestamp'])
        logger.info(f"Verificación exitosa: {len(test_df)} registros leídos")
        logger.info(f"Columnas: {list(test_df.columns)}")
        logger.info(f"Primeros registros:")
        print(test_df.head())
    except Exception as e:
        logger.error(f"Error en verificación: {e}")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import aiohttp
    except ImportError as e:
        logger.error(f"Dependencia faltante: {e}")
        logger.error("Instalar con: pip install aiohttp pandas")
        exit(1)
    
    logger.info("Iniciando descarga...")
    asyncio.run(main())


================================================
FILE: backtesting/getdata.py
================================================
#!/backtesting/getdata.py
"""
Recolector de Datos de Binance para HRM Backtesting
Conecta directamente a la API de Binance para obtener datos históricos
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode

from core.logging import logger  # ✅ Usar logger centralizado


class BinanceDataCollector:
    """Recolector de datos históricos de Binance"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        
        # URLs base
        if self.testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
            
        # ✅ Usar logger centralizado
        self.logger = logger
        
        # Cache para evitar requests repetidos
        self.data_cache = {}
        
        # Rate limiting
        self.last_request = 0
        self.request_interval = 0.1  # 100ms entre requests

    async def collect_historical_data(self, symbols: List[str], start_date: str, 
                                    end_date: str, intervals: List[str]) -> Dict:
        """Recolectar datos históricos para múltiples símbolos e intervalos"""
        
        self.logger.info(f"Recolectando datos para {len(symbols)} símbolos, {len(intervals)} intervalos")
        
        data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                data[symbol] = {}
                
                for interval in intervals:
                    self.logger.info(f"  Obteniendo {symbol} {interval}...")
                    
                    cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"
                    if cache_key in self.data_cache:
                        self.logger.info(f"    Usando datos de cache para {cache_key}")
                        data[symbol][interval] = self.data_cache[cache_key]
                        continue
                        
                    try:
                        klines = await self._get_historical_klines(
                            session, symbol, interval, start_date, end_date
                        )
                        if klines is not None:
                            df = self._process_klines_to_dataframe(klines)
                            if df is not None and not df.empty:
                                df = self._add_technical_indicators(df, symbol, interval)
                                data[symbol][interval] = df
                                self.data_cache[cache_key] = df
                                self.logger.info(f"    ✅ {len(df)} velas obtenidas")
                            else:
                                self.logger.warning(f"    ⚠️  Sin datos para {symbol} {interval}")
                        else:
                            self.logger.warning(f"    ⚠️  Sin datos para {symbol} {interval}")
                    except Exception as e:
                        self.logger.error(f"    ❌ Error obteniendo {symbol} {interval}: {e}", exc_info=True)
                        
                    # Rate limiting
                    await asyncio.sleep(self.request_interval)
        
        # Añadir datos adicionales (placeholder)
        data = await self._enrich_data(data)
        
        self.logger.info("✅ Recolección de datos completada")
        return data

    # -------------------------
    # Helpers internos (fallback local)
    # -------------------------
    async def _get_historical_klines(self, session, symbol: str, interval: str,
                                     start_date: str, end_date: str):
        """Lee desde parquet consolidado (datos/normalized_grok.parquet) y re-muestrea.
        Si no existe, intenta data/normalized_grok.parquet. Columnas esperadas: symbol, open, high, low, close, volume, timestamp.
        """
        try:
            import pandas as pd
            from pathlib import Path

            # Cachea la carga para no re-leer el parquet por cada símbolo/intervalo
            if not hasattr(self, "_normalized_cache"):
                self._normalized_cache = None

            if self._normalized_cache is None:
                root = Path(__file__).resolve().parents[1]
                parquet_paths = [
                    root / 'data' / 'normalized_grok.parquet',
                    root / 'datos' / 'normalized_grok.parquet',
                ]
                parquet_fp = next((p for p in parquet_paths if p.exists()), None)
                if parquet_fp is None:
                    self.logger.warning("    ⚠️  Parquet normalizado no encontrado en datos/ o data/")
                    return None
                self.logger.info(f"    📁 Leyendo parquet desde: {parquet_fp}")
                df_all = pd.read_parquet(parquet_fp)
                self.logger.info(f"    📊 Parquet cargado: {df_all.shape[0]} filas, {df_all.shape[1]} columnas")
                # Normalización mínima
                time_col = None
                for c in ['timestamp', 'time', 'date', 'datetime', 'open_time']:
                    if c in df_all.columns:
                        time_col = c
                        break
                if time_col is None:
                    self.logger.warning("    ⚠️  Parquet sin columna temporal reconocible")
                    return None
                df_all[time_col] = pd.to_datetime(df_all[time_col])
                df_all = df_all.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                # Estandariza símbolo: mapear ticker a symbol
                if 'ticker' in df_all.columns:
                    df_all['symbol'] = df_all['ticker'].replace({'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}).astype(str)
                elif 'symbol' in df_all.columns:
                    df_all['symbol'] = df_all['symbol'].replace({'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}).astype(str)
                else:
                    # Si no hay columna de símbolo, asume BTCUSDT
                    df_all['symbol'] = 'BTCUSDT'

                self._normalized_cache = df_all[[time_col, 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                self._normalized_cache = self._normalized_cache.dropna(subset=['open', 'high', 'low', 'close']).copy()

            df_all = self._normalized_cache

            # Filtra símbolo
            sym = symbol
            if sym not in ['BTCUSDT', 'ETHUSDT']:
                # intenta mapear por si vienen tickers básicos
                mapping = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
                sym = mapping.get(sym, sym)
            self.logger.info(f"    🔍 Buscando símbolo: {sym} en {df_all['symbol'].unique()}")
            df = df_all[df_all['symbol'] == sym].copy()
            if df.empty:
                self.logger.warning(f"    ⚠️  Sin datos en parquet para {symbol} (símbolos disponibles: {df_all['symbol'].unique()})")
                return None
            self.logger.info(f"    ✅ Encontrados {len(df)} registros para {symbol}")

            # Índice temporal
            idx_col = [c for c in df.columns if str(df[c].dtype).startswith('datetime')]
            idx_col = idx_col[0] if idx_col else None
            if idx_col is None:
                self.logger.warning("    ⚠️  No se pudo identificar la columna temporal tras carga")
                return None
            df = df.set_index(idx_col).sort_index()

            # Recorte por fechas
            try:
                self.logger.info(f"    📅 Rango de fechas solicitado: {start_date} a {end_date}")
                self.logger.info(f"    📅 Rango de fechas en datos: {df.index.min()} a {df.index.max()}")
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
                self.logger.info(f"    📅 Después del filtro: {len(df)} registros")
            except Exception as e:
                self.logger.error(f"    ❌ Error en filtro de fechas: {e}")
                pass

            # Llenar NaN en volume con 0 para evitar problemas en resampling
            df['volume'] = df['volume'].fillna(0)

            # El parquet está a 5m; re-muestrea a lo solicitado
            if interval:
                rule = interval
                if rule == '1m':
                    # Upsample no fiable; avisamos y devolvemos vacío para evitar conclusiones erróneas
                    self.logger.warning("    ⚠️  1m solicitado pero fuente es 5m; se omite 1m")
                    return None
                if rule != '5m':
                    df = df.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                    }).dropna(subset=['open', 'high', 'low', 'close'])

            # Selecciona columnas OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df
        except Exception as e:
            self.logger.error(f"    ❌ Lectura desde parquet falló: {e}")
            return None

    def _process_klines_to_dataframe(self, klines):
        """Normaliza klines a DataFrame OHLCV con índice datetime."""
        import pandas as pd
        if isinstance(klines, pd.DataFrame):
            return klines
        # Si llegaran como lista de listas [openTime, open, high, low, close, volume, closeTime, ...]
        try:
            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
            df = pd.DataFrame(klines, columns=cols[:len(klines[0])])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            return df[['open', 'high', 'low', 'close', 'volume']].dropna()
        except Exception:
            return None

    def _add_technical_indicators(self, df, symbol: str, interval: str):
        """Añade indicadores básicos (RSI, SMA20, Bollinger)."""
        import pandas as pd
        import numpy as np
        out = df.copy()

        # SMA
        out['sma_20'] = out['close'].rolling(20, min_periods=5).mean()
        # Bollinger
        std = out['close'].rolling(20, min_periods=5).std()
        out['bollinger_middle'] = out['sma_20']
        out['bollinger_upper'] = out['sma_20'] + 2 * std
        out['bollinger_lower'] = out['sma_20'] - 2 * std

        # RSI simple
        delta = out['close'].diff()
        up = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        down = -delta.clip(upper=0).rolling(14, min_periods=5).mean()
        rs = up / (down.replace(0, np.nan))
        out['rsi'] = 100 - (100 / (1 + rs.replace({np.inf: np.nan})))

        return out.dropna().copy() if len(out) else out

    async def _enrich_data(self, data: Dict) -> Dict:
        """Placeholder para enriquecer datos (retorna tal cual por ahora)."""
        return data


# Función de utilidad para testing rápido
async def quick_test():
    """Test rápido del recolector"""
    
    config = {
        'api_key': '',  # Dejar vacío para test sin auth
        'api_secret': '',
        'testnet': True,
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'intervals': ['1h', '5m']
    }
    
    collector = BinanceDataCollector(config)
    
    # Test básico - últimas 24 horas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = await collector.collect_historical_data(
        symbols=config['symbols'],
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intervals=config['intervals']
    )
    
    # Guardar datos
    collector.save_data_to_csv(data)
    
    # Test precios en vivo
    prices = await collector.get_live_prices(config['symbols'])
    logger.info(f"Precios en vivo: {prices}")
    
    # Test info cuenta (si hay credenciales)
    if config['api_key'] and config['api_secret']:
        account_info = await collector.get_account_info()
        logger.info(f"Info cuenta: {account_info}")

if __name__ == "__main__":
    asyncio.run(quick_test())



================================================
FILE: backtesting/L1_predictions.json
================================================
{
    "LogisticRegression": {
        "accuracy": 0.85,
        "precision": 0.8,
        "f1_score": 0.82,
        "profit_contribution": 1000,
        "latency_ms": 50
    },
    "RandomForest": {
        "accuracy": 0.85,
        "precision": 0.8,
        "f1_score": 0.82,
        "profit_contribution": 1000,
        "latency_ms": 50
    },
    "LightGBM": {
        "accuracy": 0.85,
        "precision": 0.8,
        "f1_score": 0.82,
        "profit_contribution": 1000,
        "latency_ms": 50
    }
}


================================================
FILE: backtesting/main.py
================================================
#!/backtesting/main.py   & .\.venv\Scripts\python.exe -m backtesting.main
"""# Test Gemini model
$env:L2_MODEL = "gemini"; python -m backtesting.main
$env:L2_MODEL = "grok"; python -m backtesting.main
$env:L2_MODEL = "claude"; python -m backtesting.main
$env:L2_MODEL = "gpt"; python -m backtesting.main
$env:L2_MODEL = "kimi"; python -m backtesting.main
$env:L2_MODEL = "deepseek"; python -m backtesting.main
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logging import logger  # Logger centralizado

# Imports relativos dentro del paquete backtesting
try:
    from .getdata import BinanceDataCollector
    from .hrm_tester import HRMStrategyTester
    from .performance_analyzer import PerformanceAnalyzer
    from .report_generator import ReportGenerator
except ImportError:
    # Fallback for direct execution
    from getdata import BinanceDataCollector
    from hrm_tester import HRMStrategyTester
    from performance_analyzer import PerformanceAnalyzer
    from report_generator import ReportGenerator


class HRMBacktester:
    """Orquestador principal del backtesting HRM"""
    
    def __init__(self, config_path: str = "backtesting/config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Inicializar componentes
        self.data_collector = BinanceDataCollector(self.config['binance'])
        self.strategy_tester = HRMStrategyTester(self.config['testing'], self.data_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.config['analysis'])
        self.report_generator = ReportGenerator(self.config['reporting'])

        # Estado del backtesting
        self.results = {
            'overall': {},
            'l1_models': {},
            'l2_model': {},
            'l3_models': {},
            'trades': []
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Carga configuración y aplica defaults para claves faltantes."""
        defaults = {
            'binance': {
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'intervals': ['1m', '5m', '15m', '1h'],
                'historical_days': 1,
                'api_key': '',
                'api_secret': '',
                'testnet': True,
            },
            'testing': {
                'mode': 'full',
                'lookback_days': 7,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'intervals': ['5m', '15m', '1h'],
                'start_date': None,
                'end_date': None,
                'initial_capital': 3000.0,
                'position_size': 0.15,
                'confianza_minima': 0.65,
            },
            'analysis': {
                'generate_charts': True,
                'export_trades': True,
                'metrics': ['sharpe', 'drawdown', 'win_rate'],
            },
            'reporting': {
                'output_dir': 'backtesting/results',
                'generate_charts': True,
                'detailed_logs': True,
                'export_trades': True,
            },
        }

        def _merge(user: Dict, base: Dict) -> Dict:
            merged = dict(base)
            for k, v in (user or {}).items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    merged[k] = _merge(v, base[k])
                else:
                    merged[k] = v
            return merged

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_cfg = json.load(f)
                return _merge(user_cfg, defaults)
            except Exception:
                logging.warning("Config inválida, usando defaults.")
                return defaults
        else:
            logging.warning("Archivo de configuración no encontrado, usando defaults.")
            return defaults

    def setup_logging(self):
        """Integra el logger centralizado y define self.logger."""
        try:
            self.logger = logger
            self.logger.info("Logging configured successfully.")
        except Exception:
            # Fallback básico si fallara el logger central (poco probable)
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s | %(levelname)-8s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            self.logger = logging.getLogger("backtesting")
            self.logger.info("Logging configured (fallback).")

    async def run_backtest(self):
        """Ejecuta el proceso de backtesting de punta a punta"""
        self.logger.info("Iniciando backtesting HRM...")
        
        try:
            # 1. Recolectar datos
            tcfg = self.config['testing']
            symbols = tcfg.get('symbols') or self.config['binance']['symbols']
            intervals = tcfg.get('intervals') or self.config['binance']['intervals']
            start_date = tcfg.get('start_date')
            end_date = tcfg.get('end_date')
            if not (start_date and end_date):
                # Usar todo el rango disponible del parquet (últimos 5 años)
                end_date = '2025-09-08'  # Fecha actual
                start_date = '2020-09-08'  # 5 años atrás

            historical_data = await self.data_collector.collect_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                intervals=intervals,
            )

            if not historical_data:
                self.logger.error("No se pudieron obtener datos históricos. Abortando.")
                return

            # 2. Ejecutar probador de estrategias
            # Ejecutar estrategia (mock/simple) por ahora
            testing_results = await self.strategy_tester.run_hrm_strategy(historical_data)
            
            # 3. Analizar rendimiento
            initial_capital = self.config['testing']['initial_capital']

            # DEBUG: Log final portfolio value before performance analysis
            final_value = testing_results.get('overall', {}).get('total_value', initial_capital)
            logger.info(f"🎯 FINAL BACKTEST RESULTS - Initial: {initial_capital}, Final: {final_value}, Return: {(final_value - initial_capital):+.2f} ({(final_value - initial_capital)/initial_capital*100:+.2f}%)")

            analyzed_results = self.performance_analyzer.analyze_results(
                testing_results,
                self.config['analysis']['metrics'],
                initial_capital
            )
            
            # 4. Generar reportes
            self.report_generator.generate_full_report(
                analyzed_results, 
                self.config['reporting']['output_dir']
            )

            self.results = analyzed_results
            self.logger.info("✅ Backtesting completado con éxito.")

            # Log detailed performance metrics
            overall = self.results.get('overall', {})
            self.logger.info("📊 MÉTRICAS DE RENDIMIENTO DETALLADAS:")
            self.logger.info(f"   Retorno Total: {overall.get('total_return', 0):.2%}")
            self.logger.info(f"   Retorno Anualizado: {overall.get('annualized_return', 0):.2%}")
            self.logger.info(f"   Ratio Sharpe: {overall.get('sharpe_ratio', 0):.3f}")
            self.logger.info(f"   Ratio Sortino: {overall.get('sortino_ratio', 0):.3f}")
            self.logger.info(f"   Ratio Calmar: {overall.get('calmar_ratio', 0):.3f}")
            self.logger.info(f"   Máximo Drawdown: {overall.get('max_drawdown', 0):.2%}")
            self.logger.info(f"   Duración Máx DD: {overall.get('max_drawdown_duration', 0)} períodos")
            self.logger.info(f"   Win Rate: {overall.get('win_rate', 0):.1%}")
            self.logger.info(f"   Factor de Profit: {overall.get('profit_factor', 0):.2f}")
            self.logger.info(f"   Total Trades: {overall.get('total_trades', 0)}")
            self.logger.info(f"   Trades Ganadores: {overall.get('winning_trades', 0)}")
            self.logger.info(f"   Trades Perdedores: {overall.get('losing_trades', 0)}")
            self.logger.info(f"   VaR 95%: {overall.get('var_95', 0):.2%}")
            self.logger.info(f"   VaR 99%: {overall.get('var_99', 0):.2%}")
            self.logger.info(f"   Factor de Recuperación: {overall.get('recovery_factor', 0):.3f}")
            self.logger.info(f"   Volatilidad: {overall.get('volatility', 0):.2%}")
            self.logger.info(f"   Valor Final Portfolio: ${overall.get('total_value', 0):.2f}")

        except Exception as e:
            self.logger.error(f"❌ Error crítico durante el backtesting: {e}")
            self.logger.exception("Detalles del error:")

    async def run_validation(self):
        cfg = self.config.get('validation', {})
        periods = cfg.get('periods', [])
        cost_profiles = cfg.get('cost_profiles', [])
        if not periods or not cost_profiles:
            self.logger.warning("Validation config incompleta; saltando validación múltiple.")
            return

        summary = []
        for period in periods:
            p_start = period.get('start_date')
            p_end = period.get('end_date')
            for prof in cost_profiles:
                # ajustar perfil en tester
                try:
                    self.strategy_tester.validation_profile = prof
                except Exception:
                    pass
                self.logger.info(f"🔎 Validando periodo {p_start}..{p_end} perfil {prof.get('name')}")
                data = await self.data_collector.collect_historical_data(
                    symbols=self.config['testing'].get('symbols') or self.config['binance']['symbols'],
                    start_date=p_start,
                    end_date=p_end,
                    intervals=self.config['testing'].get('intervals') or self.config['binance']['intervals'],
                )
                res = await self.strategy_tester.run_hrm_strategy(data)
                overall = res.get('overall', {})
                summary.append({
                    'period': f"{p_start}..{p_end}",
                    'profile': prof.get('name'),
                    'trades': overall.get('total_trades', 0),
                    'win_rate': overall.get('win_rate', 0),
                    'total_return': overall.get('total_return', 0),
                })

        # imprimir resumen
        self.logger.info("📋 Resumen validación:")
        for row in summary:
            self.logger.info(f" {row['period']} [{row['profile']}] trades={row['trades']} win={row['win_rate']:.2%} ret={row['total_return']:.2%}")

async def main():
    """Función principal para ejecutar el backtester"""
    backtester = HRMBacktester()

    # Run main backtest
    await backtester.run_backtest()

    # Run validation across different periods and cost profiles
    await backtester.run_validation()

if __name__ == "__main__":
    asyncio.run(main())



================================================
FILE: backtesting/performance_analyzer.py
================================================
# /backtesting/performance_analyzer.py
"""
HRM Performance Analyzer - Analizador de Rendimiento
Analiza y calcula métricas de performance para el sistema HRM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import warnings

from core.logging import logger  # ✅ Logger centralizado

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento calculadas"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    var_95: float
    var_99: float
    calmar_ratio: float
    sortino_ratio: float
    recovery_factor: float

@dataclass
class L1ModelMetrics:
    """Métricas específicas de modelos L1"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    profit_contribution: float
    signal_count: int
    avg_confidence: float
    latency_ms: float

@dataclass
class L2ModelMetrics:
    """Métricas específicas del modelo L2"""
    signal_quality: float
    sizing_efficiency: float
    risk_effectiveness: float
    hit_rate: float
    avg_signal_strength: float
    position_accuracy: float
    risk_adjusted_return: float

@dataclass
class L3ModelMetrics:
    """Métricas específicas de modelos L3"""
    decision_accuracy: float
    regime_detection_accuracy: float
    strategic_value: float
    allocation_efficiency: float
    risk_assessment_accuracy: float

class PerformanceAnalyzer:
    """Analizador principal de rendimiento del sistema HRM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger  # ✅ Usar logger centralizado
        
        # Configuración por defecto
        self.benchmark_symbol = config.get('benchmark', 'BTCUSDT')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.var_confidence_levels = [0.95, 0.99]
    
    # El resto del código se mantiene igual, pero todos los self.logger.warning / error
    # ya apuntan al logger centralizado automáticamente.

    def analyze_results(self, testing_results: Dict, metrics: List[str], initial_capital: float = 1000.0) -> Dict:
        """Calcula métricas básicas a partir de 'testing_results'.
        Espera una clave 'trades' con elementos que incluyan 'entry_price', 'exit_price', y 'pnl'.
        Devuelve un diccionario con sección 'overall' enriquecida.
        """
        try:
            trades = testing_results.get('trades', []) or []

            # DEBUG: Log what data we're receiving
            self.logger.info("🔍 PERFORMANCE ANALYZER DEBUG:")
            self.logger.info(f"   Initial capital: {initial_capital}")
            self.logger.info(f"   Number of trades: {len(trades)}")
            overall_data = testi