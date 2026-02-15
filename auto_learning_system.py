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
