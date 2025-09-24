# l1_operational/ai_pipeline.py

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .models import Signal, RiskAlert
import logging

from core.logging import logger

@dataclass
class ModelPrediction:
    """Resultado de predicción de un modelo"""
    model_name: str
    confidence: float
    prediction: float
    features_used: Dict[str, float]
    processing_time_ms: float

@dataclass
class AIDecision:
    """Decisión final del pipeline de IA"""
    should_execute: bool
    confidence: float
    risk_score: float
    model_votes: List[ModelPrediction]
    reasoning: str

class AIModelPipeline:
    """Pipeline jerárquico de modelos de IA para L1"""
    
    def __init__(self, models_path: str = "models/"):
        self.models_path = models_path
        self.models = {}
        self.feature_processors = {}
        self.model_weights = {
            'logistic_regression': 0.3,
            'random_forest': 0.4, 
            'lightgbm': 0.3
        }
        self.load_models()
    

    
    def load_models(self):
        """Carga todos los modelos entrenados"""

        import pickle
        import joblib
        import warnings
        try:
            # Método 1: joblib (preferido)
            return joblib.load(path)
        except Exception as e1:
            try:
                # Método 2: pickle puro
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                try:
                    # Método 3: joblib con protocolo específico
                    return joblib.load(path, allow_pickle=True)
                except Exception as e3:
                    logger.error(f"❌ Error cargando {model_name}: joblib={e1}, pickle={e2}, joblib_allow={e3}")
                    return None
            
            # Cargar metadatos de modelos
            self._load_model_metadata()
            
            logger.info(f"Loaded {len(self.models)} AI models successfully")
            
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            # Modo fallback sin IA
            self.models = {}
    
    def _load_model_metadata(self):
        """Carga metadatos de los modelos (umbrales, features, etc.)"""
        import json
        
        for model_name in self.models.keys():
            try:
                with open(f"{self.models_path}/{model_name}.meta.json", 'r') as f:
                    metadata = json.load(f)
                    self.models[f"{model_name}_meta"] = metadata
            except FileNotFoundError:
                logger.warning(f"No metadata found for {model_name}")
    
    def extract_features(self, signal: Signal, market_data: Dict) -> Dict[str, float]:
        """Extrae features para los modelos desde la señal y datos de mercado"""
        features = {}
        
        # Features básicos de la señal
        features.update({
            'signal_qty': float(signal.qty),
            'signal_side': 1.0 if signal.side == 'buy' else -1.0,
            'has_stop_loss': 1.0 if signal.stop_loss else 0.0,
            'signal_confidence': getattr(signal, 'confidence', 0.5)
        })
        
        # Features de mercado (si disponibles)
        if market_data:
            symbol_data = market_data.get(signal.symbol, {})
            features.update({
                'current_price': symbol_data.get('price', 0.0),
                'volume_24h': symbol_data.get('volume', 0.0),
                'price_change_24h': symbol_data.get('price_change_24h', 0.0),
                'volatility': symbol_data.get('volatility', 0.0)
            })
        
        # Features técnicos adicionales
        if hasattr(signal, 'technical_indicators'):
            features.update(signal.technical_indicators)
        
        return features
    
    def predict_with_model(self, model_name: str, features: Dict[str, float]) -> ModelPrediction:
        """Ejecuta predicción con un modelo específico"""
        import time
        start_time = time.time()

        # L1_{model_name} | Entrada: input features summary
        features_summary = {k: f"{v:.4f}" for k, v in list(features.items())[:5]}
        logger.info(f"L1_{model_name} | Entrada: {features_summary}")

        try:
            model = self.models[model_name]
            metadata = self.models.get(f"{model_name}_meta", {})

            # Preparar features para el modelo
            feature_vector = self._prepare_feature_vector(features, metadata)

            # Predicción
            if hasattr(model, 'predict_proba'):
                # Clasificación
                proba = model.predict_proba([feature_vector])[0]
                confidence = float(np.max(proba))
                prediction = float(np.argmax(proba))
            else:
                # Regresión
                prediction = float(model.predict([feature_vector])[0])
                confidence = min(abs(prediction), 1.0)  # Normalizar confianza

            processing_time = (time.time() - start_time) * 1000

            # L1_{model_name} | Predicción: prediction result
            logger.info(f"L1_{model_name} | Predicción: {prediction:.4f} (confianza: {confidence:.3f})")

            # L1_{model_name} | Señal generada: signal type
            signal_type = "BUY" if prediction > 0.5 else "SELL" if prediction < -0.5 else "HOLD"
            logger.info(f"L1_{model_name} | Señal generada: {signal_type}")

            return ModelPrediction(
                model_name=model_name,
                confidence=confidence,
                prediction=prediction,
                features_used=features,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Error in model {model_name}: {e}")
            return ModelPrediction(
                model_name=model_name,
                confidence=0.0,
                prediction=0.0,
                features_used=features,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _prepare_feature_vector(self, features: Dict[str, float], metadata: Dict) -> np.ndarray:
        """Prepara vector de features según requirements del modelo"""
        expected_features = metadata.get('feature_names', list(features.keys()))
        
        # Ordenar features según el modelo
        feature_vector = []
        for feature_name in expected_features:
            value = features.get(feature_name, 0.0)
            feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def ensemble_decision(self, predictions: List[ModelPrediction]) -> AIDecision:
        """Combina predicciones de múltiples modelos usando weighted voting"""
        if not predictions:
            return AIDecision(
                should_execute=True,  # Fallback: ejecutar si no hay IA
                confidence=0.5,
                risk_score=0.5,
                model_votes=[],
                reasoning="No AI models available, using fallback"
            )
        
        # Weighted voting
        weighted_sum = 0.0
        total_weight = 0.0
        risk_scores = []
        
        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 0.33)
            weighted_sum += pred.prediction * weight * pred.confidence
            total_weight += weight * pred.confidence
            
            # Calcular risk score basado en confianza
            risk_score = 1.0 - pred.confidence
            risk_scores.append(risk_score)
        
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
            confidence = total_weight / len(predictions)
        else:
            final_prediction = 0.5
            confidence = 0.0
        
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0.5
        
        # Decisión final
        should_execute = final_prediction > 0.5 and confidence > 0.6
        
        reasoning = f"Ensemble prediction: {final_prediction:.3f}, confidence: {confidence:.3f}"
        
        return AIDecision(
            should_execute=should_execute,
            confidence=confidence,
            risk_score=avg_risk_score,
            model_votes=predictions,
            reasoning=reasoning
        )
    
    def evaluate_signal(self, signal: Signal, market_data: Dict) -> AIDecision:
        """Método principal: evalúa una señal usando todo el pipeline de IA"""
        if not self.models:
            # Fallback sin IA
            return AIDecision(
                should_execute=True,
                confidence=0.5,
                risk_score=0.5,
                model_votes=[],
                reasoning="AI pipeline not available, using deterministic rules only"
            )
        
        # Extraer features
        features = self.extract_features(signal, market_data)
        
        # Ejecutar todos los modelos
        predictions = []
        for model_name in self.models:
            if not model_name.endswith('_meta'):
                pred = self.predict_with_model(model_name, features)
                predictions.append(pred)
        
        # Combinar resultados
        decision = self.ensemble_decision(predictions)
        
        logger.info(f"AI Decision for signal {signal.signal_id}: "
                   f"execute={decision.should_execute}, "
                   f"confidence={decision.confidence:.3f}")
        
        return decision
