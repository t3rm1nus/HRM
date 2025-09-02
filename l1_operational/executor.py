# l1_operational/executor.py
"""Ejecutor de órdenes para L1 - Versión corregida"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
from .models import Signal, ExecutionResult, OrderIntent
from .config import EXECUTION_CONFIG, OPERATION_MODE

logger = logging.getLogger(__name__)

class AIModelManager:
    """Gestor de modelos IA para validación de señales"""
    
    def __init__(self, models_path: str = "models/L1/"):
        self.models_path = Path(models_path)
        self.models = {}
        self.feature_names = None
        self.expected_features = 52  # Según README
        self._load_models()
        
    def _load_models(self):
        """Carga todos los modelos disponibles"""
        try:
            # Cargar modelo LightGBM
            lgbm_path = self.models_path / "modelo3_lgbm.pkl"
            if lgbm_path.exists():
                with open(lgbm_path, 'rb') as f:
                    self.models['lightgbm'] = pickle.load(f)
                logger.info("Modelo LightGBM cargado correctamente")
            
            # Cargar modelo Random Forest
            rf_path = self.models_path / "modelo2_rf.pkl"
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                logger.info("Modelo Random Forest cargado correctamente")
            
            # Cargar modelo Logistic Regression
            lr_path = self.models_path / "modelo1_lr.pkl"
            if lr_path.exists():
                with open(lr_path, 'rb') as f:
                    self.models['logistic_regression'] = pickle.load(f)
                logger.info("Modelo Logistic Regression cargado correctamente")
                
            if not self.models:
                logger.warning("No se pudieron cargar modelos IA. Funcionando sin validación IA")
                
        except Exception as e:
            logger.error(f"Error cargando modelos IA: {e}")
            self.models = {}
    
    def validate_signal_with_ai(self, signal: Signal, market_features: Dict[str, Any]) -> bool:
        """
        Valida una señal usando ensemble de modelos IA
        
        Args:
            signal: Señal a validar
            market_features: Features de mercado preparadas
            
        Returns:
            bool: True si la señal es válida según los modelos
        """
        if not self.models:
            logger.warning("No hay modelos IA disponibles. Aprobando señal por defecto")
            return True
            
        try:
            # Preparar features para predicción
            X = self._prepare_features_for_prediction(signal, market_features)
            if X is None:
                logger.error("No se pudieron preparar features. Rechazando señal")
                return False
            
            # Obtener predicciones de todos los modelos
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lightgbm':
                        pred_proba = model.predict(X)
                        # Para LightGBM, convertir a probabilidades si es necesario
                        if len(pred_proba.shape) == 1:
                            prediction = 1 if pred_proba[0] > 0.5 else 0
                            confidence = abs(pred_proba[0] - 0.5) * 2
                        else:
                            prediction = np.argmax(pred_proba[0])
                            confidence = np.max(pred_proba[0])
                    else:
                        # Para sklearn models
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X)
                            prediction = np.argmax(pred_proba[0])
                            confidence = np.max(pred_proba[0])
                        else:
                            prediction = model.predict(X)[0]
                            confidence = 0.7  # Default confidence
                    
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                    
                    logger.debug(f"Modelo {model_name}: predicción={prediction}, confianza={confidence:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error en predicción del modelo {model_name}: {e}")
                    continue
            
            if not predictions:
                logger.error("No se obtuvieron predicciones válidas. Rechazando señal")
                return False
            
            # Ensemble: mayoría ponderada por confianza
            total_weighted_score = 0
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                confidence = confidences[model_name]
                # Convertir predicción a score (-1 para sell, +1 para buy)
                score = 1 if prediction == 1 else -1
                if signal.side == 'sell':
                    score = -score  # Invertir para señales de venta
                
                total_weighted_score += score * confidence
                total_weight += confidence
            
            if total_weight > 0:
                ensemble_score = total_weighted_score / total_weight
                is_valid = ensemble_score > 0.1  # Umbral de confianza
                
                logger.info(f"Validación IA para {signal.symbol} {signal.side}: "
                          f"score={ensemble_score:.3f}, válida={is_valid}")
                return is_valid
            else:
                logger.error("No se pudo calcular ensemble score. Rechazando señal")
                return False
                
        except Exception as e:
            logger.error(f"Error en validación IA: {e}")
            return False
    
    def _prepare_features_for_prediction(self, signal: Signal, market_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Prepara features para predicción del modelo
        
        CRÍTICO: Debe generar exactamente 52 features como en entrenamiento
        """
        try:
            # Obtener features del símbolo específico
            symbol_features = market_features.get(signal.symbol, {})
            if not symbol_features:
                logger.error(f"No hay features disponibles para {signal.symbol}")
                return None
            
            # Lista de features esperadas (debe coincidir con el entrenamiento)
            expected_feature_names = [
                # Features de precio
                'price_rsi', 'price_macd', 'price_macd_signal', 'price_macd_hist',
                'price_change_24h', 'price_ema_10', 'price_ema_20', 'price_sma_10', 'price_sma_20',
                'price_bb_upper', 'price_bb_lower', 'price_bb_middle', 'price_bb_width',
                'price_atr', 'price_obv', 'price_mfi',
                
                # Features de volumen
                'volume_sma_20', 'volume_ratio', 'volume_oscillator',
                'volume_change_1h', 'volume_change_4h', 'volume_change_24h',
                
                # Features multi-timeframe (5m)
                'price_rsi_5m', 'price_macd_5m', 'price_macd_signal_5m', 'price_macd_hist_5m',
                'price_ema_10_5m', 'price_ema_20_5m', 'price_bb_width_5m',
                
                # Features de momentum
                'momentum_roc_1h', 'momentum_roc_4h', 'momentum_roc_24h',
                'momentum_williams_r', 'momentum_cci', 'momentum_stoch_k', 'momentum_stoch_d',
                
                # Features cross-asset (si es ETH)
                'eth_btc_ratio', 'eth_btc_correlation_24h', 'eth_btc_spread',
                
                # Features de mercado general
                'market_regime', 'volatility_regime', 'trend_strength',
                'support_level', 'resistance_level', 'fibonacci_level',
                
                # Features de tiempo
                'hour_of_day', 'day_of_week', 'is_weekend',
                
                # Features adicionales para completar 52
                'feature_43', 'feature_44', 'feature_45', 'feature_46',
                'feature_47', 'feature_48', 'feature_49', 'feature_50',
                'feature_51', 'feature_52'
            ]
            
            # Construir vector de features
            feature_vector = []
            
            for feature_name in expected_feature_names:
                if feature_name in symbol_features:
                    value = symbol_features[feature_name]
                    # Validar que sea numérico
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_vector.append(float(value))
                    else:
                        feature_vector.append(0.0)  # Valor por defecto
                else:
                    # Feature no disponible, usar valor por defecto
                    if 'rsi' in feature_name:
                        feature_vector.append(50.0)  # RSI neutral
                    elif 'volume' in feature_name:
                        feature_vector.append(1.0)   # Volume ratio neutral
                    elif 'correlation' in feature_name:
                        feature_vector.append(0.0)   # Sin correlación
                    elif feature_name in ['hour_of_day', 'day_of_week']:
                        feature_vector.append(float(time.gmtime().tm_hour if 'hour' in feature_name else time.gmtime().tm_wday))
                    elif feature_name == 'is_weekend':
                        feature_vector.append(float(time.gmtime().tm_wday >= 5))
                    else:
                        feature_vector.append(0.0)   # Por defecto
            
            # Verificar que tenemos exactamente 52 features
            if len(feature_vector) != self.expected_features:
                logger.error(f"Feature vector tiene {len(feature_vector)} elementos, esperados {self.expected_features}")
                # Ajustar tamaño si es necesario
                if len(feature_vector) < self.expected_features:
                    feature_vector.extend([0.0] * (self.expected_features - len(feature_vector)))
                else:
                    feature_vector = feature_vector[:self.expected_features]
            
            # Convertir a numpy array con shape correcto
            X = np.array(feature_vector).reshape(1, -1)
            
            logger.debug(f"Features preparadas para {signal.symbol}: shape={X.shape}")
            return X
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            return None


class RiskManager:
    """Gestor de riesgo para validación de órdenes"""
    
    def __init__(self):
        self.position_limits = {
            'BTCUSDT': {'max_position': 0.05, 'max_exposure': 0.20},  # 5% del capital, 20% exposición
            'ETHUSDT': {'max_position': 1.0, 'max_exposure': 0.15}    # 1 ETH, 15% exposición
        }
        
    def validate_order_risk(self, signal: Signal, portfolio: Dict[str, float], current_prices: Dict[str, float]) -> bool:
        """
        Valida si una orden cumple con los límites de riesgo
        
        Args:
            signal: Señal de trading
            portfolio: Portfolio actual
            current_prices: Precios actuales
            
        Returns:
            bool: True si la orden pasa validación de riesgo
        """
        try:
            symbol = signal.symbol
            available_usdt = portfolio.get('USDT', 0.0)
            current_price = current_prices.get(symbol, signal.price or 0)
            
            if current_price <= 0:
                logger.error(f"Precio inválido para {symbol}: {current_price}")
                return False
            
            # Calcular costo de la orden
            order_cost = signal.qty * current_price
            
            # 1. Verificar fondos suficientes para compras
            if signal.side == 'buy' and order_cost > available_usdt * 0.95:  # 95% para fees
                logger.warning(f"Fondos insuficientes para {symbol}: necesario={order_cost:.2f}, disponible={available_usdt:.2f}")
                return False
            
            # 2. Verificar límites de posición
            limits = self.position_limits.get(symbol, {})
            if limits:
                max_position = limits.get('max_position', float('inf'))
                if signal.qty > max_position:
                    logger.warning(f"Cantidad excede límite para {symbol}: {signal.qty} > {max_position}")
                    return False
            
            # 3. Verificar exposición máxima
            total_portfolio_value = self._calculate_portfolio_value(portfolio, current_prices)
            if total_portfolio_value > 0:
                max_exposure = limits.get('max_exposure', 1.0)
                exposure_ratio = order_cost / total_portfolio_value
                
                if exposure_ratio > max_exposure:
                    logger.warning(f"Exposición excede límite para {symbol}: {exposure_ratio:.2%} > {max_exposure:.2%}")
                    return False
            
            # 4. Verificar stop loss obligatorio para posiciones grandes
            if order_cost > 100 and not signal.stop_loss:  # $100+ requiere stop loss
                logger.warning(f"Stop loss requerido para orden grande: {symbol} ${order_cost:.2f}")
                return False
            
            logger.info(f"Validación de riesgo OK para {symbol}: costo=${order_cost:.2f}, exposición={exposure_ratio:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Error en validación de riesgo: {e}")
            return False
    
    def calculate_affordable_position(self, signal: Signal, available_usdt: float, current_price: float) -> float:
        """
        Calcula la cantidad máxima que se puede comprar con los fondos disponibles
        
        Args:
            signal: Señal original
            available_usdt: USDT disponible
            current_price: Precio actual del activo
            
        Returns:
            float: Cantidad ajustada que se puede permitir
        """
        if signal.side != 'buy' or current_price <= 0:
            return signal.qty
            
        # Calcular máximo affordeable (95% para fees)
        max_affordable = (available_usdt * 0.95) / current_price
        
        # Tomar el menor entre la señal original y lo que se puede permitir
        adjusted_qty = min(signal.qty, max_affordable)
        
        # Aplicar límites por símbolo
        limits = self.position_limits.get(signal.symbol, {})
        max_position = limits.get('max_position', float('inf'))
        adjusted_qty = min(adjusted_qty, max_position)
        
        if adjusted_qty != signal.qty:
            logger.info(f"Posición ajustada para {signal.symbol}: {signal.qty} -> {adjusted_qty}")
            
        return adjusted_qty
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calcula el valor total del portfolio en USDT"""
        total_value = portfolio.get('USDT', 0.0)
        
        for symbol, quantity in portfolio.items():
            if symbol != 'USDT' and quantity > 0:
                # Convertir símbolo a par de trading si es necesario
                price_key = symbol if symbol in prices else f"{symbol}USDT"
                price = prices.get(price_key, 0)
                total_value += quantity * price
                
        return total_value


class Executor:
    """Ejecutor de órdenes con validación IA y gestión de riesgo mejorada"""
    
    def __init__(self):
        self.order_counter = 0
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_latency_ms': 0.0,
            'ai_approved': 0,
            'ai_rejected': 0,
            'risk_approved': 0,
            'risk_rejected': 0
        }
        
        # Inicializar gestores
        self.ai_manager = AIModelManager()
        self.risk_manager = RiskManager()
        
    async def execute_order(self, signal: Signal, portfolio: Dict[str, float] = None, 
                          market_features: Dict[str, Any] = None, 
                          current_prices: Dict[str, float] = None) -> ExecutionResult:
        """
        Ejecuta una orden con validación completa IA + Riesgo
        
        Args:
            signal: Señal de trading
            portfolio: Portfolio actual (para validación de fondos)
            market_features: Features de mercado para validación IA
            current_prices: Precios actuales para cálculos de riesgo
        """
        self.order_counter += 1
        order_id = f"L1_ORDER_{self.order_counter}_{int(time.time())}"
        
        logger.info(f"Iniciando ejecución de orden {order_id} para señal {signal.signal_id}")
        start_time = time.time()
        
        try:
            # 1. Validación con modelos IA
            if market_features:
                ai_valid = self.ai_manager.validate_signal_with_ai(signal, market_features)
                if ai_valid:
                    self.execution_metrics['ai_approved'] += 1
                    logger.info(f"Señal {signal.symbol} {signal.side} APROBADA por IA")
                else:
                    self.execution_metrics['ai_rejected'] += 1
                    logger.warning(f"Señal {signal.symbol} {signal.side} RECHAZADA por IA")
                    return self._create_rejection_result(order_id, "AI_REJECTION", start_time)
            else:
                logger.warning("No hay features disponibles para validación IA")
                
            # 2. Validación de riesgo
            if portfolio and current_prices:
                risk_valid = self.risk_manager.validate_order_risk(signal, portfolio, current_prices)
                if risk_valid:
                    self.execution_metrics['risk_approved'] += 1
                    logger.info(f"Señal {signal.symbol} {signal.side} APROBADA por gestión de riesgo")
                else:
                    self.execution_metrics['risk_rejected'] += 1
                    logger.warning(f"Señal {signal.symbol} {signal.side} RECHAZADA por gestión de riesgo")
                    
                    # Intentar ajustar posición si es problema de fondos
                    if signal.side == 'buy':
                        available_usdt = portfolio.get('USDT', 0)
                        current_price = current_prices.get(signal.symbol, signal.price or 0)
                        
                        if current_price > 0:
                            adjusted_qty = self.risk_manager.calculate_affordable_position(
                                signal, available_usdt, current_price
                            )
                            
                            if adjusted_qty > 0 and adjusted_qty != signal.qty:
                                # Crear nueva señal ajustada
                                adjusted_signal = Signal(
                                    signal_id=signal.signal_id + "_ADJUSTED",
                                    symbol=signal.symbol,
                                    side=signal.side,
                                    qty=adjusted_qty,
                                    order_type=signal.order_type,
                                    price=signal.price,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                    strength=signal.strength * 0.8  # Reducir strength por ajuste
                                )
                                
                                logger.info(f"Ejecutando orden ajustada: {signal.qty} -> {adjusted_qty}")
                                return await self._execute_validated_order(order_id, adjusted_signal, start_time)
                    
                    return self._create_rejection_result(order_id, "RISK_REJECTION", start_time)
            else:
                logger.warning("No hay datos de portfolio/precios para validación de riesgo")
                
            # 3. Ejecutar orden validada
            return await self._execute_validated_order(order_id, signal, start_time)
            
        except Exception as e:
            logger.error(f"Error ejecutando orden {order_id}: {e}")
            self._update_metrics(False, (time.time() - start_time) * 1000)
            return self._create_error_result(order_id, str(e), start_time)
    
    async def _execute_validated_order(self, order_id: str, signal: Signal, start_time: float) -> ExecutionResult:
        """Ejecuta una orden ya validada"""
        
        # Crear intent de orden
        order_intent = OrderIntent(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            qty=signal.qty,
            order_type=signal.order_type,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # Ejecutar con retries
        for attempt in range(EXECUTION_CONFIG["MAX_RETRIES"]):
            try:
                result = await self._execute_with_exchange(order_id, order_intent)
                
                latency_ms = (time.time() - start_time) * 1000
                result.latency_ms = latency_ms
                
                # Actualizar métricas
                self._update_metrics(True, latency_ms)
                
                logger.info(f"Orden {order_id} ejecutada exitosamente en {latency_ms:.2f}ms")
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout en orden {order_id}, intento {attempt + 1}")
                if attempt < EXECUTION_CONFIG["MAX_RETRIES"] - 1:
                    await asyncio.sleep(EXECUTION_CONFIG["RETRY_DELAY_SECONDS"])
                    continue
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"Error en orden {order_id}, intento {attempt + 1}: {e}")
                if attempt < EXECUTION_CONFIG["MAX_RETRIES"] - 1:
                    await asyncio.sleep(EXECUTION_CONFIG["RETRY_DELAY_SECONDS"])
                    continue
                else:
                    raise
        
        # Si llegamos aquí, todos los intentos fallaron
        raise Exception(f"Orden {order_id} falló después de {EXECUTION_CONFIG['MAX_RETRIES']} intentos")
    
    async def _execute_with_exchange(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """Ejecuta la orden en el exchange"""
        
        if OPERATION_MODE == "PAPER":
            return await self._simulate_execution(order_id, order_intent)
        elif OPERATION_MODE == "LIVE":
            return await self._live_execution(order_id, order_intent)
        else:
            raise ValueError(f"Modo de operación desconocido: {OPERATION_MODE}")
    
    async def _simulate_execution(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """Simulación mejorada de ejecución"""
        
        # Simular latencia realista
        await asyncio.sleep(0.05 + np.random.exponential(0.02))
        
        # Simular precio de ejecución con slippage realista
        if order_intent.order_type == "market":
            base_price = order_intent.price or 50000  # Precio base
            # Slippage basado en el tamaño de la orden
            size_impact = min(order_intent.qty * 0.001, 0.005)  # Máximo 0.5% de impacto
            slippage_factor = 1 + size_impact if order_intent.side == 'buy' else 1 - size_impact
            execution_price = base_price * slippage_factor
        else:
            execution_price = order_intent.price
        
        # Fees realistas (0.1% para maker, 0.1% para taker)
        fee_rate = 0.001  # 0.1%
        fees = order_intent.qty * execution_price * fee_rate
        
        return ExecutionResult(
            order_id=order_id,
            filled_qty=order_intent.qty,
            avg_price=execution_price,
            fees=fees,
            latency_ms=0.0,  # Se calculará en el caller
            status="FILLED"
        )
    
    async def _live_execution(self, order_id: str, order_intent: OrderIntent) -> ExecutionResult:
        """Ejecución real en exchange"""
        # TODO: Implementar con cliente de Binance real
        raise NotImplementedError("Ejecución en vivo no implementada aún")
    
    def _create_rejection_result(self, order_id: str, reason: str, start_time: float) -> ExecutionResult:
        """Crea resultado para orden rechazada"""
        latency_ms = (time.time() - start_time) * 1000
        self._update_metrics(False, latency_ms)
        
        return ExecutionResult(
            order_id=order_id,
            filled_qty=0.0,
            avg_price=0.0,
            fees=0.0,
            latency_ms=latency_ms,
            status=f"REJECTED_{reason}"
        )
    
    def _create_error_result(self, order_id: str, error: str, start_time: float) -> ExecutionResult:
        """Crea resultado para orden con error"""
        latency_ms = (time.time() - start_time) * 1000
        self._update_metrics(False, latency_ms)
        
        return ExecutionResult(
            order_id=order_id,
            filled_qty=0.0,
            avg_price=0.0,
            fees=0.0,
            latency_ms=latency_ms,
            status=f"ERROR: {error}"
        )
    
    def _update_metrics(self, success: bool, latency_ms: float):
        """Actualiza métricas de ejecución"""
        self.execution_metrics['total_orders'] += 1
        
        if success:
            self.execution_metrics['successful_orders'] += 1
        else:
            self.execution_metrics['failed_orders'] += 1
        
        # Actualizar latencia promedio
        current_avg = self.execution_metrics['avg_latency_ms']
        total = self.execution_metrics['total_orders']
        if total > 0:
            self.execution_metrics['avg_latency_ms'] = (current_avg * (total - 1) + latency_ms) / total
        
        # Warning si latencia es alta
        if latency_ms > EXECUTION_CONFIG.get("LATENCY_WARNING_MS", 1000):
            logger.warning(f"Latencia alta detectada: {latency_ms:.2f}ms")
    
    def get_metrics(self) -> dict:
        """Retorna métricas completas de ejecución"""
        metrics = self.execution_metrics.copy()
        
        # Calcular tasas de éxito
        total = metrics['total_orders']
        if total > 0:
            metrics['success_rate'] = metrics['successful_orders'] / total
            metrics['ai_approval_rate'] = metrics['ai_approved'] / (metrics['ai_approved'] + metrics['ai_rejected']) if (metrics['ai_approved'] + metrics['ai_rejected']) > 0 else 0
            metrics['risk_approval_rate'] = metrics['risk_approved'] / (metrics['risk_approved'] + metrics['risk_rejected']) if (metrics['risk_approved'] + metrics['risk_rejected']) > 0 else 0
        
        return metrics