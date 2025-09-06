# l1_operational/order_manager.py - CORREGIDO
from typing import Dict, Any, Optional, List
import joblib
import pickle
import os
import uuid
import time
import asyncio
from core.logging import logger
from .config import ConfigObject
from .models import Signal, create_signal
from l2_tactic.models import TacticalSignal
from .signal_processor import process_tactical_signal


class OrderManager:
    def __init__(self, binance_client=None, market_data=None):
        """
        Inicializa el OrderManager usando la configuración de constantes.
        """
        self.config = ConfigObject
        self.binance_client = binance_client
        self.market_data = market_data or {}
        self.models = {}
        self._models_loaded = False
        self.active_orders = {}
        self.execution_stats = {}
        self.risk_limits = ConfigObject.RISK_LIMITS
        self.portfolio_limits = ConfigObject.PORTFOLIO_LIMITS
        self.execution_config = ConfigObject.EXECUTION_CONFIG
        
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
                self._models_loaded = True
        
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
        
        for model_name, model_path in model_configs:
            try:
                model = self._load_model_safely(model_path, model_name)
                if model:
                    self.models[model_name] = model
                    logger.info(f"✅ Modelo {model_name} cargado desde {model_path}")
                else:
                    logger.error(f"❌ Falló carga de {model_name}")
            except Exception as e:
                logger.error(f"❌ Error cargando modelo {model_name}: {e}", exc_info=True)

    def _load_model_safely(self, model_path: str, model_name: str):
        """
        Carga un modelo de forma segura manejando múltiples formatos
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"❌ No existe {model_path}")
                return None

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except:
                model = joblib.load(model_path)
                return model
                
        except Exception as e:
            logger.error(f"❌ Error cargando {model_name}: {e}")
            return None

    def close_connections(self):
        logger.info("🔌 close_connections() no implementado aún")

    def shutdown(self):
        logger.info("🛑 Cerrando OrderManager")
        self.close_connections()

    async def process_signals(self, signals: List[Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa señales tácticas con modelos IA cuando estén disponibles.
        Retorna una lista de órdenes procesadas.
        """
        def get_attr_or_key(obj, key, default=None):
            """Helper to get attributes or dictionary keys safely"""
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            if isinstance(obj, list) and len(obj) > 0:
                for item in obj:
                    if isinstance(item, dict) and key in item:
                        return item[key]
            return default

        processed_orders = []

        # Ensure signals is always a list
        if not isinstance(signals, list):
            signals = [signals]

        for signal in signals:
            try:
                signal_obj = None
                
                # Convertir TacticalSignal a Signal si es necesario
                if isinstance(signal, TacticalSignal):
                    try:
                        # Get price from features if not set directly
                        price = getattr(signal, 'price', None)
                        if price is None and hasattr(signal, 'features'):
                            price = signal.features.get('close')
                        
                        # Convert TacticalSignal to Signal format with all required fields
                        signal_dict = {
                            'signal_id': str(uuid.uuid4()),
                            'symbol': signal.symbol,
                            'side': signal.side,
                            'order_type': signal.type or 'market',  # Default to market if not specified
                            'strength': signal.strength,
                            'confidence': signal.confidence,
                            'timestamp': signal.timestamp.timestamp() if hasattr(signal.timestamp, 'timestamp') else signal.timestamp,
                            'features': signal.features or {},  # Include features for L1 AI processing
                            'technical_indicators': signal.features or {},  # Duplicate features for backward compatibility
                            'signal_type': signal.signal_type,
                            'strategy_id': 'L2_TACTIC',
                            'price': float(price) if price is not None else None,  # Ensure price is included
                            'stop_loss': signal.stop_loss if hasattr(signal, 'stop_loss') else None,
                            'take_profit': signal.take_profit if hasattr(signal, 'take_profit') else None,
                            'quantity': signal.quantity if hasattr(signal, 'quantity') else 0.0,  # Will be recalculated
                        }
                        
                        # Calculate quantity for the order
                        calculated_qty = self._calculate_order_quantity(signal)
                        signal_dict['quantity'] = calculated_qty
                        signal_dict['qty'] = calculated_qty  # For backward compatibility
                        
                        # Ensure all required fields are present
                        if not signal_dict.get('technical_indicators'):
                            signal_dict['technical_indicators'] = {}
                        
                        # Add strength and confidence to technical indicators
                        signal_dict['technical_indicators'].update({
                            'signal_strength': float(signal.strength),
                            'confidence': getattr(signal, 'confidence', 0.0)
                        })
                        
                        # Create the Signal object
                        signal_obj = create_signal(**signal_dict)
                        logger.debug(f"✅ Señal táctica convertida: {signal_obj.symbol} {signal_obj.side} qty={signal_dict['quantity']:.8f}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error creando señal para {signal.symbol}: {e}", exc_info=True)
                        logger.warning(f"⚠️ Señal inválida: {signal}")
                        rec = {
                            'symbol': getattr(signal, 'symbol', 'unknown'),
                            'side': getattr(signal, 'side', 'unknown'),
                            'status': 'rejected',
                            'reason': f'invalid_signal: {str(e)}',
                            'order_id': None
                        }
                        processed_orders.append(rec)
                        if state is not None:
                            state.setdefault('ordenes', []).append(self._make_order_record(rec))
                        continue
                
                elif isinstance(signal, Signal):
                    signal_obj = signal
                elif isinstance(signal, dict):
                    try:
                        # Handle dictionary signals (from signal_composer)
                        timestamp = signal.get('timestamp')
                        if hasattr(timestamp, 'timestamp'):
                            timestamp = timestamp.timestamp()
                            
                        signal_dict = {
                            'signal_id': str(uuid.uuid4()),
                            'symbol': signal.get('symbol'),
                            'side': signal.get('side'),
                            'order_type': signal.get('type', 'market'),
                            'qty': float(signal.get('quantity', 0.0)),
                            'strength': float(signal.get('strength', 0.0)),
                            'confidence': float(signal.get('confidence', 0.0)),
                            'timestamp': timestamp or time.time(),
                            'features': signal.get('features', {}),
                            'technical_indicators': signal.get('features', {}),
                            'signal_type': signal.get('signal_type', 'tactical'),
                            'strategy_id': signal.get('source', 'L2_TACTIC'),
                            'price': float(signal.get('price', 0.0)) if signal.get('price') else None,
                            'stop_loss': signal.get('stop_loss'),
                            'take_profit': signal.get('take_profit')
                        }
                        # Create the Signal object
                        # Handle both dict and list cases
                        signal_data = signal
                        if isinstance(signal, list) and len(signal) > 0:
                            signal_data = signal[0]  # Take first signal from list
                        
                        signal_obj = create_signal(**signal_dict)
                        logger.debug(f"✅ Señal dict convertida: {signal_obj.symbol} {signal_obj.side}")
                    except Exception as e:
                        logger.error(f"❌ Error creando señal desde dict: {e}", exc_info=True)
                        logger.warning(f"⚠️ Señal dict inválida: {signal}")
                        symbol = get_attr_or_key(signal, 'symbol', 'unknown')
                        side = get_attr_or_key(signal, 'side', 'unknown')
                        rec = {
                            'symbol': symbol,
                            'side': side,
                            'status': 'rejected',
                            'reason': f'invalid_dict_signal: {str(e)}',
                            'order_id': None
                        }
                        processed_orders.append(rec)
                        if state is not None:
                            state.setdefault('ordenes', []).append(self._make_order_record(rec))
                        continue
                else:
                    logger.warning(f"⚠️ Señal inválida de tipo desconocido: {type(signal)}")
                    rec = {
                        'symbol': str(signal),
                        'side': 'unknown',
                        'status': 'rejected',
                        'reason': 'unknown_signal_type',
                        'order_id': None
                    }
                    processed_orders.append(rec)
                    if state is not None:
                        state.setdefault('ordenes', []).append(self._make_order_record(rec))
                    continue

                if signal_obj is None:
                    continue  # Skip this iteration if signal conversion failed

                # Validar señal con modelos IA y procesar
                if self._models_loaded and self.models:
                    try:
                        ai_validation = await self._validate_signal_with_ai(signal_obj)
                        if not ai_validation['approved']:
                            logger.info(f"🤖 Señal rechazada por IA: {signal_obj.symbol} - {ai_validation['reason']}")
                            rec = {
                                'symbol': signal_obj.symbol,
                                'side': signal_obj.side,
                                'status': 'rejected',
                                'reason': f"ai_validation_{ai_validation['reason']}",
                                'order_id': None,
                                'ai_score': ai_validation.get('score', 0)
                            }
                            processed_orders.append(rec)
                            if state is not None:
                                state.setdefault('ordenes', []).append(self._make_order_record(rec))
                            continue

                    except Exception as e:
                        logger.error(f"❌ Error en validación IA: {e}", exc_info=True)
                        # Continue processing even if AI validation fails
                
                # Validar y procesar la orden
                try:
                    quantity = signal_obj.qty if isinstance(signal_obj, Signal) else self._calculate_order_quantity(signal_obj)
                    if quantity <= 0:
                        logger.warning(f"⚠️ Cantidad calculada inválida para {signal_obj.symbol}: {quantity}")
                        rec = {
                            'symbol': signal_obj.symbol,
                            'side': signal_obj.side,
                            'status': 'rejected',
                            'reason': 'invalid_quantity',
                            'order_id': None
                        }
                        processed_orders.append(rec)
                        if state is not None:
                            state.setdefault('ordenes', []).append(self._make_order_record(rec))
                        continue
                    
                    if self.config.OPERATION_MODE == 'PAPER':
                        order_result = self._execute_paper_order(signal_obj, quantity)
                    else:
                        order_result = await self._execute_real_order(signal_obj, quantity)
                    
                    self._update_execution_stats(order_result)
                    processed_orders.append(order_result)
                    if state is not None:
                        state.setdefault('ordenes', []).append(self._make_order_record(order_result))
                    
                    logger.info(f"✅ Señal procesada: {signal_obj.symbol} {signal_obj.side} - Status: {order_result['status']}")
                
                except Exception as e:
                    logger.error(f"❌ Error procesando orden: {e}", exc_info=True)
                    rec = {
                        'symbol': signal_obj.symbol,
                        'side': signal_obj.side,
                        'status': 'error',
                        'reason': f'processing_error: {str(e)}',
                        'order_id': None
                    }
                    processed_orders.append(rec)
                    if state is not None:
                        state.setdefault('ordenes', []).append(self._make_order_record(rec))

            except Exception as e:
                logger.error(f"❌ Error crítico procesando señal: {e}", exc_info=True)
                rec = {
                    'symbol': get_attr_or_key(signal, 'symbol', 'unknown'),
                    'side': get_attr_or_key(signal, 'side', 'unknown'),
                    'status': 'error',
                    'reason': f'critical_error: {str(e)}',
                    'order_id': None
                }
                processed_orders.append(rec)
                if state is not None:
                    state.setdefault('ordenes', []).append(self._make_order_record(rec))

        return processed_orders

        # Note: kept for compatibility; actual return happens above

    def _make_order_record(self, order_like: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza distintos formatos de resultado de orden a un registro estándar."""
        try:
            record = {}
            # If order_like already has common keys, use them
            record['order_id'] = order_like.get('order_id') or order_like.get('order_id') or str(uuid.uuid4())
            record['symbol'] = order_like.get('symbol')
            record['side'] = order_like.get('side')
            record['status'] = order_like.get('status') or order_like.get('order_status') or 'unknown'
            record['reason'] = order_like.get('reason')
            record['ts'] = order_like.get('ts') or time.time()
            record['quantity'] = order_like.get('quantity') or order_like.get('qty') or 0.0
            record['price'] = order_like.get('price')
            record['source'] = order_like.get('strategy_id') or order_like.get('source') or 'L1'
            # Keep original payload for debugging
            record['raw'] = order_like
            return record
        except Exception:
            return {
                'order_id': str(uuid.uuid4()),
                'symbol': order_like.get('symbol') if isinstance(order_like, dict) else str(order_like),
                'side': order_like.get('side') if isinstance(order_like, dict) else 'unknown',
                'status': order_like.get('status') if isinstance(order_like, dict) else 'unknown',
                'reason': order_like.get('reason') if isinstance(order_like, dict) else None,
                'ts': time.time(),
                'quantity': order_like.get('quantity', 0.0) if isinstance(order_like, dict) else 0.0,
                'price': order_like.get('price') if isinstance(order_like, dict) else None,
                'source': 'L1',
                'raw': order_like
            }

    async def _validate_signal_with_ai(self, signal) -> Dict[str, Any]:
        """
        Valida una señal usando los modelos de IA
        """
        try:
            # Extract features from technical_indicators
            if not (hasattr(signal, 'technical_indicators') and signal.technical_indicators):
                return {'approved': False, 'reason': 'no_features', 'score': 0.0}
            
            if not self._models_loaded:
                return {'approved': True, 'reason': 'no_models', 'score': 0.5}

            # Prepare all required features for model prediction
            feature_names = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 
                'macd_signal', 'rsi', 'bollinger_middle', 'bollinger_std',
                'bollinger_upper', 'bollinger_lower', 'vol_mean_20', 
                'vol_std_20', 'vol_zscore'
            ]

            # Additional calculated features
            def safe_div(a, b, default=0.0):
                """Safe division handling zeros"""
                try:
                    if b == 0:
                        return default
                    return a / b
                except:
                    return default

            def calculate_additional_features(indicators):
                base_features = {k: float(indicators.get(k, 0.0)) for k in feature_names}
                
                # Price changes
                base_features['price_change'] = base_features['close'] - base_features['open']
                base_features['price_change_pct'] = safe_div(base_features['price_change'], base_features['open']) * 100
                base_features['high_low_range'] = base_features['high'] - base_features['low']
                base_features['high_low_range_pct'] = safe_div(base_features['high_low_range'], base_features['open']) * 100
                
                # Volume analysis
                base_features['volume_price_trend'] = base_features['volume'] * base_features['price_change']
                base_features['volume_intensity'] = safe_div(base_features['volume'], base_features['vol_mean_20'])
                
                # Technical combinations
                base_features['macd_trend'] = base_features['macd'] - base_features['macd_signal']
                base_features['bb_range'] = base_features['bollinger_upper'] - base_features['bollinger_lower']
                base_features['bb_position'] = safe_div(base_features['close'] - base_features['bollinger_lower'], 
                                                      base_features['bb_range'], 0.5)
                base_features['sma_trend'] = safe_div(base_features['sma_20'], base_features['sma_50']) - 1 if base_features['sma_50'] != 0 else 0
                
                # Momentum and Volatility
                base_features['momentum'] = safe_div(base_features['close'], base_features['sma_20'])
                base_features['volatility'] = safe_div(base_features['bollinger_std'], base_features['bollinger_middle'])
                
                return base_features
            
            # Extract features as a numpy array
            import numpy as np
            all_features = calculate_additional_features(signal.technical_indicators)
            
            # Validate features and handle missing/invalid values
            def validate_feature(name, value, min_val=-1e6, max_val=1e6):
                try:
                    val = float(value)
                    if not np.isfinite(val) or val < min_val or val > max_val:
                        return 0.0
                    return val
                except:
                    return 0.0

            # Ensure consistent feature order for models
            model_feature_names = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd',
                'macd_signal', 'rsi', 'bollinger_middle', 'bollinger_std',
                'bollinger_upper', 'bollinger_lower', 'vol_mean_20',
                'vol_std_20', 'vol_zscore', 'price_change', 'price_change_pct',
                'high_low_range', 'high_low_range_pct', 'volume_price_trend',
                'volume_intensity', 'macd_trend', 'bb_position', 'sma_trend',
                'momentum', 'volatility'
            ]
            
            features = []
            for name in model_feature_names:
                value = all_features.get(name, 0.0)
                validated_value = validate_feature(name, value)
                features.append(validated_value)
            
            # Add signal-specific features with validation
            def get_signal_feature(obj, attr, default=0.0):
                try:
                    value = getattr(obj, attr, default)
                    return float(value) if value is not None else default
                except (TypeError, ValueError):
                    return default

            signal_features = [
                get_signal_feature(signal, 'strength', 0.0),
                get_signal_feature(signal, 'confidence', 0.0),
                float(signal.technical_indicators.get('signal_strength', 0.0))
            ]

            # Validate signal features
            signal_features = [validate_feature('signal_' + str(i), v, -1, 1) 
                             for i, v in enumerate(signal_features)]
            
            # Extend features list with signal features
            features.extend(signal_features)

            # Final validation to ensure exactly 52 features
            if len(features) < 52:
                features.extend([0.0] * (52 - len(features)))
            elif len(features) > 52:
                features = features[:52]            # Reshape and scale features for sklearn models
            X = np.array(features).reshape(1, -1)
            
            # First check signal strength and confidence
            signal_strength = float(signal.technical_indicators.get('signal_strength', 0.0))
            signal_confidence = float(getattr(signal, 'confidence', 0.0))
            
            # Calculate signal quality score
            quality_score = (signal_strength + signal_confidence) / 2
            
            if signal_confidence >= 0.8 and signal_strength >= 0.7:
                # High confidence signals bypass AI validation
                logger.debug(f"✅ Señal aprobada por alta confianza: conf={signal_confidence:.2f}, strength={signal_strength:.2f}")
                return {
                    'approved': True,
                    'reason': 'high_confidence',
                    'score': quality_score
                }

            # Make predictions with each model
            predictions = []
            model_scores = []
            
            for model_name, model in self.models.items():
                try:
                    pred_score = None
                    if model_name == 'lightgbm':
                        # LightGBM models
                        pred = model.predict(X)
                        pred_score = float(pred[0])
                    elif hasattr(model, 'predict_proba'):
                        # Models with probability predictions
                        pred = model.predict_proba(X)
                        pred_score = float(pred[0][1])
                    else:
                        # Basic models without probabilities
                        pred = model.predict(X) 
                        pred_score = float(pred[0])

                    if pred_score is not None:
                        predictions.append(pred_score)
                        model_scores.append({
                            'model': model_name,
                            'score': pred_score
                        })
                        logger.debug(f"Modelo {model_name} prediction: {pred_score:.3f}")
                except Exception as e:
                    logger.error(f"❌ Error en modelo {model_name}: {str(e)}")
                    continue

            if not predictions:
                logger.warning("⚠️ No se pudieron obtener predicciones")
                return {'approved': False, 'reason': 'no_predictions', 'score': 0.0}

            # Calculate ensemble score
            ensemble_score = sum(predictions) / len(predictions)
            
            # Approval threshold 
            APPROVAL_THRESHOLD = 0.6
            approved = ensemble_score >= APPROVAL_THRESHOLD
            
            result = {
                'approved': approved,
                'reason': 'ensemble_decision',
                'score': float(ensemble_score),
                'model_scores': model_scores
            }
            
            status = "✅" if approved else "❌"
            logger.debug(f"{status} Resultado validación: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Error en validación AI: {str(e)}")
            return {'approved': False, 'reason': str(e), 'score': 0.0}

    def _check_risk_limits(self, signal) -> bool:
        """
        Verifica los límites de riesgo para una señal
        """
        try:
            symbol = signal.symbol
            if symbol == 'BTCUSDT':
                max_size = self.risk_limits.get('MAX_ORDER_SIZE_BTC', 0.01)
            elif symbol == 'ETHUSDT':
                max_size = self.risk_limits.get('MAX_ORDER_SIZE_ETH', 0.1)
            else:
                logger.warning(f"⚠️ Símbolo no configurado: {symbol}")
                return False
            
            # For TacticalSignal, check strength from attributes
            min_strength = self.risk_limits.get('MIN_SIGNAL_STRENGTH', 0.6)
            if hasattr(signal, 'strength'):
                strength = signal.strength
            elif hasattr(signal, 'technical_indicators'):
                # For Signal objects, check strength from technical indicators
                strength = signal.technical_indicators.get('signal_strength', 0.0)
            else:
                strength = 0.0

            # More permissive with high confidence signals
            signal_conf = getattr(signal, 'confidence', 0.0)
            if signal_conf >= 0.7:
                min_strength *= 0.8  # Reduce minimum strength requirement for high confidence signals

            if strength < min_strength:
                logger.debug(f"Señal {symbol} rechazada por strength baja: {strength:.3f} < {min_strength}")
                return False

            min_confidence = self.risk_limits.get('MIN_CONFIDENCE', 0.5)
            if signal_conf >= 0.7:
                # High confidence signals bypass normal confidence checks
                logger.debug(f"Señal {symbol} aceptada por alta confianza: {signal_conf:.3f}")
                return True
            elif signal_conf < min_confidence:
                logger.debug(f"Señal {symbol} rechazada por confidence baja: {signal_conf:.3f} < {min_confidence}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"❌ Error verificando límites de riesgo: {e}")
            return False
