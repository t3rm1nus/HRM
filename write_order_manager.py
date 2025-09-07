import os

code = '''# l1_operational/order_manager.py - CORREGIDO
import logging
import joblib
import pickle
import os
import uuid
import time
from typing import Dict, Any, Optional, List
from .config import ConfigObject
from .models import Signal, create_signal
from l2_tactic.models import TacticalSignal
from .signal_processor import process_tactical_signal
import asyncio
from core.logging import logger

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

    async def process_signals(self, signals: List[Any]) -> Dict[str, Any]:
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
            return default

        processed_orders = []

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
                            'qty': signal.quantity if hasattr(signal, 'quantity') else None,  # Transfer original quantity if exists
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
                        processed_orders.append({
                            'symbol': getattr(signal, 'symbol', 'unknown'),
                            'side': getattr(signal, 'side', 'unknown'),
                            'status': 'rejected',
                            'reason': f'invalid_signal: {str(e)}',
                            'order_id': None
                        })
                        continue
                
                elif isinstance(signal, Signal):
                    signal_obj = signal
                else:
                    logger.warning(f"⚠️ Señal inválida de tipo desconocido: {type(signal)}")
                    processed_orders.append({
                        'symbol': str(signal),
                        'side': 'unknown',
                        'status': 'rejected',
                        'reason': 'unknown_signal_type',
                        'order_id': None
                    })
                    continue

                if signal_obj is None:
                    continue  # Skip this iteration if signal conversion failed

                # Validar señal con modelos IA y procesar
                if self._models_loaded and self.models:
                    try:
                        ai_validation = await self._validate_signal_with_ai(signal_obj)
                        if not ai_validation['approved']:
                            logger.info(f"🤖 Señal rechazada por IA: {signal_obj.symbol} - {ai_validation['reason']}")
                            processed_orders.append({
                                'symbol': signal_obj.symbol,
                                'side': signal_obj.side,
                                'status': 'rejected',
                                'reason': f"ai_validation_{ai_validation['reason']}",
                                'order_id': None,
                                'ai_score': ai_validation.get('score', 0)
                            })
                            continue

                    except Exception as e:
                        logger.error(f"❌ Error en validación IA: {e}", exc_info=True)
                        # Continue processing even if AI validation fails
                
                # Validar y procesar la orden
                try:
                    quantity = signal_obj.qty if isinstance(signal_obj, Signal) else self._calculate_order_quantity(signal_obj)
                    if quantity <= 0:
                        logger.warning(f"⚠️ Cantidad calculada inválida para {signal_obj.symbol}: {quantity}")
                        processed_orders.append({
                            'symbol': signal_obj.symbol,
                            'side': signal_obj.side,
                            'status': 'rejected',
                            'reason': 'invalid_quantity',
                            'order_id': None
                        })
                        continue
                    
                    if self.config.OPERATION_MODE == 'PAPER':
                        order_result = self._execute_paper_order(signal_obj, quantity)
                    else:
                        order_result = await self._execute_real_order(signal_obj, quantity)
                    
                    self._update_execution_stats(order_result)
                    processed_orders.append(order_result)
                    
                    logger.info(f"✅ Señal procesada: {signal_obj.symbol} {signal_obj.side} - Status: {order_result['status']}")
                
                except Exception as e:
                    logger.error(f"❌ Error procesando orden: {e}", exc_info=True)
                    processed_orders.append({
                        'symbol': signal_obj.symbol,
                        'side': signal_obj.side,
                        'status': 'error',
                        'reason': f'processing_error: {str(e)}',
                        'order_id': None
                    })

            except Exception as e:
                logger.error(f"❌ Error crítico procesando señal: {e}", exc_info=True)
                processed_orders.append({
                    'symbol': get_attr_or_key(signal, 'symbol', 'unknown'),
                    'side': get_attr_or_key(signal, 'side', 'unknown'),
                    'status': 'error',
                    'reason': f'critical_error: {str(e)}',
                    'order_id': None
                })

        return processed_orders
'''

# Write the Python code to a file
file_path = 'c:/proyectos/HRM/l1_operational/order_manager.py'
with open(file_path, 'w') as f:
    f.write(code)
