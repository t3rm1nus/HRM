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
                logger.debug(f"🔍 Tipo de señal recibida: {type(signal)}")
                if isinstance(signal, dict):
                    logger.debug(f"🔍 Contenido del dict: {signal}")
                signal_obj = None
                
                # Convertir TacticalSignal a Signal si es necesario
                if isinstance(signal, TacticalSignal):
                    try:
                        logger.debug(f"🔄 Convirtiendo TacticalSignal: {signal.symbol} {signal.side} quantity={getattr(signal, 'quantity', 'None')}")
                        
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
                            'take_profit': signal.take_profit if hasattr(signal, 'take_profit') else None
                        }
                        
                        # Use existing quantity if available, otherwise calculate
                        original_qty = getattr(signal, 'quantity', None)
                        if original_qty is not None and float(original_qty) > 0:
                            signal_dict['qty'] = float(original_qty)
                            logger.debug(f"✅ Usando cantidad existente: {original_qty:.8f}")
                        else:
                            calculated_qty = self._calculate_order_quantity(signal)
                            signal_dict['qty'] = calculated_qty
                            logger.debug(f"✅ Usando cantidad calculada: {calculated_qty:.8f}")
                        
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
                        logger.debug(f"✅ Señal táctica convertida: {signal_obj.symbol} {signal_obj.side} qty={signal_dict['qty']:.8f}")
                        logger.debug(f"🔍 Signal object qty attribute: {getattr(signal_obj, 'qty', 'None')}")
                        
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
                    logger.debug(f"🔄 Procesando Signal object: {signal.symbol} {signal.side} qty={signal.qty}")
                    signal_obj = signal
                elif isinstance(signal, dict):
                    try:
                        # Handle dictionary signals (from signal_composer)
                        timestamp = signal.get('timestamp')
                        if hasattr(timestamp, 'timestamp'):
                            timestamp = timestamp.timestamp()
                        
                        # Get quantity from signal dict
                        quantity = signal.get('quantity', 0.0)
                        logger.debug(f"🔄 Procesando señal dict: {signal.get('symbol')} {signal.get('side')} quantity={quantity}")
                        
                        # Si quantity es 0, intentar calcularla
                        if quantity <= 0:
                            # Calcular cantidad basada en strength y símbolo
                            strength = signal.get('strength', 0.5)
                            symbol = signal.get('symbol', '')
                            if symbol == 'BTCUSDT':
                                quantity = 0.01 * (1 + strength)
                            elif symbol == 'ETHUSDT':
                                quantity = 0.1 * (1 + strength)
                            logger.debug(f"🔄 Cantidad calculada: {quantity}")
                            
                        signal_dict = {
                            'signal_id': str(uuid.uuid4()),
                            'symbol': signal.get('symbol'),
                            'side': signal.get('side'),
                            'order_type': signal.get('type', 'market'),
                            'qty': float(quantity) if quantity > 0 else 0.0,
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
                    logger.info(f"🔍 Procesando señal para {signal_obj.symbol}: type={type(signal_obj)}, attrs={dir(signal_obj)}")
                    qty_from_signal = getattr(signal_obj, 'qty', None)
                    logger.info(f"💫 Cantidad en señal: {qty_from_signal}")
                    
                    quantity = qty_from_signal if qty_from_signal is not None else self._calculate_order_quantity(signal_obj)
                    logger.info(f"📊 Cantidad calculada: {quantity}")
                    # Ajuste por capital/holdings disponible antes de ejecutar
                    quantity_adj = self._adjust_quantity_for_capital_and_holdings(signal_obj, float(quantity), state)
                    if quantity_adj != quantity:
                        logger.info(f"🧮 Cantidad ajustada final: {quantity:.8f} -> {quantity_adj:.8f}")
                    quantity = quantity_adj
                    
                    if quantity <= 0:
                        logger.warning(f"⚠️ Cantidad calculada inválida para {signal_obj.symbol}: {quantity}")
                        rec = {
                            'symbol': signal_obj.symbol,
                            'side': signal_obj.side,
                            'status': 'rejected',
                            'reason': 'invalid_or_unaffordable_quantity',
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

    def _get_current_price(self, symbol: str, signal) -> float:
        """Obtiene el precio actual del símbolo desde la señal/market_data (con fallback)."""
        try:
            if hasattr(signal, 'price') and signal.price:
                return float(signal.price)
            if isinstance(self.market_data, dict):
                md = self.market_data.get(symbol) or {}
                if isinstance(md, dict):
                    p = md.get('close')
                    if p:
                        return float(p)
            if hasattr(signal, 'features') and isinstance(signal.features, dict):
                p = signal.features.get('close')
                if p:
                    return float(p)
        except Exception:
            pass
        return 110000.0 if symbol == 'BTCUSDT' else 4300.0 if symbol == 'ETHUSDT' else 1000.0

    def _get_portfolio_balances(self, state: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Extrae USDT y sizes por símbolo del state (seguro por defecto)."""
        balances = {'USDT': 0.0, 'BTCUSDT': 0.0, 'ETHUSDT': 0.0}
        try:
            if not state:
                return balances
            portfolio = state.get('portfolio', {}) or {}
            balances['USDT'] = float(portfolio.get('USDT', 0.0))
            positions = portfolio.get('positions', {}) or {}
            balances['BTCUSDT'] = float((positions.get('BTCUSDT') or {}).get('size', 0.0))
            balances['ETHUSDT'] = float((positions.get('ETHUSDT') or {}).get('size', 0.0))
        except Exception:
            return balances
        return balances

    def _adjust_quantity_for_capital_and_holdings(self, signal: Signal, quantity: float, state: Optional[Dict[str, Any]]) -> float:
        """Ajusta qty para no exceder USDT disponible (compras) ni holdings (ventas),
        aplicando reservas: hard floor (1%) y soft reserve (15%). La soft reserve puede
        saltarse en señales de alta convicción (confidence >= 0.8)."""
        try:
            symbol = signal.symbol
            side = str(signal.side).lower()
            price = self._get_current_price(symbol, signal)

            # Cap por límites de configuración
            max_size_cfg = None
            if symbol == 'BTCUSDT':
                max_size_cfg = float(self.risk_limits.get('MAX_ORDER_SIZE_BTC', 0.01))
            elif symbol == 'ETHUSDT':
                max_size_cfg = float(self.risk_limits.get('MAX_ORDER_SIZE_ETH', 0.1))
            if max_size_cfg is not None and quantity > max_size_cfg:
                logger.info(f"🔧 Cap config {symbol}: {quantity} -> {max_size_cfg}")
                quantity = max_size_cfg

            balances = self._get_portfolio_balances(state)
            usdt = balances.get('USDT', 0.0)
            total_value = 0.0
            try:
                if state:
                    total_value = float(state.get('total_value', state.get('initial_capital', 0.0)) or 0.0)
            except Exception:
                total_value = 0.0
            holdings = balances.get(symbol, 0.0)

            fee_rate = 0.001  # 0.1%
            slip = 0.001      # 0.1%
            hard_floor_pct = 0.01
            soft_reserve_pct = 0.15
            high_conf_threshold = 0.8

            hard_floor_usdt = total_value * hard_floor_pct if total_value > 0 else 0.0
            soft_reserve_usdt = total_value * soft_reserve_pct if total_value > 0 else 0.0

            if side == 'buy':
                denom = price * (1.0 + fee_rate + slip)
                # Capacidad sin romper hard floor
                capacity_hard = max(0.0, usdt - hard_floor_usdt)
                # Capacidad respetando soft reserve (por defecto)
                capacity_soft = max(0.0, usdt - soft_reserve_usdt)
                # Alta convicción permite usar hasta hard floor
                confidence = 0.0
                try:
                    confidence = float(getattr(signal, 'confidence', 0.0) or 0.0)
                except Exception:
                    confidence = 0.0
                allowed_spend = capacity_hard if confidence >= high_conf_threshold else capacity_soft

                affordable = (allowed_spend / denom) if denom > 0 else 0.0
                if affordable < quantity:
                    # Mensaje específico según el motivo del ajuste
                    if allowed_spend == capacity_soft and soft_reserve_usdt > 0:
                        logger.info(
                            f"🟦 Soft reserve activa ({soft_reserve_pct*100:.0f}%): {quantity:.8f} -> {affordable:.8f} "
                            f"(USDT={usdt:.2f}, reserve={soft_reserve_usdt:.2f})"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Ajuste por USDT/fees en {symbol}: {quantity:.8f} -> {affordable:.8f} (USDT={usdt:.2f})"
                        )
                quantity = min(quantity, max(0.0, affordable))
            elif side == 'sell':
                if holdings < quantity:
                    logger.warning(f"⚠️ Ajuste por holdings en {symbol}: {quantity:.8f} -> {holdings:.8f}")
                quantity = min(quantity, max(0.0, holdings))

            return float(quantity if quantity and quantity > 0 else 0.0)
        except Exception as e:
            logger.error(f"❌ Error ajustando qty por capital/holdings: {e}")
            return float(quantity or 0.0)

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

            # Calculate ensemble score from L1 models
            ensemble_score = sum(predictions) / len(predictions)
            
            # Get L2 PPO signal strength
            ppo_strength = float(getattr(signal, 'strength', 0.0))
            
            # Si hay alta confianza en L2 o condiciones técnicas claras, dar más peso a su decisión
            has_strong_technicals = (
                signal.technical_indicators.get('rsi', 50) < 30 or  # Oversold
                signal.technical_indicators.get('rsi', 50) > 70     # Overbought
            )
            
            if ppo_strength >= 0.7 or has_strong_technicals:
                combined_score = (ensemble_score * 0.2 + ppo_strength * 0.8)  # Más peso a L2
                logger.info(f"L2 dominante: PPO={ppo_strength:.3f}, L1={ensemble_score:.3f}, tech={has_strong_technicals}")
            else:
                combined_score = (ensemble_score * 0.5 + ppo_strength * 0.5)
                logger.info(f"Balance L1/L2: PPO={ppo_strength:.3f}, L1={ensemble_score:.3f}")
            
            # Más permisivo cuando hay acuerdo entre L1 y L2 o señales técnicas fuertes
            agreement_factor = 1 - abs(ensemble_score - ppo_strength)
            if has_strong_technicals:
                agreement_factor = min(1.0, agreement_factor * 1.2)  # Boost con señales técnicas
            confidence_boost = agreement_factor * 0.2  # Max 20% boost when perfect agreement
            
            final_score = combined_score * (1 + confidence_boost)
            
            # More permisive threshold since we're using combined intelligence
            APPROVAL_THRESHOLD = 0.45  # Ligeramente más permisivo con validación mejorada
            approved = final_score >= APPROVAL_THRESHOLD
            
            result = {
                'approved': approved,
                'reason': 'combined_l1_l2_decision',
                'score': float(final_score),
                'l1_score': float(ensemble_score),
                'l2_score': float(ppo_strength),
                'agreement': float(agreement_factor),
                'model_scores': model_scores
            }
            
            status = "✅" if approved else "❌"
            logger.debug(f"{status} Resultado validación: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Error en validación AI: {str(e)}")
            return {'approved': False, 'reason': str(e), 'score': 0.0}

    def _calculate_order_quantity(self, signal) -> float:
        """
        Calcula la cantidad a operar para una señal basada en position sizing dinámico
        """
        try:
            symbol = getattr(signal, 'symbol', None)
            if not symbol:
                logger.error("❌ Señal sin símbolo")
                return 0.0
                
            # Si ya viene con cantidad, validarla contra límites
            existing_qty = getattr(signal, 'qty', None)
            if existing_qty and existing_qty > 0:
                # Usar límites de configuración
                if symbol == 'BTCUSDT' and existing_qty <= self.risk_limits.get('MAX_ORDER_SIZE_BTC', 0.01):
                    return self.risk_limits.get('MAX_ORDER_SIZE_BTC', 0.01)
                elif symbol == 'ETHUSDT' and existing_qty <= self.risk_limits.get('MAX_ORDER_SIZE_ETH', 0.1):
                    return self.risk_limits.get('MAX_ORDER_SIZE_ETH', 0.1)
                return existing_qty
                
            # Calcular cantidad base según el balance y el símbolo
            # Usar balance por defecto si no hay portfolio manager
            usdt_balance = 3000.0  # Default 3000 USDT
            
            # Usar un máximo del 10% del balance por operación
            max_usdt = usdt_balance * 0.10
            
            # Ajustar por la fuerza de la señal (0.5 a 1.5x)
            signal_strength = float(getattr(signal, 'strength', 0.5))
            position_size = max_usdt * (0.5 + signal_strength)
            
            # Convertir a cantidad según el símbolo
            if symbol == 'BTCUSDT':
                current_price = float(signal.features.get('close', 110000))  # Precio aproximado si no hay
                qty = position_size / current_price
                min_qty = self.risk_limits.get('MAX_ORDER_SIZE_BTC', 0.01)
                return max(min_qty, min(qty, min_qty * 10))  # Entre min_qty y 10x min_qty
            elif symbol == 'ETHUSDT':
                current_price = float(signal.features.get('close', 4000))  # Precio aproximado si no hay
                qty = position_size / current_price
                min_qty = self.risk_limits.get('MAX_ORDER_SIZE_ETH', 0.1)
                return max(min_qty, min(qty, min_qty * 10))  # Entre min_qty y 10x min_qty
            
            logger.warning(f"❌ Símbolo no soportado: {symbol}")
            return 0.0  # Símbolo no soportado
            
        except Exception as e:
            logger.error(f"❌ Error calculando cantidad para {symbol}: {str(e)}")
            return 0.0

    def _execute_paper_order(self, signal, quantity) -> Dict[str, Any]:
        """
        Ejecuta una orden en modo paper trading
        """
        try:
            order_id = str(uuid.uuid4())
            
            # Simular ejecución exitosa
            order_result = {
                'order_id': order_id,
                'symbol': signal.symbol,
                'side': signal.side,
                'quantity': quantity,
                'price': signal.price or 0.0,
                'status': 'filled',
                'reason': 'paper_trading_success',
                'ts': time.time(),
                'source': getattr(signal, 'strategy_id', 'L1')
            }
            
            logger.info(f"📝 Orden paper ejecutada: {signal.symbol} {signal.side} {quantity:.8f}")
            return order_result
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando orden paper: {e}")
            return {
                'order_id': str(uuid.uuid4()),
                'symbol': signal.symbol,
                'side': signal.side,
                'quantity': quantity,
                'price': 0.0,
                'status': 'rejected',
                'reason': f'paper_execution_error: {str(e)}',
                'ts': time.time(),
                'source': getattr(signal, 'strategy_id', 'L1')
            }

    async def _execute_real_order(self, signal, quantity) -> Dict[str, Any]:
        """
        Ejecuta una orden real en Binance
        """
        try:
            if not self.binance_client:
                logger.error("❌ Binance client no disponible")
                return {
                    'order_id': str(uuid.uuid4()),
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'quantity': quantity,
                    'price': 0.0,
                    'status': 'rejected',
                    'reason': 'no_binance_client',
                    'ts': time.time(),
                    'source': getattr(signal, 'strategy_id', 'L1')
                }
            
            # Aquí iría la lógica real de Binance
            # Por ahora, simular como paper trading
            return self._execute_paper_order(signal, quantity)
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando orden real: {e}")
            return {
                'order_id': str(uuid.uuid4()),
                'symbol': signal.symbol,
                'side': signal.side,
                'quantity': quantity,
                'price': 0.0,
                'status': 'rejected',
                'reason': f'real_execution_error: {str(e)}',
                'ts': time.time(),
                'source': getattr(signal, 'strategy_id', 'L1')
            }

    def _update_execution_stats(self, order_result: Dict[str, Any]):
        """
        Actualiza estadísticas de ejecución
        """
        try:
            symbol = order_result.get('symbol', 'unknown')
            status = order_result.get('status', 'unknown')
            
            if symbol not in self.execution_stats:
                self.execution_stats[symbol] = {
                    'total_orders': 0,
                    'filled_orders': 0,
                    'rejected_orders': 0,
                    'error_orders': 0
                }
            
            self.execution_stats[symbol]['total_orders'] += 1
            
            if status == 'filled':
                self.execution_stats[symbol]['filled_orders'] += 1
            elif status == 'rejected':
                self.execution_stats[symbol]['rejected_orders'] += 1
            else:
                self.execution_stats[symbol]['error_orders'] += 1
                
        except Exception as e:
            logger.error(f"❌ Error actualizando estadísticas: {e}")

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

            # Validación más estricta basada en confluencia de señales
            signal_conf = getattr(signal, 'confidence', 0.0)
            rsi = signal.technical_indicators.get('rsi', 50.0)
            macd = signal.technical_indicators.get('macd', 0.0)
            macd_signal = signal.technical_indicators.get('macd_signal', 0.0)
            
            # Verificar confluencia de señales
            is_oversold = rsi < 30
            is_overbought = rsi > 70
            macd_trend = macd - macd_signal
            
            # Solo reducir requisitos si hay confluencia
            if signal.side.lower() == 'buy' and is_oversold and macd_trend > -1:
                min_strength *= 0.8  # Reducción moderada
                logger.info(f"✅ Confluencia alcista en {symbol}: RSI={rsi:.1f}, MACD trend={macd_trend:.2f}")
            elif signal.side.lower() == 'sell' and is_overbought and macd_trend < 1:
                min_strength *= 0.8  # Reducción moderada
                logger.info(f"✅ Confluencia bajista en {symbol}: RSI={rsi:.1f}, MACD trend={macd_trend:.2f}")
            else:
                logger.debug(f"⚠️ Sin confluencia en {symbol}: RSI={rsi:.1f}, MACD trend={macd_trend:.2f}")

            if strength < min_strength:
                logger.debug(f"❌ Señal {symbol} rechazada por fuerza insuficiente: {strength:.3f} < {min_strength}")
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
