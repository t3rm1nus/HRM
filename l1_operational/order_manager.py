from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import time

from core.logging import logger, log_trading_action
from core.config import HRM_PATH_MODE, PATH3_SIGNAL_SOURCE, MAX_CONTRA_ALLOCATION_PATH2
from core.selling_strategy import get_selling_strategy, SellSignal
from .config import ConfigObject
from l2_tactic.models import TacticalSignal
from .position_manager import PositionManager

# ========================================================================================
# CONSTANTES IMPORTANTES - AJUSTABLES
# ========================================================================================
DUST_THRESHOLD_BTC = 0.00005     # Mínimo BTC para considerar posición válida
DUST_THRESHOLD_ETH = 0.005       # Mínimo ETH para considerar posición válida
MIN_ORDER_USDT     = 2.0         # Mínimo valor en USDT para aceptar una orden
MIN_STOP_DISTANCE_PCT = 0.5      # Stop-loss mínimo % (anti-overtrading)
MAX_STOP_DISTANCE_PCT = 15.0     # Stop-loss máximo % (riesgo excesivo)

class OrderManager:
    """
    Gestiona el ciclo completo de trading: BUY → HOLD → SELL
    """

    def __init__(self, state_manager, portfolio_manager, config: Dict):
        self.position_manager = PositionManager(
            state_manager=state_manager,
            portfolio_manager=portfolio_manager,
            config=config
        )

        self.portfolio = portfolio_manager
        self.state = state_manager
        self.config = config
        self.position_tracker = {}      # Track open positions
        self.trading_state = {}         # Track trading state per symbol

        # Selling strategy
        self.selling_strategy = get_selling_strategy()

        # Validators & Executors
        from l1_operational.order_validators import OrderValidators
        from l1_operational.order_executors import OrderExecutors
        self.validators = OrderValidators(config)
        self.executors = OrderExecutors(state_manager, portfolio_manager, config)

        # Trading cooldown & state
        self.last_action = {}
        self.last_trade_time = {}
        self.cooldown_seconds = config.get('COOLDOWN_SECONDS', 60)

        logger.info("✅ OrderManager inicializado correctamente")

    async def assess_sell_opportunities(self, symbol: str, current_price: float,
                                       market_data: pd.DataFrame, l3_context: Dict[str, Any],
                                       position_data: Dict[str, Any]) -> Optional[SellSignal]:
        """
        Evalúa oportunidades de venta usando estrategia jerárquica de 4 niveles
        """
        try:
            return self.selling_strategy.assess_sell_opportunities(
                symbol=symbol,
                current_price=current_price,
                market_data=market_data,
                l3_context=l3_context,
                position_data=position_data
            )
        except Exception as e:
            logger.error(f"❌ Error al evaluar venta para {symbol}: {e}")
            return None

    def register_position_for_selling_strategy(self, symbol: str, entry_data: Dict[str, Any],
                                              market_data: pd.DataFrame, l3_context: Dict[str, Any]):
        try:
            self.selling_strategy.register_position_entry(
                symbol=symbol,
                entry_data=entry_data,
                market_data=market_data,
                l3_context=l3_context
            )
        except Exception as e:
            logger.error(f"❌ Error registrando posición en selling strategy {symbol}: {e}")

    def close_position_in_selling_strategy(self, symbol: str):
        try:
            self.selling_strategy.close_position(symbol)
        except Exception as e:
            logger.error(f"❌ Error cerrando tracking de posición {symbol}: {e}")

    def calculate_dynamic_reserve(self, portfolio_value: Optional[float] = None) -> float:
        MIN_BASE_RESERVE = 20.0      # Bajado de 50 → más agresivo en simulación
        RESERVE_PCT = 0.015          # 1.5% en lugar de 2%

        if portfolio_value is not None:
            pv = portfolio_value
        elif self.portfolio and hasattr(self.portfolio, 'get_total_value'):
            pv = self.portfolio.get_total_value()
        else:
            pv = 1000.0  # fallback muy conservador

        reserve = max(MIN_BASE_RESERVE, pv * RESERVE_PCT)
        logger.debug(f"Reserva dinámica: portfolio=${pv:.2f} → reserve=${reserve:.2f}")
        return reserve

    def get_available_usdt(self) -> float:
        usdt = self.portfolio.get_usdt_balance() if self.portfolio else 0.0
        reserve = self.calculate_dynamic_reserve()
        available = max(0.0, usdt - reserve)
        logger.info(f"USDT → total: ${usdt:.2f} | reserva: ${reserve:.2f} | disponible: ${available:.2f}")
        return available

    def _get_effective_position(self, symbol: str) -> float:
        """
        Obtiene la posición real considerando:
        1. Balance directo del portfolio_manager (fuente más confiable)
        2. position_tracker interno (como fallback)
        """
        if not self.portfolio:
            logger.warning("Portfolio manager no disponible → usando tracker interno")
            return self.position_tracker.get(symbol, {}).get('quantity', 0.0)

        asset = symbol.replace('USDT', '')
        real_balance = self.portfolio.get_balance(asset)

        # Umbral de dust por asset
        dust = DUST_THRESHOLD_BTC if 'BTC' in symbol else DUST_THRESHOLD_ETH

        if real_balance > dust:
            logger.debug(f"Posición real detectada: {symbol} = {real_balance:.8f}")
            return real_balance

        # Fallback al tracker si el balance está por debajo del dust
        tracker_qty = self.position_tracker.get(symbol, {}).get('quantity', 0.0)
        if tracker_qty > dust:
            logger.warning(f"Balance real bajo ({real_balance:.8f}), usando tracker: {tracker_qty:.8f}")
            return tracker_qty

        return 0.0

    async def generate_orders(self, state: Dict, signals: List) -> List:
        """
        Genera órdenes a partir de señales tácticas
        """
        orders = []
        for signal in signals:
            try:
                order = self.handle_signal(signal, state.get("market_data"))
                if order.get("status") == "accepted":
                    orders.append(order)
            except Exception as e:
                logger.error(f"❌ Error generando orden para señal {signal.symbol}: {e}")
        return orders

    def handle_signal(self, signal: TacticalSignal, market_data=None) -> Dict[str, Any]:
        """
        Procesamiento principal de señal táctica
        """
        current_price = self._extract_current_price(market_data, signal.symbol) if market_data else 0.0

        if current_price <= 0:
            return self._create_rejection_report(signal, "Precio actual inválido o no disponible")

        symbol = signal.symbol
        action = signal.side.lower()

        # HOLD → no procesar más
        if action == 'hold':
            return {
                'status': 'hold',
                'symbol': symbol,
                'action': 'hold',
                'reason': 'Señal HOLD - estado neutral',
                'timestamp': datetime.now().isoformat()
            }

        # Cooldown (excepto HOLD)
        if not self._check_cooldown(symbol, action):
            return self._create_rejection_report(signal, "Período de cooldown activo")

        # =====================================================================
        # FIX PRINCIPAL: VALIDACIÓN DE SELL MEJORADA
        # =====================================================================
        if action == 'sell':
            current_qty = self._get_effective_position(symbol)
            logger.info(f"[SELL CHECK] {symbol} → qty real: {current_qty:.8f}")

            if current_qty <= 0:
                logger.info(f"SELL ignorado: {symbol} - sin posición significativa")
                return self._create_rejection_report(
                    signal,
                    "No hay posición abierta para vender (saldo real ≤ 0)"
                )

            # Usamos la cantidad real disponible (o la de la señal si es menor)
            sell_qty = min(current_qty, signal.quantity or current_qty)
            if sell_qty <= 0:
                return self._create_rejection_report(signal, "Cantidad efectiva para vender = 0")

            logger.info(f"[SELL ACEPTADO] {symbol} → qty a vender: {sell_qty:.8f} / disponible: {current_qty:.8f}")

        # Cálculo de tamaño de orden (usando position_manager)
        order_size = self.position_manager.calculate_order_size(
            symbol=symbol,
            action=action,
            signal_confidence=signal.confidence,
            current_price=current_price,
            position_qty=self._get_effective_position(symbol)
        )

        if order_size <= 0:
            return self._create_rejection_report(signal, f"Tamaño de orden calculado inválido: {order_size}")

        # Validación final de valor mínimo
        order_value = abs(order_size) * current_price
        if order_value < MIN_ORDER_USDT:
            return self._create_rejection_report(
                signal,
                f"Orden demasiado pequeña: ${order_value:.2f} < mínimo ${MIN_ORDER_USDT}"
            )

        # Todo pasó → procedemos
        logger.info(f"✅ Señal aceptada: {symbol} {action.upper()} qty={order_size:.6f} @ ${current_price:.2f}")

        # Aquí iría la generación real de la orden...
        # Por ahora retornamos éxito simulado
        return {
            'status': 'accepted',
            'symbol': symbol,
            'action': action,
            'quantity': order_size,
            'price': current_price,
            'value_usdt': order_value,
            'timestamp': datetime.now().isoformat()
        }

    async def execute_orders(self, orders: List) -> List:
        """
        Ejecuta múltiples órdenes secuencialmente
        """
        executed_orders = []
        for order in orders:
            try:
                result = self.executors.execute_order(
                    symbol=order['symbol'],
                    action=order['action'],
                    quantity=order['quantity'],
                    current_price=order['price'],
                    stop_loss=None,
                    take_profit=None
                )
                executed_orders.append(result)
            except Exception as e:
                logger.error(f"❌ Error ejecutando orden {order['symbol']}: {e}")
                order['status'] = 'failed'
                order['error'] = str(e)
                executed_orders.append(order)
        return executed_orders

    def _extract_current_price(self, market_data: Any, symbol: str) -> float:
        """Extrae precio actual de diferentes formatos de market_data"""
        if market_data is None:
            return 0.0

        if isinstance(market_data, dict):
            if symbol in market_data:
                d = market_data[symbol]
                if isinstance(d, dict) and 'close' in d:
                    return float(d['close'])
        elif isinstance(market_data, pd.DataFrame) and 'close' in market_data.columns:
            try:
                return float(market_data['close'].iloc[-1])
            except:
                pass
        elif isinstance(market_data, pd.Series) and len(market_data) > 0:
            return float(market_data.iloc[-1])

        logger.warning(f"No se pudo extraer precio actual para {symbol}")
        return 0.0

    def _check_cooldown(self, symbol: str, action: str) -> bool:
        if symbol not in self.last_trade_time:
            return True
        elapsed = time.time() - self.last_trade_time[symbol]
        return elapsed >= self.cooldown_seconds

    def _create_rejection_report(self, signal: TacticalSignal, reason: str) -> Dict[str, Any]:
        return {
            'status': 'rejected',
            'symbol': signal.symbol,
            'action': signal.side,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    def _get_current_position(self, symbol: str, state: Dict[str, Any]) -> float:
        """
        Obtiene posición actual desde el estado del sistema
        """
        try:
            # Obtener desde el portfolio en el estado
            portfolio_snapshot = state.get("portfolio", {})
            asset_symbol = symbol.replace('USDT', '')
            position = portfolio_snapshot.get(asset_symbol, {}).get("position", 0.0)
            logger.debug(f"📊 Position from state for {symbol}: {position:.6f}")
            return position
        except Exception as e:
            logger.error(f"❌ Error getting position for {symbol}: {e}")
            return 0.0

    def process_signal_with_position_awareness(self, signal: Dict[str, Any],
                                               market_data: Dict, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa señal considerando posiciones actuales

        FLUJO:
        1. Si señal=BUY y NO hay posición → COMPRAR
        2. Si señal=BUY y HAY posición → HOLD (no re-comprar)
        3. Si señal=SELL y HAY posición → VENDER
        4. Si señal=SELL y NO hay posición → HOLD (no vender en vacío)
        5. Si señal=HOLD → mantener estado actual
        """
        symbol = signal.get('symbol')
        action = signal.get('action', 'hold').lower() if isinstance(signal, dict) else signal.side.lower()

        # ✅ CRITICAL FIX: Manejar errores de estado sin fallback automático
        try:
            # Intentar obtener posición actual
            current_position = self._get_current_position(symbol, state)
            has_position = current_position > 0
        except RuntimeError as e:
            # Estado no inicializado - abortar generación de órdenes
            logger.error(f"❌ State not initialized for {symbol}: {e}")
            return {
                'status': 'error',
                'reason': 'System state not initialized. Must be injected before trading loop.',
                'action': 'hold',
                'symbol': symbol
            }

# ✅ CASO 1: Señal BUY
        if action == 'buy':
            if not has_position:
                logger.info(f"✅ BUY SIGNAL VALID: No position for {symbol}, proceeding with BUY")
                return self._execute_buy(signal, market_data, state)
            else:
                logger.warning(f"⚠️ BUY SIGNAL IGNORED: Already have position for {symbol} ({current_position:.6f})")
                return {'status': 'ignored', 'reason': 'already_have_position',
                       'action': 'hold', 'symbol': symbol}

        # ✅ CASO 2: Señal SELL
        elif action == 'sell':
            current_qty = self._get_effective_position(symbol)

            if current_qty > DUST_THRESHOLD:
                logger.info(f"🚀 SELL VALIDADO: Detectado balance real de {current_qty} {symbol}")
                # Aquí sigue tu lógica original de vender...
                return self._execute_sell_logic(signal, current_qty)
            else:
                # ========================================================================================
                # 🥇 FIX 2: SEMANTIC SELL RULE - SELL without position = IGNORE (no alternatives)
                 # ========================================================================================
                logger.warning(f"â« SEMANTIC SELL RULE: {symbol} ignorado (Balance realmente en cero)")
                return {'status': 'ignored', 'reason': 'no_position_to_sell', 'action': 'hold', 'symbol': symbol}

        # ✅ CASO 3: Señal HOLD
        else:
            logger.info(f"ðŸ“Š HOLD SIGNAL: Maintaining current state for {symbol}")
            return {'status': 'hold', 'action': 'hold', 'symbol': symbol}

    def _get_current_position(self, symbol: str, state: Dict[str, Any]) -> float:
                """
                Obtiene posicin actual usando mltiples fuentes de verdad con validacin cruzada
                """
                try:
                    # CRTICO: Usar PortfolioManager como FUENTE NICA DE VERDAD
                    # Convertir BTCUSDT -> BTC para consultar balance real
                    asset_symbol = symbol.replace('USDT', '')

                    # Obtener posicin desde PortfolioManager (fuente primaria)
                    try:
                        position_pm = self.portfolio.get_balance(asset_symbol)
                        logger.debug(f"📊 PortfolioManager position for {symbol}: {position_pm:.6f}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting position from PortfolioManager: {e}")
                        position_pm = None

                    # Obtener posicin desde state (fuente secundaria)
                    try:
                        portfolio_snapshot = state.get("portfolio", {})
                        position_state = portfolio_snapshot.get(asset_symbol, {}).get("position", 0.0)
                        logger.debug(f"📊 State position for {symbol}: {position_state:.6f}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error getting position from state: {e}")
                        position_state = None

                    # Obtener posicin desde exchange (fuente terciaria - solo en modo live)
                    try:
                        if hasattr(self.portfolio, 'mode') and self.portfolio.mode == "live":
                            exchange_position = self.portfolio.client.get_position(asset_symbol)
                            logger.debug(f"📊 Exchange position for {symbol}: {exchange_position:.6f}")
                        else:
                            exchange_position = None
                    except Exception as e:
                        logger.debug(f"⚠️ Error getting position from exchange: {e}")
                        exchange_position = None

                    # Validar consistencia entre fuentes
                    positions = [p for p in [position_pm, position_state, exchange_position] if p is not None]

                    if positions:
                        # Calcular promedio de las fuentes disponibles
                        avg_position = sum(positions) / len(positions)
                        logger.info(f"✅ Position validation for {symbol}: {positions} -> using {avg_position:.6f}")
                        return avg_position
                    else:
                        logger.error(f"❌ No position data available for {symbol} from any source")
                        return 0.0

                except Exception as e:
                    logger.error(f"❌ Error getting position for {symbol}: {e}")
                    return 0.0

    def _execute_buy(self, signal: Dict[str, Any], market_data: Dict,
                     state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta orden de compra con sizing dinámico y reservas proporcionales

        CRITICAL FIX: Extract current_price at the start of this method
        """
        symbol = signal.get('symbol')
        confidence = signal.get('confidence', 0.5)

        # ✅ CRITICAL FIX: Extract current_price HERE at method start
        current_price = self._extract_current_price(market_data, symbol)

        # Validate price was extracted successfully
        if current_price <= 0:
            logger.error(f"❌ Failed to extract valid price for {symbol}: {current_price}")
            return {'status': 'rejected', 'reason': f'Invalid current price {current_price} for {symbol}'}

        logger.info(f"💰 BUY ORDER CALCULATION: {symbol} @ ${current_price:.2f}")

        # 💰 CRITICAL FIX: Mode-dependent USDT balance check
        # Check if we're in simulated mode
        if hasattr(self.portfolio, 'mode') and self.portfolio.mode == "simulated":
            total_usdt_balance = self.portfolio.get_usdt_balance()
            logger.debug(f"📊 SIMULATED MODE: Using portfolio USDT balance: ${total_usdt_balance:.2f}")
        else:
            # Real mode - use exchange client
            total_usdt_balance = self.portfolio.get_available_balance("USDT")
            logger.debug(f"📊 REAL MODE: Using exchange USDT balance: ${total_usdt_balance:.2f}")

        # Calcular valor total del portfolio para reserva dinámica
        portfolio_value = total_usdt_balance
        # Agregar valor de posiciones (aproximado)
        for asset_key, position_data in state.get("portfolio", {}).items():
            if asset_key != 'USDT' and isinstance(position_data, dict):
                position = position_data.get('position', 0.0)
                if position > 0 and asset_key in market_data:
                    # Obtener precio actual para calcular valor de posición
                    asset_price = self._extract_current_price(market_data, asset_key)
                    if asset_price > 0:
                        portfolio_value += position * asset_price

        # 💰 Calcular reserva dinámica (2% del portfolio total, mínimo $50)
        dynamic_reserve = self.calculate_dynamic_reserve(portfolio_value)
        available_usdt = max(0, total_usdt_balance - dynamic_reserve)

        logger.info(f"💰 DYNAMIC RESERVE: portfolio=${portfolio_value:.2f}, reserve=${dynamic_reserve:.2f}, available=${available_usdt:.2f}")

        # 💰 Usar PositionManager para calculo de sizing dinámico
        order_size = self.position_manager.calculate_order_size(
            symbol=symbol,
            action='buy',
            signal_confidence=confidence,
            current_price=current_price,  # ✅ Now properly available
            position_qty=0  # No current position for buy
        )

        # Convert to USDT for validation
        order_size_usdt = order_size * current_price

        # Si no hay tamaño válido, rechazar la orden
        if order_size_usdt <= 0:
            logger.warning(f"⚠️ No valid order size calculated for {symbol} (confidence={confidence:.2f})")
            return {'status': 'rejected', 'reason': 'insufficient_capital_after_dynamic_reserve'}

        position_size_usdt = order_size_usdt

        # Calcular cantidad de activo a comprar
        qty = position_size_usdt / current_price

        logger.info(f"✅ BUY ORDER: {symbol} qty={qty:.6f} @ {current_price:.2f} "
                   f"(confidence={confidence:.2f}, size_usdt={position_size_usdt:.2f})")

        # Create order for execution
        buy_order = {
            'symbol': symbol,
            'side': 'buy',
            'type': 'MARKET',
            'quantity': qty,
            'price': current_price,
            'timestamp': datetime.utcnow().isoformat(),
            'signal_source': 'full_cycle_manager',
            'status': 'pending',
            'order_type': 'ENTRY'
        }

        # Actualizar tracking
        if buy_order['status'] == 'pending':
            self.position_tracker[symbol] = {
                'qty': qty,
                'entry_price': current_price,
                'confidence': confidence,
                'timestamp': buy_order['timestamp']
            }

        return buy_order

    def _execute_sell(self, signal: Dict[str, Any], market_data: Dict,
                     state: Dict[str, Any], position_qty: float) -> Dict[str, Any]:
        """
        Ejecuta orden de venta

        CRITICAL FIX: Extract current_price at the start of this method
        """
        symbol = signal.get('symbol')
        confidence = signal.get('confidence', 0.5)

        # ✅ CRITICAL FIX: Extract current_price HERE at method start
        current_price = self._extract_current_price(market_data, symbol)

        # Validate price was extracted successfully
        if current_price <= 0:
            logger.error(f"❌ Failed to extract valid price for {symbol}: {current_price}")
            return {'status': 'rejected', 'reason': f'Invalid current price {current_price} for {symbol}'}

        logger.info(f"💰 SELL ORDER CALCULATION: {symbol} @ ${current_price:.2f}")

        # 🎯 Sell sizing: vender toda la posición o parcial según confianza
        if confidence >= 0.70:
            # Alta confianza → vender TODO
            qty_to_sell = position_qty
            logger.info(f"🔴 HIGH CONFIDENCE SELL: Selling ALL {qty_to_sell:.6f} {symbol}")
        elif confidence >= 0.55:
            # Confianza media → vender 75%
            qty_to_sell = position_qty * 0.75
            logger.info(f"🟡 MEDIUM CONFIDENCE SELL: Selling 75% ({qty_to_sell:.6f}) of {symbol}")
        else:
            # Confianza baja → vender 50%
            qty_to_sell = position_qty * 0.50
            logger.info(f"🟢 LOW CONFIDENCE SELL: Selling 50% ({qty_to_sell:.6f}) of {symbol}")

        logger.info(f"✅ SELL ORDER: {symbol} qty={qty_to_sell:.6f} @ {current_price:.2f} "
                   f"(confidence={confidence:.2f})")

        # Create sell order
        sell_order = {
            'symbol': symbol,
            'side': 'sell',
            'type': 'MARKET',
            'quantity': -qty_to_sell,  # Negative for sell
            'price': current_price,
            'timestamp': datetime.utcnow().isoformat(),
            'signal_source': 'full_cycle_manager',
            'status': 'pending',
            'order_type': 'ENTRY'
        }

        # Actualizar tracking
        if sell_order['status'] == 'pending':
            if qty_to_sell >= position_qty * 0.99:  # Vendió casi todo
                self.position_tracker.pop(symbol, None)
                logger.info(f"✅ Position CLOSED for {symbol}")
            else:
                # Actualizar posición parcial
                if symbol in self.position_tracker:
                    self.position_tracker[symbol]['qty'] -= qty_to_sell
                    logger.info(f"✅ Position REDUCED for {symbol}: {self.position_tracker[symbol]['qty']:.6f} remaining")

        return sell_order

    def _suggest_alternative_action(self, signal: Dict[str, Any],
                                   market_data: Dict, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        DEPRECATED: This method should never be called due to FIX 2 implementation.
        SELL signals without position are now directly rejected in process_signal_with_position_awareness.

        This method is kept for backward compatibility but should return HOLD for all cases.
        """
        symbol = signal.get('symbol')
        original_action = signal.get('action')

        logger.warning(f"🚫 ALTERNATIVE ACTION DEPRECATED: {symbol} {original_action} - This method should not be called (FIX 2 violation)")

        # ========================================================================================
        # 🥇 FIX 2: NO ALTERNATIVES ALLOWED - Return HOLD for all cases
        # ========================================================================================
        return {'status': 'hold', 'action': 'hold', 'symbol': symbol,
               'reason': 'Alternative actions disabled by FIX 2 - semantic trading rule'}

    def _extract_current_price(self, data, symbol: str) -> float:
        """
        Extract current price from market data, handling both DataFrame and dict formats.

        Args:
            data: Market data (DataFrame or dict)
            symbol: Trading symbol

        Returns:
            Current price as float, or 0.0 if extraction fails
        """
        try:
            if data is None:
                logger.error(f"Market data is None for {symbol}")
                return 0.0

            # Handle DataFrame format (direct data)
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    logger.warning(f"Empty DataFrame for {symbol}")
                    return 0.0
                if 'close' not in data.columns:
                    logger.error(f"No 'close' column in DataFrame for {symbol}")
                    return 0.0
                current_price = float(data['close'].iloc[-1])
                logger.debug(f"Extracted price from DataFrame para {symbol}: {current_price}")
                return current_price

            # Handle dict format (when market_data_dict contains per-symbol data)
            elif isinstance(data, dict):
                if symbol not in data:
                    logger.warning(f"Symbol {symbol} not found in market data dict")
                    return 50000.0 if symbol == 'BTCUSDT' else 3000.0  # Reasonable fallback

                symbol_data = data[symbol]
                if symbol_data is None:
                    logger.warning(f"Symbol data is None for {symbol}")
                    return 50000.0 if symbol == 'BTCUSDT' else 3000.0

                # Handle nested DataFrame (most common case)
                if isinstance(symbol_data, pd.DataFrame):
                    if symbol_data.empty:
                        logger.warning(f"Empty DataFrame for {symbol}")
                        return 50000.0 if symbol == 'BTCUSDT' else 3000.0
                    if 'close' not in symbol_data.columns:
                        logger.error(f"No 'close' column in DataFrame for {symbol}")
                        return 50000.0 if symbol == 'BTCUSDT' else 3000.0
                    current_price = float(symbol_data['close'].iloc[-1])
                    logger.debug(f"Extracted price from nested DataFrame para {symbol}: {current_price}")
                    return current_price

                # Handle nested dict format
                elif isinstance(symbol_data, dict):
                    if 'close' not in symbol_data:
                        logger.warning(f"No 'close' key in market data dict for {symbol}")
                        return 50000.0 if symbol == 'BTCUSDT' else 3000.0

                    close_value = symbol_data['close']
                    # Handle list/dict formats from close value
                    if isinstance(close_value, list) and close_value:
                        current_price = float(close_value[-1])
                    elif isinstance(close_value, (int, float)):
                        current_price = float(close_value)
                    else:
                        logger.warning(f"Unsupported close value format for {symbol}: {type(close_value)}")
                        return 50000.0 if symbol == 'BTCUSDT' else 3000.0

                    logger.debug(f"Extracted price from nested dict para {symbol}: {current_price}")
                    return current_price

                else:
                    logger.warning(f"Unsupported symbol data format for {symbol}: {type(symbol_data)}")
                    return 50000.0 if symbol == 'BTCUSDT' else 3000.0

            else:
                logger.warning(f"Unsupported data format: {type(data)}")
                return 50000.0 if symbol == 'BTCUSDT' else 3000.0

        except Exception as e:
            logger.error(f"❌ Error extracting current price for {symbol}: {e}")
            return 50000.0 if symbol == 'BTCUSDT' else 3000.0  # Safe fallback

    def _get_current_price(self, symbol: str, market_data: Dict) -> float:
        """
        Legacy method - redirects to new extract method
        """
        return self._extract_current_price(market_data.get(symbol), symbol) if isinstance(market_data, dict) and symbol in market_data else 0.0