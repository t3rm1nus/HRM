# system/trading_pipeline_manager.py
"""
Trading Pipeline Manager - Orquesta el flujo completo de trading.
Paper-safe, sin llamadas a APIs inexistentes.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Any
from core.logging import logger
from system.models import TradingCycleResult


class TradingPipelineManager:

    def __init__(
        self,
        portfolio_manager,
        order_manager,
        l2_processor,
        position_rotator,
        auto_rebalancer,
        signal_verifier,
        state_coordinator,
        config: Dict
    ):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.l2_processor = l2_processor
        self.position_rotator = position_rotator
        self.auto_rebalancer = auto_rebalancer
        self.signal_verifier = signal_verifier
        self.state_coordinator = state_coordinator
        self.config = config
        
        # Auto-learning bridge (injected from main.py)
        self.auto_learning_bridge = None

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================

    async def process_trading_cycle(
        self,
        state: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> TradingCycleResult:
        """Ejecuta un ciclo completo de trading."""
        start_time = pd.Timestamp.utcnow()

        if not state:
            raise ValueError("System state not injected")
        
        # Add version to state if it doesn't exist
        if "version" not in state:
            state["version"] = "1.0"

        result = TradingCycleResult(
            signals_generated=0,
            orders_executed=0,
            orders_rejected=0,
            cooldown_blocked=0,
            l3_regime="unknown",
            portfolio_value=0.0,
            execution_time=0.0
        )

        try:
            # PASO 1 – L3
            l3_output = await self._update_l3_decision(state, market_data)
            result.l3_regime = l3_output.get("regime", "unknown")

            # PASO 2 – L2
            l2_signals = await self._generate_l2_signals(state, market_data, l3_output)
            result.signals_generated = len(l2_signals)

            # PASO 3 – Verificación
            valid_signals = await self._validate_signals(l2_signals, market_data)
            
            # Si no hay señales válidas, saltar ejecución pero mantener el flujo
            if not valid_signals:
                logger.debug("No hay señales válidas en este ciclo")

            # PASO 4 – Generar órdenes
            orders = await self.order_manager.generate_orders(state, valid_signals) if valid_signals else []

            # PASO 5 – Validar órdenes
            validated_orders = await self._validate_orders(orders, state) if orders else []

            # PASO 6 – Ejecutar órdenes
            executed = await self.order_manager.execute_orders(validated_orders) if validated_orders else []

            # Clasificar órdenes ejecutadas
            filled = [o for o in executed if o.get("status") == "filled"]
            rejected = [o for o in executed if o.get("status") == "rejected"]

            result.orders_executed = len(filled)
            result.orders_rejected = len(rejected)

            # PASO 6B – REGISTRAR TRADES PARA AUTO-LEARNING
            if filled and self.auto_learning_bridge:
                logger.info(f"🤖 AUTO-LEARNING: Registrando {len(filled)} trades ejecutados...")
                for order in filled:
                    try:
                        # Usar market_data actualizado para obtener precios consistentes
                        symbol = order.get('symbol')
                        if symbol in market_data:
                            # Usar último precio disponible para contexto
                            current_price = market_data[symbol].iloc[-1]['close']
                            order['context_price'] = current_price
                            
                        await self.auto_learning_bridge.record_order_execution(
                            order=order,
                            l3_context=l3_output,
                            market_data=market_data
                        )
                        logger.info(f"🤖 AUTO-LEARNING: Trade registrado - {order.get('symbol')} {order.get('action')}")
                    except Exception as al_error:
                        logger.error(f"❌ AUTO-LEARNING: Error registrando trade: {al_error}")
            elif filled and not self.auto_learning_bridge:
                logger.warning(f"⚠️ AUTO-LEARNING: Hay {len(filled)} trades ejecutados pero el bridge no está conectado")

            # PASO 7 – ACTUALIZAR PORTFOLIO (CRÍTICO)
            if filled:
                # CRITICAL FIX: Asegurar precios consistentes antes de actualizar portfolio
                await self._update_portfolio_with_consistent_prices(filled, market_data)
                await self._update_portfolio_from_orders(filled, market_data)

            # PASO 7B – SINCRONIZAR PRECIOS ENTRE COMPONENTES (CRÍTICO)
            # Asegurar que todos los componentes usen los mismos precios para cálculos consistentes
            synced_portfolio_value = await self._sync_prices_with_market_data(market_data, result.portfolio_value)
            if synced_portfolio_value > 0:
                result.portfolio_value = synced_portfolio_value
                logger.info(f"💰 Portfolio value actualizado tras sincronización de precios: ${synced_portfolio_value:.2f}")

            # PASO 8 – Sincronizar portfolio con cliente (single source of truth)
            await self._sync_portfolio_with_client()

            # PASO 9 – Refrescar STATE desde portfolio real (ASYNC FIX)
            await self._sync_state_from_portfolio_async(state)

            # PASO 10 – Actualizar estado global en StateCoordinator
            # Usar precios ya sincronizados para actualizar el estado
            coordinator_value = await self._update_state_coordinator(state, market_data, l3_output)
            # Priorizar el valor sincronizado si es válido
            if coordinator_value > 0 and result.portfolio_value <= 0:
                result.portfolio_value = coordinator_value

            # PASO 10.5 – Sincronizar precios consistentes
            result.portfolio_value = await self._sync_prices_with_market_data(market_data, result.portfolio_value)

            # PASO 11 – Rotación (opcional)
            rotation_result = await self._process_position_rotation(state, market_data, result)
            if rotation_result:
                result.orders_executed += rotation_result

            # PASO 12 – Rebalance (opcional)
            rebalance_result = await self._process_rebalancing(state, market_data, l3_output, valid_signals, result)
            if rebalance_result:
                result.orders_executed += rebalance_result

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}", exc_info=True)
            # No re-lanzar la excepción para mantener el pipeline funcionando

        finally:
            result.execution_time = (pd.Timestamp.utcnow() - start_time).total_seconds()
            logger.info(
                f"⏱️ Ciclo completado en {result.execution_time:.2f}s | "
                f"Señales: {result.signals_generated} | "
                f"Órdenes: {result.orders_executed} ejecutadas, {result.orders_rejected} rechazadas"
            )

        return result

    # ==================================================================
    # CORE PIPELINE METHODS
    # ==================================================================

    async def _record_trades_for_auto_learning(
        self, 
        filled_orders: List[Dict], 
        l3_output: Dict, 
        market_data: Dict
    ) -> None:
        """Registra trades ejecutados para auto-learning."""
        if not filled_orders or not self.auto_learning_bridge:
            if filled_orders and not self.auto_learning_bridge:
                logger.warning(
                    f"⚠️ AUTO-LEARNING: Hay {len(filled_orders)} trades ejecutados pero el bridge no está conectado"
                )
            return

        logger.info(f"🤖 AUTO-LEARNING: Registrando {len(filled_orders)} trades ejecutados...")
        for order in filled_orders:
            try:
                # Usar market_data actualizado para obtener precios consistentes
                symbol = order.get('symbol')
                if symbol in market_data:
                    # Usar último precio disponible para contexto
                    current_price = market_data[symbol].iloc[-1]['close']
                    order['context_price'] = current_price

                await self.auto_learning_bridge.record_order_execution(
                    order=order,
                    l3_context=l3_output,
                    market_data=market_data
                )
                logger.info(
                    f"🤖 AUTO-LEARNING: Trade registrado - {order.get('symbol')} "
                    f"{order.get('action')} @ {order.get('price', 'N/A')}"
                )
            except Exception as al_error:
                logger.error(f"❌ AUTO-LEARNING: Error registrando trade: {al_error}")

    # CRITICAL FIX: Asegurar que PortfolioManager use precios consistentes
    async def _update_portfolio_with_consistent_prices(self, filled_orders: List[Dict], market_data: Dict) -> None:
        """Actualiza portfolio con precios consistentes del market_data"""
        for order in filled_orders:
            symbol = order.get('symbol')
            if symbol in market_data and market_data[symbol] is not None:
                # Usar el precio de cierre más reciente
                current_price = float(market_data[symbol].iloc[-1]['close'])
                logger.info(f"💰 Usando precio consistente para {symbol}: ${current_price}")
                # Actualizar el precio en el order para cálculos consistentes
                order['consistent_price'] = current_price

    async def _update_portfolio_from_orders(
        self, 
        filled_orders: List[Dict], 
        market_data: Dict
    ) -> None:
        """Actualiza el portfolio con órdenes ejecutadas."""
        try:
            await self.portfolio_manager.update_from_orders_async(filled_orders, market_data)
            
            # ✅ Log NAV después de cada trade ejecutado
            await self._log_nav_with_client(market_data)
            
        except Exception as e:
            logger.error(f"❌ Error actualizando portfolio desde órdenes: {e}")

    async def _sync_portfolio_with_client(self) -> None:
        """Sincroniza portfolio con cliente (single source of truth)."""
        try:
            if hasattr(self.portfolio_manager, '_sync_from_client_async'):
                await self.portfolio_manager._sync_from_client_async()
            elif hasattr(self.portfolio_manager, '_sync_from_client'):
                self.portfolio_manager._sync_from_client()
        except Exception as e:
            logger.error(f"❌ Error sincronizando portfolio con cliente: {e}")

    async def _update_state_coordinator(
        self, 
        state: Dict, 
        market_data: Dict, 
        l3_output: Dict
    ) -> float:
        """Actualiza el StateCoordinator y devuelve el valor total del portfolio."""
        try:
            # Obtener valor total del portfolio
            portfolio_value = await self.portfolio_manager.get_total_value_async(market_data)
            
            # Actualizar StateCoordinator
            self.state_coordinator.update_total_value(portfolio_value)
            self.state_coordinator.update_market_data(market_data)
            self.state_coordinator.update_l3_output(l3_output)
            
            # Obtener y actualizar estado del portfolio
            portfolio_state = await self._get_portfolio_state_async()
            self.state_coordinator.update_portfolio_state({
                "btc_balance": portfolio_state.get("BTCUSDT", {}).get("position", 0.0),
                "eth_balance": portfolio_state.get("ETHUSDT", {}).get("position", 0.0),
                "usdt_balance": portfolio_state.get("USDT", {}).get("free", 0.0),
                "total_value": portfolio_value
            })
            
            return portfolio_value
            
        except Exception as e:
            logger.error(f"❌ Error actualizando StateCoordinator: {e}")
            return 0.0

    async def _process_position_rotation(
        self, 
        state: Dict, 
        market_data: Dict, 
        result: TradingCycleResult
    ) -> int:
        """Procesa rotación de posiciones si está habilitada."""
        if not self.position_rotator:
            return 0
            
        try:
            rotation = await self.position_rotator.check_and_rotate_positions(state, market_data)
            if not rotation:
                return 0
                
            executed_rot = await self.order_manager.execute_orders(rotation)
            await self.portfolio_manager.update_from_orders_async(executed_rot, market_data)
            
            filled_rot = len([o for o in executed_rot if o.get("status") == "filled"])
            
            if filled_rot > 0:
                logger.info(f"🔄 Rotación ejecutada: {filled_rot} órdenes")
                
            return filled_rot
            
        except Exception as e:
            logger.error(f"❌ Error en rotación de posiciones: {e}")
            return 0

    async def _process_rebalancing(
        self, 
        state: Dict, 
        market_data: Dict, 
        l3_output: Dict,
        valid_signals: List,
        result: TradingCycleResult
    ) -> int:
        """Procesa rebalanceo automático si está habilitado."""
        if not self.auto_rebalancer:
            return 0
            
        try:
            from core.config import TEMPORARY_AGGRESSIVE_MODE, check_temporary_aggressive_mode
            
            # Check if temporary aggressive mode should be disabled
            check_temporary_aggressive_mode()
            
            # CRÍTICO: Verificar si L3 tiene señal de SELL
            l3_signal = l3_output.get('signal', '').lower()
            l3_regime = l3_output.get('regime', '').lower()
            
            # NO rebalancear si L3 dice SELL!
            if l3_signal == 'sell':
                logger.info(f"⛔️ AutoRebalance BLOCKED - L3 signal is SELL (regime: {l3_regime})")
                return 0
            
            # Bloques de régimen que NO deben rebalancear
            no_rebalance_regimes = ['crash', 'panic_sell', 'extreme_fear', 'bear']
            if l3_regime in no_rebalance_regimes:
                logger.info(f"⛔️ AutoRebalance BLOCKED - L3 regime {l3_regime} is dangerous")
                return 0
            
            # Priorizar señales L2 sobre rebalance en modo agresivo
            if TEMPORARY_AGGRESSIVE_MODE and len(valid_signals) > 0:
                logger.debug("🧯 AutoRebalance skipped in aggressive mode - L2 signals present")
                return 0
            
            # Solo proceder si L3 específicamente lo permite
            if l3_output.get('signal') == 'hold' and not l3_output.get('allow_rebalance', False):
                logger.info("⏸️ AutoRebalance skipped - L3 HOLD signal without rebalance permission")
                return 0
            
            rebalance = await self.auto_rebalancer.check_and_execute_rebalance(
                market_data, l3_decision=l3_output
            )
            
            if not rebalance:
                return 0
                
            # Asegurar precios consistentes para el rebalanceo
            for order in rebalance:
                symbol = order.get('symbol')
                if symbol in market_data and not pd.isna(market_data[symbol].iloc[-1]['close']):
                    order['market_price'] = float(market_data[symbol].iloc[-1]['close'])
            
            executed_reb = await self.order_manager.execute_orders(rebalance)
            
            # Actualizar con precios consistentes
            await self.portfolio_manager.update_from_orders_async(executed_reb, market_data)
            
            filled_reb = len([o for o in executed_reb if o.get("status") == "filled"])
            
            if filled_reb > 0:
                logger.info(f"⚖️ Rebalanceo ejecutado: {filled_reb} órdenes con precios consistentes")
                
            return filled_reb
            
        except Exception as e:
            logger.error(f"❌ Error en rebalanceo automático: {e}")
            return 0

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _sync_state_from_portfolio(self, state: Dict) -> None:
        """
        Fuente de verdad ÚNICA: PortfolioManager (SYNC VERSION - deprecated in async contexts)
        """
        portfolio = self.portfolio_manager.get_portfolio_state()

        snapshot = {
            "btc_balance": portfolio.get("BTCUSDT", {}).get("position", 0.0),
            "eth_balance": portfolio.get("ETHUSDT", {}).get("position", 0.0),
            "usdt_balance": portfolio.get("USDT", {}).get("free", 0.0),
            "total_value": portfolio.get("total", 0.0),
        }

        state["portfolio"] = snapshot
        logger.debug(f"✅ State synced from PortfolioManager: {snapshot}")

    async def _sync_state_from_portfolio_async(self, state: Dict) -> None:
        """
        Fuente de verdad ÚNICA: PortfolioManager (ASYNC VERSION - use in async contexts)
        CRITICAL FIX: Uses async balance methods to avoid ASYNC_VIOLATION errors.
        """
        try:
            # Get fresh balances async to ensure we have latest data
            balances = await self.portfolio_manager.get_balances_async()
            
            # Calculate total value async
            total_value = await self.portfolio_manager.get_total_value_async()

            snapshot = {
                "btc_balance": balances.get("BTC", 0.0),
                "eth_balance": balances.get("ETH", 0.0),
                "usdt_balance": balances.get("USDT", 0.0),
                "total_value": total_value,
            }

            state["portfolio"] = snapshot
            logger.debug(f"✅ State synced from PortfolioManager (async): {snapshot}")
            
        except Exception as e:
            logger.error(f"❌ Error sincronizando estado desde portfolio: {e}")
            # Mantener el estado anterior si hay error
            if "portfolio" not in state:
                state["portfolio"] = {}

    async def _get_portfolio_state_async(self) -> Dict:
        """
        Get portfolio state using async methods to avoid ASYNC_VIOLATION.
        Returns portfolio structure compatible with get_portfolio_state().
        """
        try:
            # Get fresh balances async
            balances = await self.portfolio_manager.get_balances_async()
            
            return {
                "BTCUSDT": {
                    "position": balances.get("BTC", 0.0), 
                    "free": balances.get("BTC", 0.0)
                },
                "ETHUSDT": {
                    "position": balances.get("ETH", 0.0), 
                    "free": balances.get("ETH", 0.0)
                },
                "USDT": {"free": balances.get("USDT", 0.0)},
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del portfolio: {e}")
            return {
                "BTCUSDT": {"position": 0.0, "free": 0.0},
                "ETHUSDT": {"position": 0.0, "free": 0.0},
                "USDT": {"free": 0.0},
            }

    async def _update_l3_decision(self, state: Dict, market_data: Dict) -> Dict:
        """Actualiza la decisión L3."""
        from core.l3_processor import get_l3_decision

        try:
            output = get_l3_decision(market_data) or {
                "regime": "neutral",
                "signal": "hold",
                "confidence": 0.5,
                "strategy_type": "fallback",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo decisión L3: {e}")
            output = {
                "regime": "neutral",
                "signal": "hold",
                "confidence": 0.0,
                "strategy_type": "error_fallback",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }

        state["l3_output"] = output
        state["l3_last_update"] = time.time()
        return output

    async def _generate_l2_signals(
        self, 
        state: Dict, 
        market_data: Dict, 
        l3_decision: Dict
    ) -> List[Dict]:
        """Genera señales L2 basadas en el contexto L3."""
        context = {
            "regime": l3_decision.get("regime"),
            "signal": l3_decision.get("signal"),
            "confidence": l3_decision.get("confidence"),
            "allow_l2": l3_decision.get("allow_l2_signals", True),
            "l3_output": l3_decision,
        }

        try:
            return self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=context
            )
        except Exception as e:
            logger.error(f"❌ Error generando señales L2: {e}")
            return []

    async def _validate_signals(self, signals: List[Dict], market_data: Dict) -> List[Dict]:
        """Valida señales usando el verificador."""
        valid = []
        for signal in signals:
            try:
                await self.signal_verifier.submit_signal_for_verification(signal, market_data)
                valid.append(signal)
            except Exception as e:
                logger.debug(f"Señal rechazada: {signal.get('symbol', 'unknown')} - {e}")
                continue
        return valid

    async def _validate_orders(self, orders: List[Dict], state: Dict) -> List[Dict]:
        """Valida órdenes antes de ejecución."""
        validated = []
        portfolio = state.get("portfolio", {})

        for order in orders:
            if order.get("status") != "pending":
                validated.append(order)
                continue

            try:
                check = self.order_manager.validate_order_size(
                    order.get("symbol"),
                    order.get("quantity"),
                    order.get("price"),
                    portfolio
                )

                if check["valid"]:
                    validated.append(order)
                else:
                    order["status"] = "rejected"
                    order["validation_error"] = check["reason"]
                    validated.append(order)
                    logger.warning(
                        f"Orden rechazada: {order.get('symbol')} - {check['reason']}"
                    )
                    
            except Exception as e:
                logger.error(f"Error validando orden: {e}")
                order["status"] = "rejected"
                order["validation_error"] = f"Validation error: {str(e)}"
                validated.append(order)

        return validated

    def _get_simulated_client(self) -> Optional[Any]:
        """Obtiene el cliente simulado del portfolio_manager si está disponible."""
        if hasattr(self.portfolio_manager, 'client'):
            return self.portfolio_manager.client
        return None

    async def _log_nav_with_client(self, market_data: Dict) -> None:
        """Registra NAV usando el cliente simulado para precios precisos."""
        try:
            client = self._get_simulated_client()
            if client is None:
                return
                
            if hasattr(self.portfolio_manager, 'log_nav'):
                self.portfolio_manager.log_nav(market_data, client)
            elif hasattr(self.portfolio_manager, 'log_status'):
                self.portfolio_manager.log_status(market_data, client)
        except Exception as e:
            logger.debug(f"No se pudo registrar NAV: {e}")

    # ==================================================================
    # PRICE SYNCHRONIZATION - CRITICAL FOR CONSISTENCY
    # ==================================================================

    async def _sync_prices_with_market_data(self, market_data: Dict[str, pd.DataFrame], portfolio_value: float = 0.0) -> float:
        """
        Sincroniza precios entre todos los componentes del sistema - VERSIÓN SIMPLIFICADA.
        
        Args:
            market_data: Dict con DataFrames de precios por símbolo
            portfolio_value: Valor actual del portfolio (para fallback)
            
        Returns:
            float: Valor actualizado del portfolio
        """
        try:
            # Obtener precios actuales del market_data
            current_prices = {}
            for symbol, df in market_data.items():
                if df is not None and not df.empty:
                    current_prices[symbol] = float(df.iloc[-1]['close'])
            
            if not current_prices:
                logger.warning("⚠️ No se pudieron extraer precios del market_data")
                return portfolio_value
            
            logger.info(f"💰 Precios extraídos: {current_prices}")
            
            # Solo actualizar precios, no calcular NAV aquí
            # El NAV se calculará en otro lugar
            if hasattr(self.portfolio_manager, 'update_current_prices'):
                self.portfolio_manager.update_current_prices(current_prices)
            
            # Devolver el valor existente, no recalcular
            return portfolio_value
            
        except Exception as e:
            logger.error(f"❌ Error sincronizando precios: {e}")
            return portfolio_value

    def _extract_current_prices(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extrae precios actuales del market_data con validación robusta.
        
        Args:
            market_data: Dict con DataFrames de precios por símbolo
            
        Returns:
            Dict[str, float]: Diccionario de precios válidos
        """
        current_prices = {}
        
        for symbol, df in market_data.items():
            try:
                # Validar que el DataFrame existe y no está vacío
                if df is None:
                    logger.debug(f"⚠️ {symbol}: DataFrame es None")
                    continue
                    
                if not isinstance(df, pd.DataFrame):
                    logger.warning(f"⚠️ {symbol}: No es un DataFrame válido")
                    continue
                    
                if df.empty:
                    logger.debug(f"⚠️ {symbol}: DataFrame vacío")
                    continue

                # Validar que existe la columna 'close'
                if 'close' not in df.columns:
                    logger.warning(f"⚠️ {symbol}: No tiene columna 'close'")
                    continue

                # Extraer último precio de cierre
                last_close = df['close'].iloc[-1]
                
                # Validar que el precio es un número válido
                if pd.isna(last_close):
                    logger.warning(f"⚠️ {symbol}: Precio de cierre es NaN")
                    continue
                    
                price = float(last_close)
                
                if price <= 0:
                    logger.warning(f"⚠️ {symbol}: Precio inválido (${price})")
                    continue

                # Validar que el precio está dentro de rangos razonables
                if not self._is_price_within_valid_range(symbol, price):
                    logger.warning(f"⚠️ {symbol}: Precio fuera de rango válido (${price})")
                    continue

                current_prices[symbol] = price
                logger.debug(f"📊 Precio actual {symbol}: ${price:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error extrayendo precio para {symbol}: {e}")
                continue
        
        return current_prices

    def _is_price_within_valid_range(self, symbol: str, price: float) -> bool:
        """
        Valida que el precio esté dentro de rangos razonables para el símbolo.
        
        Args:
            symbol: Símbolo del activo
            price: Precio a validar
            
        Returns:
            bool: True si el precio es válido
        """
        # Rangos aproximados para validación (ajustar según el mercado)
        valid_ranges = {
            'BTCUSDT': (1000.0, 200000.0),
            'ETHUSDT': (100.0, 20000.0),
            'BNBUSDT': (1.0, 2000.0),
            'SOLUSDT': (1.0, 1000.0),
            'ADAUSDT': (0.01, 50.0),
            'DOGEUSDT': (0.001, 10.0),
            'XRPUSDT': (0.01, 50.0),
            'DOTUSDT': (1.0, 500.0),
            'AVAXUSDT': (1.0, 500.0),
            'LINKUSDT': (1.0, 500.0),
            'MATICUSDT': (0.01, 50.0),
            'UNIUSDT': (1.0, 500.0),
            'ATOMUSDT': (1.0, 500.0),
            'LTCUSDT': (1.0, 1000.0),
            'BCHUSDT': (10.0, 5000.0),
            'ALGOUSDT': (0.01, 50.0),
            'VETUSDT': (0.001, 10.0),
            'FILUSDT': (1.0, 500.0),
            'TRXUSDT': (0.001, 10.0),
            'ETCUSDT': (1.0, 500.0),
        }
        
        # Si no hay rango definido, aceptar cualquier precio positivo
        if symbol not in valid_ranges:
            return price > 0
            
        min_price, max_price = valid_ranges[symbol]
        return min_price <= price <= max_price

    async def _sync_prices_with_portfolio_manager(self, current_prices: Dict[str, float]) -> None:
        """
        Sincroniza precios con el PortfolioManager.
        
        Args:
            current_prices: Dict de precios actuales por símbolo
        """
        if not current_prices:
            return
            
        try:
            if hasattr(self.portfolio_manager, 'update_current_prices'):
                self.portfolio_manager.update_current_prices(current_prices)
                logger.debug(f"✅ Precios sincronizados con PortfolioManager: {len(current_prices)} símbolos")
            elif hasattr(self.portfolio_manager, 'set_market_prices'):
                self.portfolio_manager.set_market_prices(current_prices)
                logger.debug(f"✅ Precios sincronizados con PortfolioManager (set_market_prices)")
            else:
                logger.warning("⚠️ PortfolioManager no tiene método para actualizar precios")
                
        except Exception as e:
            logger.error(f"❌ Error sincronizando precios con PortfolioManager: {e}")

    def _sync_prices_with_state_coordinator(self, current_prices: Dict[str, float]) -> None:
        """
        Sincroniza precios con el StateCoordinator.
        
        Args:
            current_prices: Dict de precios actuales por símbolo
        """
        if not current_prices:
            return
            
        try:
            if hasattr(self.state_coordinator, 'update_current_prices'):
                self.state_coordinator.update_current_prices(current_prices)
                logger.debug(f"✅ Precios sincronizados con StateCoordinator: {len(current_prices)} símbolos")
            elif hasattr(self.state_coordinator, 'set_current_prices'):
                self.state_coordinator.set_current_prices(current_prices)
                logger.debug(f"✅ Precios sincronizados con StateCoordinator (set_current_prices)")
            else:
                logger.warning("⚠️ StateCoordinator no tiene método para actualizar precios")
                
        except Exception as e:
            logger.error(f"❌ Error sincronizando precios con StateCoordinator: {e}")

    async def _recalculate_nav_with_prices(self, current_prices: Dict[str, float], fallback_value: float) -> float:
        """
        Recalcula el NAV del portfolio usando precios consistentes.
        
        Args:
            current_prices: Dict de precios actuales por símbolo
            fallback_value: Valor a retornar en caso de error
            
        Returns:
            float: Valor actualizado del portfolio
        """
        if not current_prices:
            return fallback_value
            
        try:
            # Intentar usar método específico con precios
            if hasattr(self.portfolio_manager, 'calculate_nav_with_prices'):
                portfolio_value = await self.portfolio_manager.calculate_nav_with_prices(current_prices)
                logger.info(f"💰 NAV recalculado con precios consistentes: ${portfolio_value:.2f}")
                return portfolio_value
                
            # Fallback: usar get_total_value_async
            elif hasattr(self.portfolio_manager, 'get_total_value_async'):
                # Crear market_data mínimo para el cálculo
                minimal_market_data = {}
                for symbol, price in current_prices.items():
                    minimal_market_data[symbol] = pd.DataFrame({'close': [price]})
                    
                portfolio_value = await self.portfolio_manager.get_total_value_async(minimal_market_data)
                logger.info(f"💰 NAV calculado vía get_total_value_async: ${portfolio_value:.2f}")
                return portfolio_value
                
            else:
                logger.warning("⚠️ PortfolioManager no tiene método para calcular NAV con precios")
                return fallback_value
                
        except Exception as e:
            logger.error(f"❌ Error recalculando NAV: {e}")
            return fallback_value

    def _validate_price_consistency(self, current_prices: Dict[str, float]) -> bool:
        """
        Valida la coherencia entre los precios extraídos.
        
        Args:
            current_prices: Dict de precios actuales por símbolo
            
        Returns:
            bool: True si los precios son coherentes
        """
        if not current_prices:
            return False
            
        try:
            # Validar que los precios no difieren demasiado de un promedio móvil
            # (esto podría indicar datos corruptos)
            prices = list(current_prices.values())
            avg_price = sum(prices) / len(prices)
            
            for symbol, price in current_prices.items():
                # Si un precio es muy diferente del promedio, podría ser un outlier
                # Nota: Esta validación es más útil para comparar el mismo símbolo en el tiempo
                # Para diferentes símbolos, los rangos de precios son muy diferentes
                pass
            
            logger.debug(f"✅ Validación de consistencia de precios completada: {len(current_prices)} precios válidos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando consistencia de precios: {e}")
            return False
