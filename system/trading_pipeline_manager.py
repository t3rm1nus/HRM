# system/trading_pipeline_manager.py
"""
Trading Pipeline Manager - Orquesta el flujo completo de trading.

Este módulo coordina todos los pasos del ciclo de trading sin implementar
lógica de negocio nueva - solo llama a componentes existentes en orden.
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Any
from core.logging import logger
from system.models import TradingCycleResult


class TradingPipelineManager:
    """
    Orquesta el flujo completo de un ciclo de trading.
    
    Responsabilidades:
    1. Coordinar llamadas a componentes existentes
    2. Mantener el orden de ejecución del pipeline
    3. Manejar errores sin interrumpir el ciclo
    4. Retornar métricas del ciclo
    
    NO implementa lógica de trading - solo coordina.
    """
    
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
        """
        Inicializa el pipeline manager.
        
        Args:
            portfolio_manager: Gestor de portfolio
            order_manager: Gestor de órdenes
            l2_processor: Procesador L2 táctico
            position_rotator: Rotador de posiciones
            auto_rebalancer: Auto-rebalanceador
            signal_verifier: Verificador de señales
            state_coordinator: Coordinador de estado
            config: Configuración del sistema
        """
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.l2_processor = l2_processor
        self.position_rotator = position_rotator
        self.auto_rebalancer = auto_rebalancer
        self.signal_verifier = signal_verifier
        self.state_coordinator = state_coordinator
        self.config = config
    
    async def process_trading_cycle(
        self,
        state: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> TradingCycleResult:
        """
        Procesa un ciclo completo de trading.
        
        ORDEN DE EJECUCIÓN (CRÍTICO):
        1. Sync balances
        2. Monitor stop-loss
        3. Update L3 decision
        4. Generate L2 signals
        5. Validate signals
        6. Generate orders
        7. Validate orders
        8. Execute orders
        9. Update portfolio
        10. Check position rotation
        11. Check auto-rebalance
        
        Args:
            state: Estado global del sistema
            market_data: Datos de mercado actualizados
            
        Returns:
            TradingCycleResult con métricas del ciclo
            
        Raises:
            RuntimeError: Si el estado no está correctamente inyectado
        """
        start_time = pd.Timestamp.utcnow()
        
        # VALIDACIÓN DE INVARIANTE: Estado debe estar inyectado
        if not state or "version" not in state:
            logger.error("❌ Estado no inyectado - ciclo abortado")
            raise RuntimeError("System state not injected. Must be initialized before trading loop.")
        
        # ✅ CRITICAL: Validate l3_output before processing signals
        if not state.get("l3_output") or state["l3_output"].get("confidence", 0) < 0.1:
            logger.error("🔄 Recalculating L3 due to invalid state")
            # Forzar recálculo del L3
            state["l3_output"] = await self._recalculate_l3_output(market_data)
        
        # VALIDACIÓN DE INVARIANTE: Versión no debe cambiar durante el ciclo
        initial_version = state.get("version")
        
        # Inicializar resultado
        result = TradingCycleResult(
            signals_generated=0,
            orders_executed=0,
            orders_rejected=0,
            cooldown_blocked=0,
            l3_regime='unknown',
            portfolio_value=0.0,
            execution_time=0.0
        )
        
        try:
            # PASO 1: Sync balances
            logger.info("📊 PASO 1: Sincronizando balances...")
            sync_success = await self._sync_balances(state, market_data)
            
            # FIX CRÍTICO: Limpiar BLIND MODE sticky cuando los balances se sincronizan
            if sync_success:
                logger.info("✅ Balances sincronizados - limpiando BLIND MODE sticky")
                from core.state_manager import transition_system_state
                transition_system_state(
                    "NORMAL",
                    "Balances sincronizados exitosamente",
                    {"l3_balance_sync_failed": False, "l3_mode": "NORMAL"}
                )
            
            # PASO 2: Monitor stop-loss
            logger.info("🛡️ PASO 2: Monitoreando stop-loss...")
            executed_sl = await self._monitor_stop_losses(state, market_data)
            if executed_sl:
                result.orders_executed += len(executed_sl)
            
            # PASO 3: Actualizando decisión L3...
            logger.info("🧠 PASO 3: Actualizando decisión L3...")

            # CRITICAL FIX: Ensure l3_output exists before L3 processing
            if 'l3_output' not in state or not state['l3_output']:
                logger.warning("⚠️ L3 output missing, initializing default")
                state['l3_output'] = {
                    'regime': 'neutral',
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strategy_type': 'fallback',
                    'timestamp': pd.Timestamp.utcnow().isoformat(),
                    'asset_allocation': {
                        'BTC': 0.4,
                        'ETH': 0.3,
                        'CASH': 0.3
                    }
                }

            try:
                # Your existing L3 update code here
                l3_output = await self._update_l3_decision(state, market_data)
                
                # CRITICAL: Validate L3 output before using it
                if l3_output and isinstance(l3_output, dict):
                    state['l3_output'] = l3_output
                    logger.info(f"✅ L3 actualizado: {l3_output.get('regime', 'unknown')}")
                else:
                    logger.error("❌ L3 returned invalid output, using fallback")
                    # Keep existing l3_output (fallback from above)
                    
            except Exception as l3_error:
                logger.error(f"❌ Error actualizando L3: {l3_error}")
                # Don't crash - use existing l3_output as fallback
                logger.warning("⚠️ Using fallback L3 output")

            l3_decision = state['l3_output']
            result.l3_regime = l3_decision.get('regime', 'unknown')
            
            # PASO 4: Generate L2 signals
            logger.info("📡 PASO 4: Generando señales L2...")

            # CRITICAL: Validate L3 output exists
            if 'l3_output' not in state or not state['l3_output']:
                raise RuntimeError("L3 OUTPUT LOST - cannot generate L2 signals")

            l3_output = state['l3_output']

            # Validate required fields
            required_fields = ['regime', 'signal', 'confidence']
            missing_fields = [f for f in required_fields if f not in l3_output]

            if missing_fields:
                logger.error(f"❌ L3 output missing fields: {missing_fields}")
                # Use safe defaults
                l3_output['regime'] = l3_output.get('regime', 'neutral')
                l3_output['signal'] = l3_output.get('signal', 'hold')
                l3_output['confidence'] = l3_output.get('confidence', 0.5)
                state['l3_output'] = l3_output

            # Now safe to proceed with L2 signal generation
            l2_signals = await self._generate_l2_signals(state, market_data, l3_output)
            result.signals_generated = len(l2_signals)
            
            # PASO 5: Validate signals
            logger.info("✅ PASO 5: Validando señales...")
            valid_signals = await self._validate_signals(l2_signals, market_data)
            
            # PASO 6: Generate orders
            logger.info("📝 PASO 6: Generando órdenes...")
            orders = await self._generate_orders(state, valid_signals)
            
            # PASO 7: Validate orders
            logger.info("🔍 PASO 7: Validando órdenes...")
            validated_orders = await self._validate_orders(orders, state)
            
            # PASO 8: Execute orders
            logger.info("⚡ PASO 8: Ejecutando órdenes...")
            executed = await self._execute_orders(validated_orders)
            result.orders_executed += len([o for o in executed if o.get('status') == 'filled'])
            result.orders_rejected += len([o for o in executed if o.get('status') == 'rejected'])
            
            # PASO 9: Update portfolio
            logger.info("💰 PASO 9: Actualizando portfolio...")
            await self._update_portfolio(executed, market_data)
            result.portfolio_value = self.portfolio_manager.get_total_value(market_data)
            
            # PASO 10: Check position rotation
            logger.info("🔄 PASO 10: Verificando rotación de posiciones...")
            rotation = await self._check_position_rotation(state, market_data)
            if rotation:
                result.orders_executed += len(rotation)
            
            # PASO 11: Check auto-rebalance
            logger.info("⚖️ PASO 11: Verificando auto-rebalance...")
            rebalance = await self._check_auto_rebalance(state, market_data, l3_decision)
            if rebalance:
                result.orders_executed += len(rebalance)
            
            # VALIDACIÓN DE INVARIANTE: Versión debe permanecer constante
            final_version = state.get("version")
            if initial_version != final_version:
                logger.error(f"🚨 Invariante violado: Versión cambió de {initial_version} a {final_version}")
                raise RuntimeError(f"State version changed during cycle: {initial_version} -> {final_version}")
            
        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")
            # Retornar resultado parcial sin re-raise
        
        finally:
            result.execution_time = (pd.Timestamp.utcnow() - start_time).total_seconds()
            logger.info(f"⏱️ Ciclo completado en {result.execution_time:.2f}s")
        
        return result
    
    # ========================================================================
    # MÉTODOS PRIVADOS - Solo llaman a componentes existentes
    # ========================================================================
    
    async def _sync_balances(self, state: Dict, market_data: Dict) -> bool:
        """Sincroniza balances con exchange."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - sincronización de balances abortada")
                return False
            
            sync_success = await self.portfolio_manager.sync_with_exchange()
            if sync_success:
                logger.info("✅ Balances sincronizados")
                
                # FIX FINAL - REGLA DE ORO
                # El StateCoordinator NO calcula portfolio. Solo lo refleja.
                # Sincronizar portfolio REAL → STATE (obligatorio)
                real_portfolio = self.portfolio_manager.get_portfolio_state()
                
                # Actualizar state con balances reales
                self.state_coordinator.update_state({
                    "portfolio": {
                        "btc_balance": real_portfolio.get("BTCUSDT", {}).get("position", 0.0),
                        "eth_balance": real_portfolio.get("ETHUSDT", {}).get("position", 0.0),
                        "usdt_balance": real_portfolio.get("USDT", {}).get("free", 0.0),
                        "total_value": real_portfolio.get("total", 0.0),
                    }
                })
                
                # CRITICAL FIX: Update direct state keys too
                state['btc_balance'] = real_portfolio.get("BTCUSDT", {}).get("position", 0.0)
                state['eth_balance'] = real_portfolio.get("ETHUSDT", {}).get("position", 0.0)
                state['usdt_balance'] = real_portfolio.get("USDT", {}).get("free", 0.0)
                
                # DEBUG: Log what we're syncing
                logger.info(f"✅ Portfolio real sincronizado en StateCoordinator:")
                logger.info(f"   BTC: {state['btc_balance']:.8f}")
                logger.info(f"   ETH: {state['eth_balance']:.8f}")
                logger.info(f"   USDT: {state['usdt_balance']:.2f}")
                logger.info(f"   Total: {real_portfolio.get('total', 0.0):.2f}")
            else:
                logger.warning("⚠️ Sincronización de balances falló")
                
                # Aunque falle, usar último snapshot válido
                try:
                    real_portfolio = self.portfolio_manager.get_portfolio_state()
                    self.state_coordinator.update_state({
                        "portfolio": {
                            "btc_balance": real_portfolio.get("BTCUSDT", {}).get("position", 0.0),
                            "eth_balance": real_portfolio.get("ETHUSDT", {}).get("position", 0.0),
                            "usdt_balance": real_portfolio.get("USDT", {}).get("free", 0.0),
                            "total_value": real_portfolio.get("total", 0.0),
                        }
                    })
                    
                    # CRITICAL FIX: Update direct state keys too
                    state['btc_balance'] = real_portfolio.get("BTCUSDT", {}).get("position", 0.0)
                    state['eth_balance'] = real_portfolio.get("ETHUSDT", {}).get("position", 0.0)
                    state['usdt_balance'] = real_portfolio.get("USDT", {}).get("free", 0.0)
                    
                    logger.info("✅ Último snapshot de portfolio sincronizado en StateCoordinator")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo usar snapshot de portfolio: {e}")
            
            return sync_success
        except Exception as e:
            logger.error(f"❌ Error sincronizando balances: {e}")
            return False
    
    async def _monitor_stop_losses(self, state: Dict, market_data: Dict) -> List:
        """Monitorea y ejecuta stop-loss activos."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - monitoreo de stop-loss abortado")
                return []
            
            current_positions = {}
            for symbol in self.config.get("SYMBOLS", []):
                if symbol != "USDT":
                    current_positions[symbol] = self.portfolio_manager.get_balance(symbol)
            
            executed = await self.order_manager.monitor_and_execute_stop_losses_with_validation(
                market_data, current_positions
            )
            
            if executed:
                await self.portfolio_manager.update_from_orders_async(executed, market_data)
                logger.info(f"🛡️ Ejecutados {len(executed)} stop-loss")
            
            return executed or []
        except Exception as e:
            logger.error(f"❌ Error monitoreando stop-loss: {e}")
            return []
    
    async def _update_l3_decision(
        self, 
        state: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Actualiza decisión L3 con cache."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - actualización L3 abortada")
                return {
                    'regime': 'error',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'allow_l2_signals': False
                }
            
            from core.l3_processor import get_l3_decision, get_current_regime
            from comms.config import APAGAR_L3
            
            if APAGAR_L3:
                l3_output = {
                    'regime': 'disabled',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_type': 'l3_disabled',
                    'timestamp': pd.Timestamp.utcnow().isoformat()
                }
                # DIRECT UPDATE - Sin intermediarios frágiles
                state["l3_output"] = l3_output
                state["l3_last_update"] = time.time()
                return l3_output
            
            # Usar cache
            l3_cache = state.get("l3_decision_cache")
            if l3_cache and not self._should_refresh_l3(state, market_data):
                logger.debug("⏸️ Usando L3 cache")
                return l3_cache
            
            # Refresh L3 - DIRECT UPDATE
            l3_output = get_l3_decision(market_data)
            
            # Asegurar estructura completa
            if not l3_output:
                l3_output = {
                    'regime': 'neutral',
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strategy_type': 'fallback',
                    'timestamp': pd.Timestamp.utcnow().isoformat()
                }
            
            # DIRECT UPDATE - Sin intermediarios frágiles
            state["l3_output"] = l3_output
            state["l3_last_update"] = time.time()
            
            logger.info(f"✅ L3 actualizado: {l3_output.get('regime')} (confidence: {l3_output.get('confidence', 0):.2f})")
            return l3_output
            
        except Exception as e:
            logger.error(f"❌ Error actualizando L3: {e}")
            # DIRECT UPDATE - Sin intermediarios frágiles
            error_output = {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'error',
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }
            state["l3_output"] = error_output
            state["l3_last_update"] = time.time()
            return error_output
    
    def _should_refresh_l3(self, state: Dict, market_data: Dict) -> bool:
        """Determina si refrescar L3."""
        try:
            from core.l3_processor import get_current_regime
            
            l3_last_update = state.get("l3_last_update", 0)
            cache_age = (time.time() - l3_last_update) / 60 if l3_last_update else float('inf')
            
            if cache_age > 30:
                return True
            
            current_regime = get_current_regime(market_data)
            previous_regime = state.get("l3_previous_regime")
            
            if previous_regime != current_regime:
                state["l3_previous_regime"] = current_regime
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Error verificando refresh L3: {e}")
            return False
    
    async def _generate_l2_signals(
        self,
        state: Dict,
        market_data: Dict,
        l3_decision: Dict
    ) -> List:
        """Genera señales L2."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - generación L2 abortada")
                return []
            
            l3_context = {
                'regime': l3_decision.get('regime', 'unknown'),
                'subtype': l3_decision.get('subtype', 'unknown'),
                'confidence': l3_decision.get('confidence', 0.5),
                'signal': l3_decision.get('signal', 'hold'),
                'allow_l2': l3_decision.get('allow_l2_signals', True),
                'l3_output': l3_decision
            }
            
            signals = self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=l3_context
            )
            
            logger.info(f"📡 Generadas {len(signals)} señales L2")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generando señales L2: {e}")
            return []
    
    async def _validate_signals(self, signals: List, market_data: Dict) -> List:
        """Valida señales."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - validación de señales abortada")
                return []
            
            valid = []
            for signal in signals:
                try:
                    await self.signal_verifier.submit_signal_for_verification(
                        signal, market_data
                    )
                    valid.append(signal)
                except Exception as e:
                    logger.warning(f"⚠️ Señal rechazada: {e}")
            
            return valid
        except Exception as e:
            logger.error(f"❌ Error validando señales: {e}")
            return []
    
    async def _generate_orders(self, state: Dict, signals: List) -> List:
        """Genera órdenes."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - generación de órdenes abortada")
                return []
            
            orders = await self.order_manager.generate_orders(state, signals)
            logger.info(f"📝 Generadas {len(orders)} órdenes")
            return orders
        except Exception as e:
            logger.error(f"❌ Error generando órdenes: {e}")
            return []
    
    async def _validate_orders(self, orders: List, state: Dict) -> List:
        """Valida órdenes."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - validación de órdenes abortada")
                return []
            
            validated = []
            for order in orders:
                if order.get("status") != "pending":
                    validated.append(order)
                    continue
                
                validation = self.order_manager.validate_order_size(
                    order.get("symbol"),
                    order.get("quantity", 0.0),
                    order.get("price", 0.0),
                    state.get("portfolio", {})
                )
                
                if validation["valid"]:
                    validated.append(order)
                else:
                    order["status"] = "rejected"
                    order["validation_error"] = validation["reason"]
                    validated.append(order)
            
            return validated
        except Exception as e:
            logger.error(f"❌ Error validando órdenes: {e}")
            return []
    
    async def _execute_orders(self, orders: List) -> List:
        """Ejecuta órdenes."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - ejecución de órdenes abortada")
                return []
            
            executed = await self.order_manager.execute_orders(orders)
            logger.info(f"✅ Ejecutadas {len(executed)} órdenes")
            return executed
        except Exception as e:
            logger.error(f"❌ Error ejecutando órdenes: {e}")
            return []
    
    async def _update_portfolio(self, orders: List, market_data: Dict):
        """Actualiza portfolio."""
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - actualización de portfolio abortada")
                return
            
            await self.portfolio_manager.update_from_orders_async(orders, market_data)
            logger.debug("✅ Portfolio actualizado")
        except Exception as e:
            logger.error(f"❌ Error actualizando portfolio: {e}")
    
    async def _check_position_rotation(self, state: Dict, market_data: Dict) -> List:
        """Verifica rotación de posiciones."""
        if not self.position_rotator:
            return []
        
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - verificación de rotación abortada")
                return []
            
            rotation = await self.position_rotator.check_and_rotate_positions(
                state, market_data
            )
            
            if rotation:
                executed = await self.order_manager.execute_orders(rotation)
                await self.portfolio_manager.update_from_orders_async(executed, market_data)
                logger.info(f"🔄 Rotación: {len(executed)} órdenes")
                return executed
            
            return []
        except Exception as e:
            logger.error(f"❌ Error en rotación: {e}")
            return []
    
    async def _check_auto_rebalance(
        self,
        state: Dict,
        market_data: Dict,
        l3_decision: Dict
    ) -> List:
        """Verifica auto-rebalance."""
        if not self.auto_rebalancer:
            return []
        
        try:
            # VALIDACIÓN DE INVARIANTE: StateCoordinator debe estar inyectado
            if not self.state_coordinator:
                logger.error("❌ StateCoordinator no inyectado - verificación de auto-rebalance abortada")
                return []
            
            rebalance = await self.auto_rebalancer.check_and_execute_rebalance(
                market_data, l3_decision=l3_decision
            )
            
            if rebalance:
                executed = await self.order_manager.execute_orders(rebalance)
                await self.portfolio_manager.update_from_orders_async(executed, market_data)
                logger.info(f"⚖️ Rebalance: {len(executed)} órdenes")
                return executed
            
            return []
        except Exception as e:
            logger.error(f"❌ Error en rebalance: {e}")
            return []

    async def _recalculate_l3_output(self, market_data: Dict) -> Dict:
        """Recalcula L3 output cuando el estado es inválido."""
        try:
            from core.l3_processor import get_l3_decision
            from comms.config import APAGAR_L3
            
            if APAGAR_L3:
                l3_output = {
                    'regime': 'disabled',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_type': 'l3_disabled',
                    'timestamp': pd.Timestamp.utcnow().isoformat()
                }
                # DIRECT UPDATE - Sin intermediarios frágiles
                return l3_output
            
            l3_output = get_l3_decision(market_data)
            
            # Asegurar estructura completa
            if not l3_output:
                l3_output = {
                    'regime': 'neutral',
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strategy_type': 'fallback',
                    'timestamp': pd.Timestamp.utcnow().isoformat()
                }
            
            logger.info(f"🔄 L3 recalculado: {l3_output.get('regime')} (confidence: {l3_output.get('confidence', 0):.2f})")
            return l3_output
            
        except Exception as e:
            logger.error(f"❌ Error recalculando L3: {e}")
            # DIRECT UPDATE - Sin intermediarios frágiles
            error_output = {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'error',
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }
            return error_output
