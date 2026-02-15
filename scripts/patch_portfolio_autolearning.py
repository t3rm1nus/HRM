#!/usr/bin/env python3
"""
PATCH: Fix PortfolioManager, SimulatedExchangeClient & Auto-Learning
Autor: HRM Fix Script
Objetivo: Reflejar trades en logs y activar autolearning correctamente.
"""

import asyncio
import inspect
from typing import Dict, Any, Optional
from datetime import datetime

from core.logging import logger

# =============================================================================
# 1️⃣ FORZAR INICIALIZACIÓN ASÍNCRONA DEL PORTFOLIOMANAGER
# =============================================================================

async def fix_portfolio_async():
    """
    Forzar inicialización asíncrona del PortfolioManager con SimulatedExchangeClient.
    Esto asegura que los balances se sincronicen correctamente desde el cliente simulado.
    """
    from core.portfolio_manager import PortfolioManager
    from l1_operational.simulated_exchange_client import SimulatedExchangeClient
    
    pm = PortfolioManager()
    sim_client = SimulatedExchangeClient.get_instance()
    
    if sim_client is None:
        logger.error("❌ SimulatedExchangeClient instance is None")
        return None
    
    # Inicializar asíncronamente
    await pm.initialize_async()
    
    # Sincronizar con el cliente simulado
    if hasattr(pm, '_sync_from_client_async'):
        await pm._sync_from_client_async()
    
    logger.info("✅ PortfolioManager initialized asynchronously with SimulatedExchangeClient")
    return pm


# =============================================================================
# 2️⃣ PATCH DE SIMULATEDEXCHANGECLIENT PARA REFLEJAR TRADES EN TIEMPO REAL
# =============================================================================

class SimulatedExchangeClientPatcher:
    """
    Parchea SimulatedExchangeClient para reflejar trades en tiempo real
    y actualizar NAV inmediatamente después de cada trade.
    """
    
    def __init__(self):
        self.original_execute_trade = None
        self.patched = False
    
    def patch(self):
        """Aplicar el parche al SimulatedExchangeClient"""
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        
        sim_client = SimulatedExchangeClient.get_instance()
        if sim_client is None:
            logger.error("❌ Cannot patch: SimulatedExchangeClient instance is None")
            return False
        
        # Guardar referencia al método original
        if hasattr(sim_client, 'execute_trade'):
            self.original_execute_trade = sim_client.execute_trade
        elif hasattr(sim_client, 'execute_order'):
            self.original_execute_trade = sim_client.execute_order
        else:
            logger.error("❌ Cannot patch: No execute_trade or execute_order method found")
            return False
        
        # Crear función parcheada
        async def patched_execute_trade(symbol: str, side: str, amount: float, price: float = None):
            """
            Ejecutar trade y actualizar NAV inmediatamente después.
            """
            # Ejecutar trade original
            if inspect.iscoroutinefunction(self.original_execute_trade):
                result = await self.original_execute_trade(symbol, side, amount, price)
            else:
                result = self.original_execute_trade(symbol, side, amount, price)
            
            # Actualizar NAV inmediatamente después de trade
            try:
                from core.portfolio_manager import PortfolioManager
                pm = PortfolioManager.get_instance() if hasattr(PortfolioManager, 'get_instance') else None
                
                if pm is None:
                    # Crear instancia temporal si no existe
                    pm = PortfolioManager()
                
                # Obtener precios de mercado actuales
                market_prices = {}
                for sym in ['BTCUSDT', 'ETHUSDT']:
                    if hasattr(sim_client, 'get_market_price'):
                        market_prices[sym] = sim_client.get_market_price(sym)
                    elif hasattr(sim_client, 'get_price'):
                        if inspect.iscoroutinefunction(sim_client.get_price):
                            market_prices[sym] = await sim_client.get_price(sym)
                        else:
                            market_prices[sym] = sim_client.get_price(sym)
                
                # Actualizar NAV si el método existe
                if hasattr(pm, 'update_nav'):
                    pm.update_nav(market_prices)
                elif hasattr(pm, 'calculate_nav'):
                    nav_data = pm.calculate_nav(market_prices)
                    pm.nav = nav_data.get('total_nav', 0.0)
                
                # Log actualizado con formato claro - CRITICAL FIX: Use async method
                nav_value = getattr(pm, 'nav', 0.0)
                if hasattr(pm, 'get_total_value_async'):
                    nav_value = await pm.get_total_value_async()
                elif hasattr(pm, 'get_total_value'):
                    # Fallback to sync only if async not available
                    nav_value = pm.get_total_value()
                
                logger.info(f"📈 Trade executed: {side.upper()} {amount} {symbol} at {price}, NAV: ${nav_value:.2f}")
                
                # También loguear el estado completo del portfolio
                if hasattr(pm, 'log_nav'):
                    pm.log_nav(market_prices, sim_client)
                
            except Exception as e:
                logger.error(f"❌ Error updating NAV after trade: {e}")
            
            return result
        
        # Aplicar parche
        if hasattr(sim_client, 'execute_trade'):
            sim_client.execute_trade = patched_execute_trade
        elif hasattr(sim_client, 'execute_order'):
            sim_client.execute_order = patched_execute_trade
        
        self.patched = True
        logger.info("✅ SimulatedExchangeClient patched: trades will reflect NAV immediately")
        return True
    
    def unpatch(self):
        """Restaurar método original"""
        if not self.patched or self.original_execute_trade is None:
            return
        
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        sim_client = SimulatedExchangeClient.get_instance()
        
        if sim_client and self.original_execute_trade:
            if hasattr(sim_client, 'execute_trade'):
                sim_client.execute_trade = self.original_execute_trade
            elif hasattr(sim_client, 'execute_order'):
                sim_client.execute_order = self.original_execute_trade
        
        self.patched = False
        logger.info("✅ SimulatedExchangeClient unpatched")


# =============================================================================
# 3️⃣ REINTEGRAR AUTO-LEARNING CON NAV ACTUALIZADO
# =============================================================================

class AutoLearningIntegrator:
    """
    Reintegra el sistema de Auto-Learning con NAV actualizado después de cada ciclo.
    """
    
    def __init__(self):
        self.al_system = None
        self.state_manager = None
        self.l2_processor = None
        self.trading_metrics = None
        self.initialized = False
    
    async def initialize(self, 
                        state_manager=None,
                        order_manager=None,
                        portfolio_manager=None,
                        l2_processor=None,
                        trading_metrics=None):
        """
        Inicializar la integración del Auto-Learning con todos los componentes.
        """
        from auto_learning_system import AutoLearningSystem
        from system.state_coordinator import StateCoordinator
        from l2_tactic.tactical_signal_processor import L2TacticProcessor
        from core.trading_metrics import TradingMetrics
        
        # Obtener o crear instancias
        self.al_system = AutoLearningSystem.get_instance() if hasattr(AutoLearningSystem, 'get_instance') else AutoLearningSystem()
        
        if state_manager is None:
            self.state_manager = StateCoordinator.get_instance() if hasattr(StateCoordinator, 'get_instance') else StateCoordinator()
        else:
            self.state_manager = state_manager
        
        if l2_processor is None:
            self.l2_processor = L2TacticProcessor.get_instance() if hasattr(L2TacticProcessor, 'get_instance') else L2TacticProcessor()
        else:
            self.l2_processor = l2_processor
        
        if trading_metrics is None:
            self.trading_metrics = TradingMetrics.get_instance() if hasattr(TradingMetrics, 'get_instance') else TradingMetrics()
        else:
            self.trading_metrics = trading_metrics
        
        # Integrar componentes
        try:
            self.al_system.integrate(
                state_manager=self.state_manager,
                order_manager=order_manager,  # ya está integrado
                portfolio_manager=portfolio_manager,
                l2_processor=self.l2_processor,
                trading_metrics=self.trading_metrics
            )
            self.initialized = True
            logger.info("✅ Auto-Learning integrated with NAV update capability")
            return True
        except Exception as e:
            logger.error(f"❌ Error integrating Auto-Learning: {e}")
            return False
    
    async def update_nav_after_trade(self, trade_data: Dict[str, Any]):
        """
        Actualizar NAV después de un trade y registrar para auto-learning.
        """
        if not self.initialized:
            logger.warning("⚠️ Auto-Learning not initialized, skipping NAV update")
            return False
        
        try:
            # Registrar el trade para auto-learning
            if hasattr(self.al_system, 'record_trade'):
                self.al_system.record_trade(trade_data)
            
            # Actualizar métricas de trading
            if self.trading_metrics and hasattr(self.trading_metrics, 'update_from_orders'):
                self.trading_metrics.update_from_orders([trade_data], trade_data.get('portfolio_value', 0))
            
            logger.debug(f"📊 Trade recorded for auto-learning: {trade_data.get('symbol')} {trade_data.get('side')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating NAV for auto-learning: {e}")
            return False


# =============================================================================
# FUNCIÓN PRINCIPAL DE APLICACIÓN DEL PARCHE
# =============================================================================

async def apply_patch():
    """
    Aplicar todos los parches:
    1. Forzar inicialización asíncrona del PortfolioManager
    2. Parchear SimulatedExchangeClient
    3. Reintegrar Auto-Learning
    """
    logger.info("=" * 70)
    logger.info("🩹 APLICANDO PARCHE: PortfolioManager, SimulatedExchangeClient & Auto-Learning")
    logger.info("=" * 70)
    
    results = {
        'portfolio_async': False,
        'sim_client_patch': False,
        'auto_learning': False
    }
    
    # 1. Inicializar PortfolioManager asíncronamente
    try:
        pm = await fix_portfolio_async()
        if pm is not None:
            results['portfolio_async'] = True
            logger.info("✅ Portfolio async initialization: SUCCESS")
        else:
            logger.warning("⚠️ Portfolio async initialization: returned None")
    except Exception as e:
        logger.error(f"❌ Portfolio async initialization failed: {e}")
    
    # 2. Parchear SimulatedExchangeClient
    try:
        patcher = SimulatedExchangeClientPatcher()
        patch_result = patcher.patch()
        results['sim_client_patch'] = patch_result
        if patch_result:
            logger.info("✅ SimulatedExchangeClient patch: SUCCESS")
        else:
            logger.warning("⚠️ SimulatedExchangeClient patch: FAILED")
    except Exception as e:
        logger.error(f"❌ SimulatedExchangeClient patch failed: {e}")
    
    # 3. Reintegrar Auto-Learning
    try:
        integrator = AutoLearningIntegrator()
        
        # Obtener componentes existentes si están disponibles
        from core.portfolio_manager import PortfolioManager
        from l1_operational.order_manager import OrderManager
        
        pm = PortfolioManager.get_instance() if hasattr(PortfolioManager, 'get_instance') else None
        om = OrderManager.get_instance() if hasattr(OrderManager, 'get_instance') else None
        
        al_result = await integrator.initialize(
            order_manager=om,
            portfolio_manager=pm
        )
        results['auto_learning'] = al_result
        if al_result:
            logger.info("✅ Auto-Learning integration: SUCCESS")
        else:
            logger.warning("⚠️ Auto-Learning integration: FAILED")
    except Exception as e:
        logger.error(f"❌ Auto-Learning integration failed: {e}")
    
    # Resumen
    logger.info("=" * 70)
    logger.info("📋 RESUMEN DEL PARCHE:")
    logger.info(f"   Portfolio Async Init: {'✅' if results['portfolio_async'] else '❌'}")
    logger.info(f"   SimClient Patch:      {'✅' if results['sim_client_patch'] else '❌'}")
    logger.info(f"   Auto-Learning:        {'✅' if results['auto_learning'] else '❌'}")
    
    if all(results.values()):
        logger.info("🎉 TODOS LOS PARCHES APLICADOS CORRECTAMENTE")
    else:
        logger.warning("⚠️ Algunos parches no se aplicaron correctamente")
    
    logger.info("=" * 70)
    
    return results


# Función síncrona para compatibilidad
def apply_patch_sync():
    """Versión síncrona de apply_patch para compatibilidad"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Si ya hay un loop corriendo, crear tarea
            return asyncio.create_task(apply_patch())
        else:
            return loop.run_until_complete(apply_patch())
    except RuntimeError:
        # No hay loop, crear uno nuevo
        return asyncio.run(apply_patch())


if __name__ == "__main__":
    # Ejecutar parche
    asyncio.run(apply_patch())
