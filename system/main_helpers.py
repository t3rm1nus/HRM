# -*- coding: utf-8 -*-
"""Funciones auxiliares del runtime principal de HRM."""

import asyncio
from colorama import Fore, Style
from core.logging import logger


async def update_portfolio_nav_from_simulated_exchange(portfolio_manager, market_data):
    """
    Actualizar el NAV del PortfolioManager usando balances reales del SimulatedExchangeClient.
    
    Esta función se ejecuta después de cada ciclo de trading para asegurar que el NAV
    refleje los balances actuales del cliente simulado y los precios de mercado.
    """
    try:
        # Verificar que tenemos un cliente disponible
        if not hasattr(portfolio_manager, 'client') or not portfolio_manager.client:
            logger.debug("⚠️ No client available for NAV update")
            return
        
        client = portfolio_manager.client
        
        # Solo actualizar en modo paper/simulated
        is_paper_mode = False
        if hasattr(client, 'paper_mode'):
            is_paper_mode = client.paper_mode
        elif portfolio_manager.mode == "simulated":
            is_paper_mode = True
        
        if not is_paper_mode:
            return
        
        # Obtener balances del SimulatedExchangeClient
        if hasattr(client, 'get_balances'):
            import inspect
            if inspect.iscoroutinefunction(client.get_balances):
                balances = await client.get_balances()
            else:
                balances = client.get_balances()
        elif hasattr(client, 'get_account_balances'):
            import inspect
            if inspect.iscoroutinefunction(client.get_account_balances):
                balances = await client.get_account_balances()
            else:
                balances = client.get_account_balances()
        else:
            return
        
        # Extraer balances
        current_btc = balances.get("BTC", 0.0)
        current_eth = balances.get("ETH", 0.0)
        current_usdt = balances.get("USDT", 0.0)
        
        # Validar que no todos los balances sean cero
        if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
            logger.warning("⚠️ NAV Update: All balances are zero, skipping update")
            return
        
        # Obtener precios de mercado
        btc_price = 0.0
        eth_price = 0.0
        
        if hasattr(client, 'get_market_price'):
            btc_price = client.get_market_price("BTCUSDT")
            eth_price = client.get_market_price("ETHUSDT")
        elif market_data:
            # Fallback a market_data si no hay get_market_price
            btc_data = market_data.get("BTCUSDT", {})
            eth_data = market_data.get("ETHUSDT", {})
            
            if isinstance(btc_data, dict) and "close" in btc_data:
                btc_price = btc_data["close"]
            elif hasattr(btc_data, 'iloc'):
                btc_price = btc_data.iloc[-1]["close"]
                
            if isinstance(eth_data, dict) and "close" in eth_data:
                eth_price = eth_data["close"]
            elif hasattr(eth_data, 'iloc'):
                eth_price = eth_data.iloc[-1]["close"]
        
        # Calcular NAV total
        btc_value = current_btc * btc_price
        eth_value = current_eth * eth_price
        total_nav = current_usdt + btc_value + eth_value
        
        # Actualizar el portfolio del PortfolioManager
        portfolio_manager.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
        portfolio_manager.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
        portfolio_manager.portfolio["USDT"] = {"free": current_usdt}
        portfolio_manager.portfolio["total"] = total_nav
        
        # Actualizar peak_value si es necesario
        if total_nav > portfolio_manager.peak_value:
            portfolio_manager.peak_value = total_nav
            portfolio_manager.portfolio["peak_value"] = total_nav
        
        # Log de actualización (solo en debug para no saturar)
        logger.debug(f"📊 NAV Updated: BTC={current_btc:.6f} @ ${btc_price:.2f}, ETH={current_eth:.4f} @ ${eth_price:.2f}, USDT=${current_usdt:.2f}, TOTAL=${total_nav:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error updating NAV from SimulatedExchangeClient: {e}")

async def shutdown_cleanup(market_data_manager=None, exchange_client=None, realtime_loader=None):
    """
    Shutdown limpio global - cierra todas las conexiones y sesiones.
    
    Asegura que:
    - Todas las sesiones aiohttp se cierran
    - No quedan warnings asyncio al terminar
    - Los recursos se liberan correctamente
    """
    logger.info("🧹 Iniciando shutdown limpio global...")
    
    # Cerrar MarketDataManager
    if market_data_manager is not None:
        try:
            await market_data_manager.close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando MarketDataManager: {e}")
    
    # Cerrar RealTimeLoader si existe
    if realtime_loader is not None:
        try:
            await realtime_loader.close()
            logger.info("✅ RealTimeLoader cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando RealTimeLoader: {e}")
    
    # Cerrar ExchangeClient
    if exchange_client is not None:
        try:
            if hasattr(exchange_client, 'close'):
                await exchange_client.close()
            elif hasattr(exchange_client, 'close_connection'):
                await exchange_client.close_connection()
            logger.info("✅ ExchangeClient cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando ExchangeClient: {e}")
    
    # Cerrar todas las sesiones aiohttp pendientes
    try:
        import aiohttp
        import asyncio
        
        # Cerrar todas las sesiones aiohttp abiertas
        for task in asyncio.all_tasks():
            if 'aiohttp' in str(task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("✅ Sesiones aiohttp cerradas")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando sesiones aiohttp: {e}")
    
    # Forzar cierre del event loop para evitar warnings
    try:
        loop = asyncio.get_running_loop()
        # Cancelar todas las tareas pendientes
        pending_tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending_tasks:
            task.cancel()
        
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            logger.info(f"✅ {len(pending_tasks)} tareas pendientes canceladas")
    except Exception as e:
        logger.warning(f"⚠️ Error cancelando tareas pendientes: {e}")
    
    logger.info("🧹 Shutdown limpio completado")

async def log_portfolio_comparison(cycle_context, portfolio_manager, cycle_id, market_data):
        """
        Log portfolio comparison using cycle_context snapshot.
        
        🔑 CRITICAL FIX: Uses cycle_context snapshot instead of calling get_asset_balance_async()
        🔑 This ensures we use the SAME balances throughout the entire cycle
        
        CRITICAL FIX: Uses cycle_context to avoid multiple balance calls
        """
        try:
            # ================================================================
            # STEP 1: VERIFICAR CYCLE_CONTEXT DISPONIBLE
            # ================================================================
            if cycle_context is None:
                logger.warning("⚠️ cycle_context is None, cannot log portfolio comparison")
                return
            
            # ================================================================
            # STEP 2: EXTRAER BALANCES DEL SNAPSHOT DEL CICLO
            # ================================================================
            # 🔑 CRITICAL: Usar el snapshot del ciclo en lugar de llamar a get_balances_async()
            balances = cycle_context.get("balances", {})
            prices = cycle_context.get("prices", {})
            
            current_btc = balances.get("BTC", 0.0)
            current_eth = balances.get("ETH", 0.0)
            current_usdt = balances.get("USDT", 0.0)
            
            # ================================================================
            # STEP 3: VALIDACIÓN - PROTECCIÓN CONTRA PÉRDIDA DE ESTADO
            # ================================================================
            if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                logger.critical("🚨 ERROR: cycle_context shows all balances zero!")
                return
            
            # ================================================================
            # STEP 4: OBTENER PRECIOS DEL SNAPSHOT Y CALCULAR NAV
            # ================================================================
            btc_price = prices.get("BTCUSDT", 0.0)
            eth_price = prices.get("ETHUSDT", 0.0)
            
            # Calcular NAV
            btc_value = current_btc * btc_price
            eth_value = current_eth * eth_price
            current_total = current_usdt + btc_value + eth_value
            
            # ================================================================
            # STEP 5: LOG DEL ESTADO REAL DEL PORTFOLIO (DESDE SNAPSHOT)
            # ================================================================
            initial_capital = portfolio_manager.initial_balance if hasattr(portfolio_manager, 'initial_balance') else 500.0
            color = Fore.GREEN if current_total >= initial_capital else Fore.RED
            
            logger.info("=" * 70)
            logger.info(f"{color}💰 PORTFOLIO COMPARISON - Cycle {cycle_id} (POST-EXECUTION SNAPSHOT){Style.RESET_ALL}")
            logger.info(f"   BTC:  {current_btc:.6f} @ ${btc_price:.2f} = ${btc_value:.2f}")
            logger.info(f"   ETH:  {current_eth:.6f} @ ${eth_price:.2f} = ${eth_value:.2f}")
            logger.info(f"   USDT: ${current_usdt:.2f}")
            logger.info(f"   {'─' * 50}")
            logger.info(f"{color}   TOTAL NAV: ${current_total:.2f} (Initial: ${initial_capital:.2f}){Style.RESET_ALL}")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.opt(exception=True).error(f"❌ Error in portfolio comparison: {e}")
