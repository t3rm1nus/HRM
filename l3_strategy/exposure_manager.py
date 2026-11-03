import logging
import pandas as pd
from core.logging import logger
from typing import Dict, Optional, List, Any

# Importar AsyncContextDetector si está disponible
try:
    from core.async_balance_helper import AsyncContextDetector
except ImportError:
    # Fallback si no está disponible
    class AsyncContextDetector:
        @staticmethod
        def is_in_async_context():
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    return True
                except RuntimeError:
                    return False
            except:
                return False

class ExposureManager:
    """Gestión de exposición con sincronización real del portfolio."""

    def __init__(self, market_data_provider=None):
        """
        Initialize ExposureManager.

        Args:
            market_data_provider: Provider for market data access
        """
        self.market_data_provider = market_data_provider

    def _get_dynamic_allocations(self, regime: str, confidence: float = 0.5, total_nav: float = 0.0) -> dict:
        """
        Calcular allocations dinámicas según el régimen de mercado.
        
        Args:
            regime: Régimen de mercado (TRENDING_HIGH, RANGE, VOLATILE, etc.)
            confidence: Nivel de confianza del régimen (0.0 - 1.0)
            total_nav: Valor total del portfolio en USDT
            
        Returns:
            Dict con allocations: {'BTCUSDT': X.XX, 'ETHUSDT': X.XX, 'USDT': X.XX}
        """
        # Normalizar régimen
        regime_upper = regime.upper() if regime else 'DEFAULT'
        
        # Determinar subtipo de trending basado en confidence
        if 'TRENDING' in regime_upper:
            if confidence >= 0.65:
                regime_key = 'TRENDING_HIGH'
            elif confidence >= 0.50:
                regime_key = 'TRENDING_MEDIUM'
            else:
                regime_key = 'TRENDING_LOW'
        else:
            regime_key = regime_upper
        
        # Allocations por régimen
        allocations_map = {
            'TRENDING_HIGH': {'BTC': 0.45, 'ETH': 0.30, 'USDT': 0.25},
            'TRENDING_MEDIUM': {'BTC': 0.35, 'ETH': 0.25, 'USDT': 0.40},
            'TRENDING_LOW': {'BTC': 0.20, 'ETH': 0.15, 'USDT': 0.65},
            'RANGE': {'BTC': 0.20, 'ETH': 0.15, 'USDT': 0.65},
            'TIGHT_RANGE': {'BTC': 0.20, 'ETH': 0.15, 'USDT': 0.65},
            'VOLATILE': {'BTC': 0.15, 'ETH': 0.10, 'USDT': 0.75},
            'HIGH_VOLATILITY': {'BTC': 0.15, 'ETH': 0.10, 'USDT': 0.75},
            'BEAR': {'BTC': 0.05, 'ETH': 0.05, 'USDT': 0.90},
            'BEARISH': {'BTC': 0.05, 'ETH': 0.05, 'USDT': 0.90},
            'DEFAULT': {'BTC': 0.25, 'ETH': 0.20, 'USDT': 0.55}
        }
        
        # Obtener allocations base
        base_allocations = allocations_map.get(regime_key, allocations_map['DEFAULT'])
        
        # Aplicar capital protection: reducir allocations crypto a la mitad si NAV < $200
        if total_nav < 200.0:
            logger.warning(f"🛡️ CAPITAL PROTECTION: NAV=${total_nav:.2f} < $200, reduciendo exposición crypto 50%")
            allocations = {
                'BTC': base_allocations['BTC'] * 0.5,
                'ETH': base_allocations['ETH'] * 0.5,
                'USDT': base_allocations['USDT'] + (base_allocations['BTC'] * 0.5) + (base_allocations['ETH'] * 0.5)
            }
        else:
            allocations = base_allocations.copy()
        
        # Log de allocations dinámicas
        logger.info(f"📊 DYNAMIC ALLOCATION [{regime_key}]: BTC={allocations['BTC']*100:.0f}%, ETH={allocations['ETH']*100:.0f}%, USDT={allocations['USDT']*100:.0f}%")
        
        # Convertir a formato de símbolos
        return {
            'BTCUSDT': allocations['BTC'],
            'ETHUSDT': allocations['ETH'],
            'USDT': allocations['USDT']
        }

    def calculate_rebalancing(self, portfolio_manager, target_allocations: dict = None, market_data: dict = None, regime: str = None, confidence: float = 0.5) -> dict:
        """
        Calcular rebalanceo basado en POSICIONES REALES del portfolio.
        Unifica exposición entre portfolio_state_real y portfolio_state_simulated.
        
        ENHANCED: Ahora soporta allocations dinámicas según régimen de mercado.

        Args:
            portfolio_manager: Manager con posiciones actuales (IGNORADO - usamos singleton)
            target_allocations: {'BTCUSDT': 0.2, 'ETHUSDT': 0.15, 'USDT': 0.65} (opcional, si no se provee se calcula dinámicamente)
            market_data: Datos de mercado opcionales con precios actuales
            regime: Régimen de mercado para allocations dinámicas (ej: 'RANGE', 'TRENDING_HIGH')
            confidence: Nivel de confianza del régimen (0.0 - 1.0)

        Returns:
            Dict con decisiones de rebalanceo
        """
        try:
            # ========================================================================================
            # CRÍTICO: USAR SINGLETON DE PORTFOLIO_MANAGER COMO FUENTE ÚNICA DE VERDAD
            # FIX: El portfolio_manager que llega como parámetro NO tiene los balances correctos.
            # Usamos el singleton global que SÍ tiene los balances correctos.
            # ========================================================================================
            from core.portfolio_manager import PortfolioManager
            pm_singleton = PortfolioManager.get_instance()
            
            btc_position = 0.0
            eth_position = 0.0
            usdt_balance = 0.0
            
            # Leer del balance_cache del singleton (fuente síncrona de verdad)
            if hasattr(pm_singleton, '_balance_cache') and pm_singleton._balance_cache:
                balance_cache = pm_singleton._balance_cache
                btc_position = float(balance_cache.get('BTC', 0.0))
                eth_position = float(balance_cache.get('ETH', 0.0))
                usdt_balance = float(balance_cache.get('USDT', 0.0))
                
                # ✅ LOG EXPLÍCITO DEL FIX - Mostrar que los balances se leyeron del singleton
                logger.info(f"✅ [EXPOSURE_FIX] Singleton balances: BTC={btc_position:.6f}, ETH={eth_position:.6f}, USDT={usdt_balance:.2f}")
            else:
                logger.warning(f"⚠️ [EXPOSURE_FIX] _balance_cache no disponible en singleton, intentando portfolio...")
                # Fallback al portfolio del singleton
                if hasattr(pm_singleton, 'portfolio') and pm_singleton.portfolio:
                    portfolio = pm_singleton.portfolio
                    btc_data = portfolio.get('BTCUSDT', {})
                    btc_position = float(btc_data.get('position', btc_data.get('free', 0.0)))
                    
                    eth_data = portfolio.get('ETHUSDT', {})
                    eth_position = float(eth_data.get('position', eth_data.get('free', 0.0)))
                    
                    usdt_data = portfolio.get('USDT', {})
                    usdt_balance = float(usdt_data.get('free', 0.0))
                    
                    logger.info(f"✅ [EXPOSURE_FIX] Singleton portfolio fallback: BTC={btc_position:.6f}, ETH={eth_position:.6f}, USDT={usdt_balance:.2f}")
            
            # Si aún no tenemos balances, intentar con el portfolio_manager que llegó (último recurso)
            if btc_position == 0.0 and eth_position == 0.0:
                logger.warning(f"⚠️ [EXPOSURE_FIX] Singleton vacío, intentando portfolio_manager recibido...")
                if hasattr(portfolio_manager, 'portfolio') and portfolio_manager.portfolio:
                    portfolio = portfolio_manager.portfolio
                    btc_data = portfolio.get('BTCUSDT', {})
                    btc_position = float(btc_data.get('position', btc_data.get('free', 0.0)))
                    
                    eth_data = portfolio.get('ETHUSDT', {})
                    eth_position = float(eth_data.get('position', eth_data.get('free', 0.0)))
                    
                    usdt_data = portfolio.get('USDT', {})
                    usdt_balance = float(usdt_data.get('free', 0.0))
                    
                    logger.warning(f"⚠️ [EXPOSURE_FIX] Param portfolio fallback: BTC={btc_position}, ETH={eth_position}, USDT={usdt_balance}")

            # ✅ OBTENER PRECIOS ACTUALES PARA CALCULAR VALORES
            btc_price = self._get_current_price('BTCUSDT', market_data)
            eth_price = self._get_current_price('ETHUSDT', market_data)

            # ========================================================================================
            # CRÍTICO: UNIFICAR CÁLCULO DE EXPOSICIÓN - NINGÚN "0% ALLOCATION" CUANDO HAY POSICIONES
            # ========================================================================================
            # ✅ CALCULAR VALORES REALES EN USDT
            btc_value = btc_position * btc_price if btc_position > 0 else 0.0
            eth_value = eth_position * eth_price if eth_position > 0 else 0.0
            total_value = btc_value + eth_value + usdt_balance

            # ✅ CALCULAR ALLOCATIONS ACTUALES REALES
            # NUNCA mostrar 0% si hay posiciones reales
            current_btc_pct = (btc_value / total_value * 100) if total_value > 0 and btc_position > 0 else 0.0
            current_eth_pct = (eth_value / total_value * 100) if total_value > 0 and eth_position > 0 else 0.0
            current_usdt_pct = (usdt_balance / total_value * 100) if total_value > 0 else 0.0

            logger.info(f"📊 PORTFOLIO STATE REAL (unificado):")
            logger.info(f"   BTC: {btc_position:.6f} units (${btc_value:.2f}) = {current_btc_pct:.1f}%")
            logger.info(f"   ETH: {eth_position:.6f} units (${eth_value:.2f}) = {current_eth_pct:.1f}%")
            logger.info(f"   USDT: ${usdt_balance:.2f} = {current_usdt_pct:.1f}%")
            logger.info(f"   TOTAL VALUE: ${total_value:.2f}")

            # ========================================================================================
            # DYNAMIC ALLOCATIONS: Calcular allocations según régimen si no se proveen explícitamente
            # ========================================================================================
            if target_allocations is None and regime is not None:
                target_allocations = self._get_dynamic_allocations(regime, confidence, total_value)
            elif target_allocations is None:
                # Fallback a allocations por defecto
                target_allocations = {'BTCUSDT': 0.25, 'ETHUSDT': 0.20, 'USDT': 0.55}
                logger.info(f"📊 DEFAULT ALLOCATION (no regime provided): BTC=25%, ETH=20%, USDT=55%")

            # ✅ CALCULAR TARGETS EN USDT
            target_btc_value = total_value * target_allocations.get('BTCUSDT', 0.25)
            target_eth_value = total_value * target_allocations.get('ETHUSDT', 0.20)
            target_usdt_value = total_value * target_allocations.get('USDT', 0.55)

            logger.info(f"📊 TARGET ALLOCATIONS:")
            logger.info(f"   BTC: {target_allocations.get('BTCUSDT', 0.25)*100:.1f}% (${target_btc_value:.2f})")
            logger.info(f"   ETH: {target_allocations.get('ETHUSDT', 0.20)*100:.1f}% (${target_eth_value:.2f})")
            logger.info(f"   USDT: {target_allocations.get('USDT', 0.55)*100:.1f}% (${target_usdt_value:.2f})")

            # ========================================================================================
            # VALIDACIÓN: No generar compras si USDT disponible < target USDT
            # ========================================================================================
            usdt_available = usdt_balance
            usdt_needed_for_btc = max(0, target_btc_value - btc_value)
            usdt_needed_for_eth = max(0, target_eth_value - eth_value)
            total_usdt_needed = usdt_needed_for_btc + usdt_needed_for_eth
            
            # Si no hay suficiente USDT para las compras necesarias, bloquear compras
            if total_usdt_needed > usdt_available:
                logger.warning(f"🚫 INSUFFICIENT USDT: Available=${usdt_available:.2f}, Need=${total_usdt_needed:.2f}")
                logger.warning(f"🚫 BUY ORDERS BLOCKED: Cannot execute purchases without sufficient USDT")
                # Forzar HOLD en compras
                force_hold_buys = True
            else:
                force_hold_buys = False

            # ✅ CALCULAR AJUSTES NECESARIOS
            btc_adjustment_value = target_btc_value - btc_value
            eth_adjustment_value = target_eth_value - eth_value
            usdt_adjustment_value = target_usdt_value - usdt_balance

            # Determinar acciones con validación de USDT
            btc_action = self._determine_action(btc_adjustment_value)
            eth_action = self._determine_action(eth_adjustment_value)
            
            # Si force_hold_buys es True y la acción es compra, cambiar a hold
            if force_hold_buys and btc_action == 'buy':
                logger.warning(f"🚫 BTC BUY BLOCKED: Insufficient USDT for target allocation")
                btc_action = 'hold'
            if force_hold_buys and eth_action == 'buy':
                logger.warning(f"🚫 ETH BUY BLOCKED: Insufficient USDT for target allocation")
                eth_action = 'hold'

            # ✅ DETERMINAR ACCIONES
            rebalancing = {
                'BTCUSDT': {
                    'current_position': btc_position,
                    'current_value': btc_value,
                    'target_value': target_btc_value,
                    'adjustment_value': btc_adjustment_value,
                    'adjustment_units': btc_adjustment_value / btc_price if btc_price > 0 else 0,
                    'action': btc_action
                },
                'ETHUSDT': {
                    'current_position': eth_position,
                    'current_value': eth_value,
                    'target_value': target_eth_value,
                    'adjustment_value': eth_adjustment_value,
                    'adjustment_units': eth_adjustment_value / eth_price if eth_price > 0 else 0,
                    'action': eth_action
                },
                'USDT': {
                    'current_balance': usdt_balance,
                    'target_balance': target_usdt_value,
                    'adjustment': usdt_adjustment_value,
                    'action': 'rebalance_source' if usdt_adjustment_value > 0 else 'hold',
                    'available': usdt_available,
                    'needed': total_usdt_needed
                }
            }

            # ✅ LOG DETALLADO
            logger.info(f"📊 REBALANCING DECISIONS:")
            logger.info(f"   BTC: {rebalancing['BTCUSDT']['action']} {abs(btc_adjustment_value):.2f} USDT")
            logger.info(f"   ETH: {rebalancing['ETHUSDT']['action']} {abs(eth_adjustment_value):.2f} USDT")
            logger.info(f"   USDT: Need {usdt_adjustment_value:.2f} USDT")

            return rebalancing

        except Exception as e:
            logger.error(f"❌ Error calculating rebalancing: {e}", exc_info=True)
            return {}

    def _determine_action(self, adjustment_value: float, threshold: float = 50.0) -> str:
        """
        Determinar acción basada en ajuste necesario.

        Args:
            adjustment_value: Ajuste en USDT (+ = comprar, - = vender)
            threshold: Threshold mínimo para actuar ($50)

        Returns:
            'buy', 'sell', o 'hold'
        """
        if abs(adjustment_value) < threshold:
            return 'hold'
        elif adjustment_value > 0:
            return 'buy'
        else:
            return 'sell'

    def _get_current_price(self, symbol: str, market_data: dict = None) -> float:
        """
        Obtener precio actual del símbolo.
        
        Args:
            symbol: Símbolo a consultar (ej: 'BTCUSDT', 'ETHUSDT')
            market_data: Datos de mercado opcionales con precios actuales
            
        Returns:
            Precio actual del símbolo o 0.0 si no se puede obtener
        """
        try:
            # 1. PRIORIDAD: Extraer de market_data si está disponible
            if market_data and symbol in market_data:
                symbol_data = market_data[symbol]
                if isinstance(symbol_data, dict):
                    # Intentar obtener 'close' o 'price'
                    if 'close' in symbol_data:
                        price = float(symbol_data['close'])
                        logger.debug(f"📊 {symbol} price from market_data['close']: ${price:,.2f}")
                        return price
                    elif 'price' in symbol_data:
                        price = float(symbol_data['price'])
                        logger.debug(f"📊 {symbol} price from market_data['price']: ${price:,.2f}")
                        return price
                elif hasattr(symbol_data, 'iloc') and not symbol_data.empty:
                    # Es un DataFrame
                    try:
                        price = float(symbol_data['close'].iloc[-1])
                        logger.debug(f"📊 {symbol} price from DataFrame: ${price:,.2f}")
                        return price
                    except:
                        pass
            
            # 2. Usar market_data_provider si está disponible
            if self.market_data_provider:
                # Intentar diferentes métodos según la API del provider
                if hasattr(self.market_data_provider, 'get_market_prices'):
                    # Método async - necesitamos manejarlo cuidadosamente
                    import asyncio
                    try:
                        # Si estamos en async context, usar await
                        loop = asyncio.get_running_loop()
                        # No podemos await aquí, intentar método sync alternativo
                    except RuntimeError:
                        pass
                
                # Intentar método get_price (sync)
                if hasattr(self.market_data_provider, 'get_price'):
                    try:
                        price = self.market_data_provider.get_price(symbol)
                        if price and price > 0:
                            logger.debug(f"📊 {symbol} price from market_data_provider.get_price: ${price:,.2f}")
                            return float(price)
                    except Exception as e:
                        logger.debug(f"⚠️ get_price failed for {symbol}: {e}")
                
                # Intentar método get_latest_price (común en Binance adapters)
                if hasattr(self.market_data_provider, 'get_latest_price'):
                    try:
                        price = self.market_data_provider.get_latest_price(symbol)
                        if price and price > 0:
                            logger.debug(f"📊 {symbol} price from market_data_provider.get_latest_price: ${price:,.2f}")
                            return float(price)
                    except Exception as e:
                        logger.debug(f"⚠️ get_latest_price failed for {symbol}: {e}")
                
                # Intentar obtener de _cache si es MarketDataManager
                if hasattr(self.market_data_provider, '_cache'):
                    cache = self.market_data_provider._cache
                    if cache and hasattr(cache, 'data') and cache.data:
                        if symbol in cache.data:
                            symbol_data = cache.data[symbol]
                            if isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty and 'close' in symbol_data.columns:
                                price = float(symbol_data['close'].iloc[-1])
                                logger.debug(f"📊 {symbol} price from cache: ${price:,.2f}")
                                return price
            
            # 3. FALLBACK: Precios hardcodeados actualizados (solo como último recurso)
            # Estos valores deberían actualizarse periódicamente
            price_map = {
                'BTCUSDT': 67780.0,   # Precio actualizado BTC
                'ETHUSDT': 1972.78    # Precio actualizado ETH
            }
            fallback_price = price_map.get(symbol, 0.0)
            if fallback_price > 0:
                logger.warning(f"⚠️ {symbol}: Usando precio hardcodeado fallback: ${fallback_price:,.2f}")
            return fallback_price
            
        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            # Último fallback seguro
            if 'BTC' in symbol:
                return 67780.0
            elif 'ETH' in symbol:
                return 1972.78
            return 0.0

# Mantener compatibilidad con código existente
def gestionar_exposicion(universo, portfolio_state, market_data, regime, confidence: float = 0.5):
    """
    Legacy function para compatibilidad. Deprecated - usar ExposureManager en su lugar.
    
    ENHANCED: Ahora utiliza allocations dinámicas según el régimen de mercado.
    """
    logger.debug("⚠️ gestionar_exposicion está deprecated. Usar ExposureManager.calculate_rebalancing en su lugar.")

    # Crear una instancia temporal para compatibilidad
    exposure_manager = ExposureManager()

    # Convertir inputs antiguos al nuevo formato
    class LegacyPortfolioManager:
        def get_balance(self, asset):
            asset_key = f"{asset.lower()}_balance"
            return portfolio_state.get(asset_key, 0.0)

    fake_portfolio_manager = LegacyPortfolioManager()

    # Usar allocations dinámicas basadas en régimen (pasando None para que se calculen internamente)
    # El cálculo dinámico se realizará en calculate_rebalancing usando _get_dynamic_allocations
    return exposure_manager.calculate_rebalancing(
        fake_portfolio_manager, 
        target_allocations=None,  # Se calculará dinámicamente según régimen
        market_data=market_data,
        regime=regime,
        confidence=confidence
    )
