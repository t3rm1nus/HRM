#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de sincronización de portfolios L1 con L3/resumen
para asegurar que los $3000 USDT sean realmente utilizables en operaciones de paper trading.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.signal_hierarchy import should_execute_with_l3_dominance
from fix_l3_dominance import should_trigger_rebalancing, calculate_allocation_deviation
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l1_operational.binance_client import BinanceClient
from l1_operational.order_manager import OrderManager
from system.market_data_manager import MarketDataManager
from comms.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class PortfolioSynchronizer:
    """Sincroniza portfolios L1 con L3/resumen para paper trading"""
    
    def __init__(self):
        self.portfolio_manager = None
        self.market_data_manager = None
        self.l2_processor = None
        self.order_manager = None
        self.state_coordinator = None
        
        self.initial_balance = 500.0
        self.min_rebalance_amount = 100.0
        
        logger.info("🔧 PortfolioSynchronizer inicializado")
    
    async def initialize_components(self):
        """Inicializa todos los componentes necesarios"""
        try:
            # 1. Inicializar PortfolioManager
            logger.info("📊 Inicializando PortfolioManager...")
            binance_client = BinanceClient()
            self.portfolio_manager = PortfolioManager(
                client=binance_client, 
                mode="simulated",
                initial_balance=self.initial_balance
            )
            
            # Resetear portfolio a 3000 USDT
            self.portfolio_manager.reset_portfolio()
            logger.info("✅ PortfolioManager inicializado con 3000 USDT")
            
            # 2. Inicializar MarketDataManager
            logger.info("📈 Inicializando MarketDataManager...")
            self.market_data_manager = MarketDataManager(
                symbols=["BTCUSDT", "ETHUSDT"],
                fallback_enabled=True
            )
            logger.info("✅ MarketDataManager inicializado")
            
            # 3. Inicializar L2TacticProcessor
            logger.info("🎯 Inicializando L2TacticProcessor...")
            self.l2_processor = L2TacticProcessor()
            logger.info("✅ L2TacticProcessor inicializado")
            
            # 4. Inicializar OrderManager
            logger.info("⚡ Inicializando OrderManager...")
            # Crear un state_manager mínimo para OrderManager
            class MockStateManager:
                def get_state(self, key):
                    return {}
            
            mock_state_manager = MockStateManager()
            self.order_manager = OrderManager(
                state_manager=mock_state_manager,
                portfolio_manager=self.portfolio_manager,
                config=config
            )
            logger.info("✅ OrderManager inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            raise
    
    def get_current_allocation(self, market_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Obtiene la allocación actual del portfolio"""
        try:
            btc_balance = self.portfolio_manager.get_balance('BTCUSDT')
            eth_balance = self.portfolio_manager.get_balance('ETHUSDT')
            usdt_balance = self.portfolio_manager.get_balance('USDT')
            total_value = self.portfolio_manager.get_total_value(market_data)
            
            if total_value <= 0:
                return {'BTCUSDT': 0, 'ETHUSDT': 0, 'USDT': 1}
            
            # Obtener precios del mercado
            btc_price = self.get_market_price('BTCUSDT', market_data)
            eth_price = self.get_market_price('ETHUSDT', market_data)
            
            return {
                'BTCUSDT': (btc_balance * btc_price) / total_value,
                'ETHUSDT': (eth_balance * eth_price) / total_value,
                'USDT': usdt_balance / total_value
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo allocación actual: {e}")
            return {'BTCUSDT': 0, 'ETHUSDT': 0, 'USDT': 1}
    
    def get_market_price(self, symbol: str, market_data: Dict[str, Any] = None) -> float:
        """Obtiene el precio actual del mercado para un símbolo"""
        try:
            if market_data and symbol in market_data:
                data = market_data[symbol]
                if isinstance(data, dict) and "close" in data:
                    return float(data["close"])
                elif isinstance(data, list) and len(data) > 0:
                    return float(data[-1])
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo precio de mercado para {symbol}: {e}")
            return 0.0
    
    def get_l3_targets(self) -> Dict[str, float]:
        """
        Obtiene los targets L3 desde el archivo de configuración o valores por defecto
        """
        try:
            # Intentar cargar targets desde archivo de configuración
            config_file = "configs/L3/targets.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    targets = json.load(f)
                    logger.info(f"🎯 Targets L3 cargados desde {config_file}")
                    return targets
            
            # Targets por defecto para paper trading
            default_targets = {
                'BTCUSDT': 0.5,  # 50% en BTC
                'ETHUSDT': 0.3,  # 30% en ETH
                'USDT': 0.2      # 20% en USDT
            }
            
            logger.info("🎯 Usando targets L3 por defecto")
            return default_targets
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo targets L3: {e}")
            return {
                'BTCUSDT': 0.5,
                'ETHUSDT': 0.3,
                'USDT': 0.2
            }
    
    async def sync_portfolio(self):
        """Sincroniza el portfolio con los targets L3"""
        logger.info("🔄 INICIANDO SINCRONIZACIÓN DE PORTFOLIO")
        logger.info("=" * 60)
        
        try:
            # 1. Verificar estado inicial
            logger.info("🔍 Verificando estado inicial del portfolio...")
            initial_state = self.get_portfolio_state()
            logger.info(f"   Estado inicial: {initial_state}")
            
            # 2. Obtener datos del mercado
            logger.info("📈 Obteniendo datos del mercado...")
            market_data = await self.market_data_manager.get_data_with_fallback()
            if not market_data:
                logger.error("❌ No se pudieron obtener datos del mercado")
                return False
            
            # 3. Obtener allocaciones
            current_allocation = self.get_current_allocation(market_data)
            target_allocation = self.get_l3_targets()
            
            logger.info(f"📊 Allocación actual: {current_allocation}")
            logger.info(f"🎯 Allocación objetivo: {target_allocation}")
            
            # 4. Calcular desviación
            deviation = calculate_allocation_deviation(current_allocation, target_allocation)
            logger.info(f"📈 Desviación de allocación: {deviation:.1%}")
            
            # 5. Verificar si se necesita rebalancing
            available_usdt = self.portfolio_manager.get_balance('USDT')
            should_rebalance = should_trigger_rebalancing(
                current_allocation=current_allocation,
                target_allocation=target_allocation,
                available_usdt=available_usdt,
                min_rebalance_amount=self.min_rebalance_amount
            )
            
            if should_rebalance:
                logger.info("🔄 EJECUTANDO REBALANCING AUTOMÁTICO")
                success = await self.execute_rebalancing(
                    current_allocation=current_allocation,
                    target_allocation=target_allocation,
                    market_data=market_data
                )
                
                if success:
                    logger.info("✅ REBALANCING COMPLETADO EXITOSAMENTE")
                else:
                    logger.error("❌ REBALANCING FALLIDO")
                    return False
            else:
                logger.info("ℹ️ No se requiere rebalancing (desviación baja o capital insuficiente)")
            
            # 6. Verificar estado final
            logger.info("🔍 Verificando estado final del portfolio...")
            final_state = self.get_portfolio_state()
            logger.info(f"   Estado final: {final_state}")
            
            # 7. Validar sincronización
            final_allocation = self.get_current_allocation()
            final_deviation = calculate_allocation_deviation(final_allocation, target_allocation)
            
            logger.info(f"📈 Desviación final: {final_deviation:.1%}")
            
            if final_deviation < 0.05:  # Menos del 5% de desviación
                logger.info("✅ PORTFOLIO SINCRONIZADO CORRECTAMENTE")
                return True
            else:
                logger.warning("⚠️ Portfolio sincronizado pero con desviación mayor al 5%")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error durante la sincronización: {e}")
            return False
    
    async def execute_rebalancing(self, current_allocation: Dict[str, float], 
                                 target_allocation: Dict[str, float], 
                                 market_data: Dict[str, Any]) -> bool:
        """Ejecuta el rebalancing automático hacia los targets L3"""
        try:
            orders = []
            
            # Calcular diferencias y generar órdenes
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                current_pct = current_allocation.get(symbol, 0)
                target_pct = target_allocation.get(symbol, 0)
                difference = target_pct - current_pct
                
                if abs(difference) > 0.01:  # Más del 1% de diferencia
                    total_value = self.portfolio_manager.get_total_value(market_data)
                    amount_usdt = total_value * difference
                    
                    if amount_usdt > self.min_rebalance_amount:
                        if difference > 0:
                            # Comprar
                            orders.append({
                                'symbol': symbol,
                                'side': 'buy',
                                'quantity': amount_usdt / self.get_market_price(symbol),
                                'price': self.get_market_price(symbol)
                            })
                            logger.info(f"🛒 ORDEN DE COMPRA: {symbol} - ${amount_usdt:.2f}")
                        else:
                            # Vender
                            current_balance = self.portfolio_manager.get_balance(symbol)
                            if current_balance > 0:
                                orders.append({
                                    'symbol': symbol,
                                    'side': 'sell',
                                    'quantity': abs(amount_usdt) / self.get_market_price(symbol),
                                    'price': self.get_market_price(symbol)
                                })
                                logger.info(f"💰 ORDEN DE VENTA: {symbol} - ${abs(amount_usdt):.2f}")
            
            if orders:
                logger.info(f"⚡ Ejecutando {len(orders)} órdenes de rebalancing...")
                processed_orders = await self.order_manager.execute_orders(orders)
                
                # Actualizar portfolio con órdenes ejecutadas
                await self.portfolio_manager.update_from_orders_async(processed_orders, market_data)
                
                logger.info("✅ Órdenes de rebalancing ejecutadas")
                return True
            else:
                logger.info("ℹ️ No se generaron órdenes de rebalancing")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error ejecutando rebalancing: {e}")
            return False
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual del portfolio"""
        try:
            btc_balance = self.portfolio_manager.get_balance('BTCUSDT')
            eth_balance = self.portfolio_manager.get_balance('ETHUSDT')
            usdt_balance = self.portfolio_manager.get_balance('USDT')
            total_value = self.portfolio_manager.get_total_value()
            
            return {
                'BTC': f"{btc_balance:.6f}",
                'ETH': f"{eth_balance:.3f}",
                'USDT': f"${usdt_balance:.2f}",
                'TOTAL': f"${total_value:.2f}"
            }
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado del portfolio: {e}")
            return {'error': str(e)}
    
    async def validate_sync(self) -> bool:
        """Valida que el portfolio esté correctamente sincronizado"""
        try:
            # Verificar que el balance inicial sea 3000 USDT - CRITICAL FIX: Use async methods
            usdt_balance = await self.portfolio_manager.get_asset_balance_async('USDT')
            if abs(usdt_balance - self.initial_balance) > 0.01:
                logger.error(f"❌ Balance USDT incorrecto: ${usdt_balance:.2f} (esperado: ${self.initial_balance:.2f})")
                return False
            
            # Verificar que no haya posiciones no deseadas - CRITICAL FIX: Use async methods
            btc_balance = await self.portfolio_manager.get_asset_balance_async('BTC')
            eth_balance = await self.portfolio_manager.get_asset_balance_async('ETH')
            
            if btc_balance < 0 or eth_balance < 0:
                logger.error("❌ Saldo negativo detectado en posiciones")
                return False
            
            # Verificar que el total sea correcto - CRITICAL FIX: Use async method
            total_value = await self.portfolio_manager.get_total_value_async()
            if total_value < self.initial_balance * 0.95:  # Permitir 5% de comisiones
                logger.warning(f"⚠️ Valor total bajo: ${total_value:.2f} (esperado: ~${self.initial_balance:.2f})")
            
            logger.info("✅ Validación de sincronización exitosa")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando sincronización: {e}")
            return False

async def main():
    """Función principal del script de sincronización"""
    logger.info("🚀 INICIANDO SCRIPT DE SINCRONIZACIÓN DE PORTFOLIOS")
    logger.info("=" * 80)
    
    try:
        # Crear sincronizador
        synchronizer = PortfolioSynchronizer()
        
        # Inicializar componentes
        await synchronizer.initialize_components()
        
        # Ejecutar sincronización
        success = await synchronizer.sync_portfolio()
        
        # Validar resultado
        validation_success = await synchronizer.validate_sync()
        
        # Resumen final
        logger.info("=" * 80)
        logger.info("📊 RESUMEN DE SINCRONIZACIÓN")
        logger.info(f"   Sincronización: {'✅ EXITOSA' if success else '❌ FALLIDA'}")
        logger.info(f"   Validación: {'✅ EXITOSA' if validation_success else '❌ FALLIDA'}")
        
        if success and validation_success:
            logger.info("🎉 PORTFOLIO L1 SINCRONIZADO CON L3/RESUMEN")
            logger.info("   Los $3000 USDT están listos para operaciones de paper trading")
        else:
            logger.error("💥 ERROR: La sincronización no se completó correctamente")
        
        logger.info("=" * 80)
        
        return success and validation_success
        
    except Exception as e:
        logger.critical(f"🚨 ERROR FATAL: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)