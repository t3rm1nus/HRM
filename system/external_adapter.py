#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExternalAdapter - Adaptador para comunicaciones externas del sistema HRM

Responsabilidades:
- Conexión y comunicación con servicios externos (Binance, News API, Reddit)
- Gestión de conexiones y timeouts
- Validación de datos externos
- Manejo de errores de red y servicios externos

Prohibido:
- Tomar decisiones de trading o inversión
- Procesar señales de trading
- Ejecutar órdenes en exchanges
- Realizar cálculos de estrategia
- Validar lógica de negocio de trading
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Importaciones del sistema HRM
from l1_operational.binance_client import BinanceClient
from l1_operational.data_feed import DataFeed
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.bus_adapter import BusAdapterAsync
from l3_strategy.sentiment_inference import download_reddit, download_news

from core.config import get_config
from core.logging import logger
from core.error_handler import ErrorHandler


class ExternalAdapter:
    """Adaptador para comunicaciones externas del sistema HRM"""
    
    def __init__(self, config: Dict[str, Any], state: Dict[str, Any]):
        self.config = config
        self.state = state
        self.components = {}
        self.initialized = False
        
    async def initialize_external_services(self) -> Dict[str, Any]:
        """
        Inicializa todos los servicios externos del sistema
        
        Returns:
            Dict[str, Any]: Estado de los servicios externos
        """
        if self.initialized:
            logger.warning("ExternalAdapter: Servicios externos ya inicializados")
            return self.components
            
        logger.info("🚀 Iniciando ExternalAdapter - Conexión a servicios externos")
        
        try:
            # Paso 1: Inicializar cliente Binance
            await self._initialize_binance_client()
            
            # Paso 2: Inicializar Data Feed
            await self._initialize_data_feed()
            
            # Paso 3: Inicializar RealTime Data Loader
            await self._initialize_realtime_loader()
            
            # Paso 4: Inicializar Bus Adapter
            await self._initialize_bus_adapter()
            
            # Paso 5: Inicializar servicios de sentimiento
            await self._initialize_sentiment_services()
            
            # Paso 6: Validación final de servicios externos
            await self._validate_external_services()
            
            self.initialized = True
            logger.info("✅ ExternalAdapter: Servicios externos inicializados exitosamente")
            
            return self.components
            
        except Exception as e:
            logger.error(f"❌ ExternalAdapter: Error durante la inicialización: {e}")
            raise
    
    async def _initialize_binance_client(self):
        """Inicializa el cliente Binance"""
        logger.info("🔧 Inicializando cliente Binance...")
        
        try:
            binance_client = BinanceClient()
            self.components['binance_client'] = binance_client
            
            # Verificar conexión
            connection_status = await binance_client.check_connection()
            if connection_status:
                logger.info("✅ Cliente Binance conectado exitosamente")
            else:
                logger.warning("⚠️ Cliente Binance: Conexión fallida, continuando con modo simulado")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando cliente Binance: {e}")
            self.components['binance_client'] = None
    
    async def _initialize_data_feed(self):
        """Inicializa el Data Feed"""
        logger.info("🔧 Inicializando Data Feed...")
        
        try:
            data_feed = DataFeed(self.config)
            self.components['data_feed'] = data_feed
            
            # Verificar disponibilidad del feed
            feed_status = await data_feed.check_availability()
            if feed_status:
                logger.info("✅ Data Feed disponible")
            else:
                logger.warning("⚠️ Data Feed: No disponible, usando fallback")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando Data Feed: {e}")
            self.components['data_feed'] = None
    
    async def _initialize_realtime_loader(self):
        """Inicializa el RealTime Data Loader"""
        logger.info("🔧 Inicializando RealTime Data Loader...")
        
        try:
            realtime_loader = RealTimeDataLoader(self.config)
            self.components['realtime_loader'] = realtime_loader
            
            # Verificar disponibilidad del loader
            loader_status = await realtime_loader.check_availability()
            if loader_status:
                logger.info("✅ RealTime Data Loader disponible")
            else:
                logger.warning("⚠️ RealTime Data Loader: No disponible, usando datos históricos")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando RealTime Data Loader: {e}")
            self.components['realtime_loader'] = None
    
    async def _initialize_bus_adapter(self):
        """Inicializa el Bus Adapter"""
        logger.info("🔧 Inicializando Bus Adapter...")
        
        try:
            bus_adapter = BusAdapterAsync(self.config, self.state)
            await bus_adapter.start()
            self.components['bus_adapter'] = bus_adapter
            
            logger.info("✅ Bus Adapter inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Bus Adapter: {e}")
            self.components['bus_adapter'] = None
    
    async def _initialize_sentiment_services(self):
        """Inicializa servicios de análisis de sentimiento"""
        logger.info("🔧 Inicializando servicios de sentimiento...")
        
        try:
            # Verificar disponibilidad de servicios externos
            reddit_available = await self._check_reddit_availability()
            news_available = await self._check_news_availability()
            
            self.components['sentiment_services'] = {
                'reddit_available': reddit_available,
                'news_available': news_available,
                'last_update': datetime.utcnow()
            }
            
            if reddit_available:
                logger.info("✅ Reddit API disponible")
            else:
                logger.warning("⚠️ Reddit API no disponible")
                
            if news_available:
                logger.info("✅ News API disponible")
            else:
                logger.warning("⚠️ News API no disponible")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando servicios de sentimiento: {e}")
            self.components['sentiment_services'] = {
                'reddit_available': False,
                'news_available': False,
                'last_update': datetime.utcnow()
            }
    
    async def _check_reddit_availability(self) -> bool:
        """Verifica disponibilidad de Reddit API"""
        try:
            # Intentar descargar datos de Reddit
            df_reddit = await download_reddit()
            return not df_reddit.empty
        except Exception as e:
            logger.warning(f"Reddit API no disponible: {e}")
            return False
    
    async def _check_news_availability(self) -> bool:
        """Verifica disponibilidad de News API"""
        try:
            # Intentar descargar datos de noticias
            df_news = download_news()
            return not df_news.empty
        except Exception as e:
            logger.warning(f"News API no disponible: {e}")
            return False
    
    async def _validate_external_services(self):
        """Valida el estado de todos los servicios externos"""
        logger.info("🔍 Validando servicios externos...")
        
        required_services = ['binance_client', 'data_feed', 'realtime_loader', 'bus_adapter']
        missing_services = []
        
        for service in required_services:
            if service not in self.components or self.components[service] is None:
                missing_services.append(service)
        
        if missing_services:
            raise RuntimeError(f"Servicios externos faltantes: {missing_services}")
        
        # Validar estado básico
        if not self.state or 'market_data' not in self.state:
            raise RuntimeError("Estado del sistema inválido para servicios externos")
        
        logger.info("✅ Validación de servicios externos completada")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Obtiene un componente externo por nombre"""
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Obtiene todos los componentes externos"""
        return self.components.copy()
    
    async def get_external_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de todos los servicios externos"""
        status = {}
        
        for name, component in self.components.items():
            if component is None:
                status[name] = "unavailable"
            elif hasattr(component, 'check_availability'):
                try:
                    available = await component.check_availability()
                    status[name] = "available" if available else "unavailable"
                except Exception:
                    status[name] = "error"
            else:
                status[name] = "initialized"
        
        return status
    
    async def cleanup(self):
        """Limpieza de recursos externos"""
        logger.info("🧹 ExternalAdapter: Iniciando limpieza de recursos externos...")
        
        try:
            # Cierre de conexiones externas
            for component_name in ['bus_adapter', 'data_feed', 'realtime_loader']:
                component = self.components.get(component_name)
                if hasattr(component, 'close'):
                    await component.close()
                    logger.info(f"✅ {component_name} cerrado")
            
            # Cierre de cliente Binance
            binance_client = self.components.get('binance_client')
            if hasattr(binance_client, 'close'):
                await binance_client.close()
                logger.info("✅ Cliente Binance cerrado")
            
            logger.info("✅ ExternalAdapter: Limpieza de recursos externos completada")
        except Exception as e:
            logger.error(f"❌ Error durante la limpieza de recursos externos: {e}")


async def create_external_adapter(config: Dict[str, Any], state: Dict[str, Any]) -> ExternalAdapter:
    """
    Factory function para crear y configurar el ExternalAdapter
    
    Returns:
        ExternalAdapter: Instancia configurada del adaptador externo
    """
    adapter = ExternalAdapter(config, state)
    await adapter.initialize_external_services()
    return adapter