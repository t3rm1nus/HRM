#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SystemOrchestrator - Coordinador de inicialización del sistema HRM

Responsabilidades:
- Inicialización secuencial de componentes del sistema
- Configuración de entorno y variables de entorno
- Establecimiento de conexiones con servicios externos
- Validación de dependencias y pre-requisitos
- Arranque coordinado de subsistemas (L1, L2, L3)

Prohibido:
- Tomar decisiones de trading o inversión
- Procesar señales de trading
- Ejecutar órdenes en exchanges
- Realizar cálculos de estrategia
- Validar lógica de negocio de trading
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Importaciones del sistema HRM
from core.state_manager import initialize_state, validate_state_structure
from core.config import get_config
from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler
from core.incremental_signal_verifier import get_signal_verifier
from core.trading_metrics import get_trading_metrics
from core.position_rotator import PositionRotator, AutoRebalancer

from l1_operational.data_feed import DataFeed
from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.order_manager import OrderManager
from l1_operational.bus_adapter import BusAdapterAsync

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.risk_controls.manager import RiskControlManager

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus
from l2_tactic.config import L2Config


class SystemOrchestrator:
    """Coordinador de inicialización del sistema HRM"""
    
    def __init__(self):
        self.components = {}
        self.initialized = False
        
    async def orchestrate_initialization(self) -> Dict[str, Any]:
        """
        Orquesta la inicialización completa del sistema
        
        Returns:
            Dict[str, Any]: Estado del sistema inicializado
        """
        if self.initialized:
            logger.warning("SystemOrchestrator: Sistema ya inicializado")
            return self.components
            
        logger.info("🚀 Iniciando SystemOrchestrator - Coordinación de inicialización")
        
        try:
            # Paso 1: Configuración básica del sistema
            await self._initialize_basic_components()
            
            # Paso 2: Configuración de componentes L1
            await self._initialize_l1_components()
            
            # Paso 3: Configuración de componentes L2
            await self._initialize_l2_components()
            
            # Paso 4: Configuración de componentes L3 (si está habilitado)
            if not APAGAR_L3:
                await self._initialize_l3_components()
            else:
                logger.info("🔴 L3 MODULE DISABLED - Skipping L3 initialization")
                self.components['l3_output'] = {
                    'regime': 'disabled',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_type': 'l3_disabled',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Paso 5: Validación final del sistema
            await self._validate_system_state()
            
            self.initialized = True
            logger.info("✅ SystemOrchestrator: Sistema inicializado exitosamente")
            
            return self.components
            
        except Exception as e:
            logger.error(f"❌ SystemOrchestrator: Error durante la inicialización: {e}")
            raise
    
    async def _initialize_basic_components(self):
        """Inicializa componentes básicos del sistema"""
        logger.info("🔧 Inicializando componentes básicos...")
        
        # 1. Configuración del sistema
        env_config = get_config("live")
        self.components['config'] = env_config
        logger.info(f"✅ Configuración cargada: {list(env_config.keys())}")
        
        # 2. Estado del sistema
        state = initialize_state(env_config["SYMBOLS"], 3000.0)
        state = validate_state_structure(state)
        self.components['state'] = state
        logger.info(f"✅ Estado del sistema inicializado: {len(env_config['SYMBOLS'])} símbolos")
        
        # 3. Portfolio Manager
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        portfolio_mode = "live" if binance_mode == "LIVE" else "simulated"
        initial_balance = 0.0 if binance_mode == "LIVE" else 3000.0
        
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )
        self.components['portfolio_manager'] = portfolio_manager
        logger.info(f"✅ Portfolio Manager inicializado: modo={portfolio_mode}, balance={initial_balance}")
        
        # 4. Clientes y adaptadores
        binance_client = BinanceClient()
        self.components['binance_client'] = binance_client
        
        bus_adapter = BusAdapterAsync(config, state)
        self.components['bus_adapter'] = bus_adapter
        
        data_feed = DataFeed(config)
        self.components['data_feed'] = data_feed
        
        loader = RealTimeDataLoader(config)
        self.components['loader'] = loader
        
        logger.info("✅ Componentes básicos inicializados")
    
    async def _initialize_l1_components(self):
        """Inicializa componentes L1 del sistema"""
        logger.info("🔧 Inicializando componentes L1...")
        
        # 1. Order Manager
        order_manager = OrderManager(
            binance_client=self.components['binance_client'],
            market_data=self.components['state'].get("market_data", {}),
            portfolio_manager=self.components['portfolio_manager']
        )
        self.components['order_manager'] = order_manager
        logger.info("✅ Order Manager L1 inicializado")
        
        # 2. L1 Models (importación segura)
        try:
            from l1_operational.trend_ai import models as l1_models
            self.components['l1_models'] = l1_models
            logger.info(f"✅ L1 AI Models cargados: {list(l1_models.keys())}")
        except ImportError as e:
            logger.error(f"❌ Error cargando L1 models: {e}")
            self.components['l1_models'] = {}
        
        logger.info("✅ Componentes L1 inicializados")
    
    async def _initialize_l2_components(self):
        """Inicializa componentes L2 del sistema"""
        logger.info("🔧 Inicializando componentes L2...")
        
        # 1. L2 Config
        l2_config = L2Config()
        self.components['l2_config'] = l2_config
        
        # 2. L2 Processor
        portfolio_manager = self.components['portfolio_manager']
        l2_processor = L2TacticProcessor(
            l2_config, 
            portfolio_manager=portfolio_manager, 
            apagar_l3=APAGAR_L3
        )
        self.components['l2_processor'] = l2_processor
        
        # 3. Risk Manager
        risk_manager = RiskControlManager(l2_config)
        self.components['risk_manager'] = risk_manager
        
        logger.info("✅ Componentes L2 inicializados")
    
    async def _initialize_l3_components(self):
        """Inicializa componentes L3 del sistema"""
        logger.info("🔧 Inicializando componentes L3...")
        
        # Importación segura de L3 components
        try:
            from l3_strategy.l3_processor import generate_l3_output
            from l3_strategy.sentiment_inference import update_sentiment_texts
            
            # 1. Sentiment Analysis
            sentiment_texts = await update_sentiment_texts()
            self.components['sentiment_texts'] = sentiment_texts
            
            # 2. L3 Output Generation
            l3_output = generate_l3_output(
                self.components['state'], 
                texts_for_sentiment=sentiment_texts
            )
            self.components['l3_output'] = l3_output
            
            logger.info("✅ Componentes L3 inicializados")
            
        except ImportError as e:
            logger.error(f"❌ Error cargando L3 components: {e}")
            self.components['l3_output'] = {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'l3_error',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error inicializando L3: {e}")
            self.components['l3_output'] = {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'l3_error',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _validate_system_state(self):
        """Valida el estado final del sistema"""
        logger.info("🔍 Validando estado del sistema...")
        
        required_components = [
            'config', 'state', 'portfolio_manager', 'binance_client',
            'order_manager', 'l2_processor', 'l3_output'
        ]
        
        missing_components = []
        for component in required_components:
            if component not in self.components:
                missing_components.append(component)
        
        if missing_components:
            raise RuntimeError(f"Componentes faltantes: {missing_components}")
        
        # Validación de estado básico
        state = self.components['state']
        if not state or 'market_data' not in state:
            raise RuntimeError("Estado del sistema inválido")
        
        logger.info("✅ Validación del sistema completada exitosamente")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Obtiene un componente por nombre"""
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, Any]:
        """Obtiene todos los componentes del sistema"""
        return self.components.copy()
    
    async def cleanup(self):
        """Limpieza de recursos"""
        logger.info("🧹 SystemOrchestrator: Iniciando limpieza de recursos...")
        
        try:
            # Cierre de adaptadores
            for component_name in ['bus_adapter', 'data_feed', 'loader']:
                component = self.components.get(component_name)
                if hasattr(component, 'close'):
                    await component.close()
                    logger.info(f"✅ {component_name} cerrado")
            
            logger.info("✅ SystemOrchestrator: Limpieza completada")
        except Exception as e:
            logger.error(f"❌ Error durante la limpieza: {e}")


async def create_system_orchestrator() -> SystemOrchestrator:
    """
    Factory function para crear y configurar el SystemOrchestrator
    
    Returns:
        SystemOrchestrator: Instancia configurada del orchestrator
    """
    orchestrator = SystemOrchestrator()
    await orchestrator.orchestrate_initialization()
    return orchestrator