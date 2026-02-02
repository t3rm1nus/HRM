#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Data Manager - HRM Trading System

Centraliza la obtención de market data con lógica de fuentes primarias y fallbacks.
Gestiona validación, reparación y caché de datos de mercado.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from core.logging import logger
from core.unified_validation import UnifiedValidator
from core.config import get_config
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.data_feed import DataFeed
from comms.config import config


class FallbackStrategy(Enum):
    """Estrategia de fallback para fuentes de datos."""
    EXTERNAL_TO_REALTIME_TO_DATAFEED = "external->realtime->datafeed"
    REALTIME_TO_DATAFEED = "realtime->datafeed"
    DATAFEED_ONLY = "datafeed_only"


@dataclass
class CacheEntry:
    """Entrada de caché para datos de mercado."""
    data: Dict[str, pd.DataFrame]
    timestamp: float
    source: str
    validation_passed: bool


class MarketDataManager:
    """
    Gestor centralizado de datos de mercado.
    
    Características:
    - Fuentes primarias y fallbacks automáticos
    - Validación y reparación automática
    - Caché de datos válidos
    - Logging detallado de decisiones
    - Configuración flexible
    """
    
    def __init__(self, symbols: Optional[List[str]] = None, fallback_enabled: bool = True, config_dict: Optional[Dict[str, Any]] = None):
        """
        Inicializa el MarketDataManager.
        
        Args:
            symbols: Lista de símbolos a manejar (usa config global si no se proporciona)
            fallback_enabled: Habilita estrategia de fallback (usa config global si no se proporciona)
            config_dict: Configuración opcional (usa config global si no se proporciona)
        """
        # Construir config_dict combinando parámetros con config global
        self.config = config_dict or config.copy()
        
        if symbols is not None:
            self.config["SYMBOLS"] = symbols
        if fallback_enabled is not None:
            self.config["FALLBACK_STRATEGY"] = "external->realtime->datafeed" if fallback_enabled else "datafeed_only"
        
        # Configuración
        self.symbols = self.config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        self.validation_retries = self.config.get("VALIDATION_RETRIES", 3)
        self.cache_valid_seconds = self.config.get("CACHE_VALID_SECONDS", 30)
        self.fallback_strategy = FallbackStrategy(
            self.config.get("FALLBACK_STRATEGY", "external->realtime->datafeed")
        )
        
        # Componentes
        self.realtime_loader = None
        self.data_feed = None
        self.external_adapter = None
        
        # Caché
        self._cache: Optional[CacheEntry] = None
        self._cache_lock = asyncio.Lock()
        
        # Contadores de estadísticas
        self.stats = {
            "attempts": 0,
            "successes": 0,
            "fallbacks": 0,
            "cache_hits": 0,
            "validation_failures": 0,
            "repaired": 0
        }
        
        logger.info(f"✅ MarketDataManager inicializado con {len(self.symbols)} símbolos")
        logger.info(f"   Estrategia: {self.fallback_strategy.value}")
        logger.info(f"   Cache: {self.cache_valid_seconds}s")
        logger.info(f"   Reintentos: {self.validation_retries}")
    
    async def _init_components(self):
        """Inicializa los componentes de carga de datos."""
        if not self.realtime_loader:
            self.realtime_loader = RealTimeDataLoader(self.config)
            logger.info("✅ RealTimeLoader inicializado")
        
        if not self.data_feed:
            self.data_feed = DataFeed(self.config)
            logger.info("✅ DataFeed inicializado")
    
    async def _get_external_data(self) -> Dict[str, pd.DataFrame]:
        """
        Intenta obtener datos de ExternalAdapter (fuente primaria).
        
        Returns:
            Dict con datos de mercado o dict vacío si falla
        """
        if not self.external_adapter:
            logger.debug("⚠️ ExternalAdapter no disponible")
            return {}
        
        try:
            # Intentar obtener datos del ExternalAdapter
            if hasattr(self.external_adapter, 'get_component') and \
               hasattr(self.external_adapter.get_component('realtime_loader'), 'get_market_data'):
                
                data = await self.external_adapter.get_component('realtime_loader').get_market_data()
                logger.info("📡 Intentando obtener datos de ExternalAdapter (fuente primaria)")
                
                if data:
                    logger.info(f"✅ ExternalAdapter exitoso: {list(data.keys())}")
                    return data
                else:
                    logger.warning("⚠️ ExternalAdapter retornó datos vacíos")
                    return {}
            else:
                logger.warning("⚠️ ExternalAdapter no tiene métodos requeridos")
                return {}
                
        except Exception as e:
            logger.error(f"❌ ExternalAdapter falló: {e}")
            return {}
    
    async def _get_realtime_data(self) -> Dict[str, pd.DataFrame]:
        """
        Intenta obtener datos de RealTimeLoader.
        
        Returns:
            Dict con datos de mercado o dict vacío si falla
        """
        try:
            await self._init_components()
            logger.info("📡 Intentando obtener datos de RealTimeLoader (fallback 1)")
            
            data = await self.realtime_loader.get_realtime_data()
            
            if data:
                logger.info(f"✅ RealTimeLoader exitoso: {list(data.keys())}")
                return data
            else:
                logger.warning("⚠️ RealTimeLoader retornó datos vacíos")
                return {}
                
        except Exception as e:
            logger.error(f"❌ RealTimeLoader falló: {e}")
            return {}
    
    async def _get_datafeed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Intenta obtener datos de DataFeed (último fallback).
        
        Returns:
            Dict con datos de mercado o dict vacío si falla
        """
        try:
            await self._init_components()
            logger.info("📡 Intentando obtener datos de DataFeed (fallback 2)")
            
            data = await self.data_feed.get_market_data()
            
            if data:
                logger.info(f"✅ DataFeed exitoso: {list(data.keys())}")
                return data
            else:
                logger.warning("⚠️ DataFeed retornó datos vacíos")
                return {}
                
        except Exception as e:
            logger.error(f"❌ DataFeed falló: {e}")
            return {}
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Valida y repara datos de mercado usando UnifiedValidator.
        
        Args:
            data: Datos a validar
            
        Returns:
            Dict con datos validados y reparados
        """
        if not data:
            logger.warning("⚠️ Validación: Datos vacíos")
            return {}
        
        self.stats["attempts"] += 1
        
        # Validar estructura general
        is_valid, validation_msg = UnifiedValidator.validate_market_data_structure(data)
        
        if not is_valid:
            logger.warning(f"⚠️ Validación fallida: {validation_msg}")
            self.stats["validation_failures"] += 1
            return {}
        
        # Validar datos por símbolo
        valid_data = {}
        repair_count = 0
        
        for symbol, symbol_data in data.items():
            if symbol not in self.symbols:
                continue
            
            try:
                # Validar datos del símbolo
                symbol_valid_data, symbol_msg = UnifiedValidator.validate_symbol_data_required(
                    [symbol], {symbol: symbol_data}
                )
                
                if symbol_valid_data:
                    valid_data[symbol] = symbol_valid_data[symbol]
                    logger.debug(f"✅ {symbol}: {symbol_msg}")
                else:
                    logger.warning(f"⚠️ {symbol}: {symbol_msg}")
                    # Intentar reparar datos
                    repaired = self._repair_symbol_data(symbol, symbol_data)
                    if repaired is not None:
                        valid_data[symbol] = repaired
                        repair_count += 1
                        logger.info(f"🔧 {symbol}: Datos reparados")
                    else:
                        logger.error(f"❌ {symbol}: No se pudo reparar")
                        
            except Exception as e:
                logger.error(f"❌ Error validando {symbol}: {e}")
        
        if repair_count > 0:
            self.stats["repaired"] += repair_count
            logger.info(f"🔧 Reparados {repair_count} símbolos")
        
        if valid_data:
            self.stats["successes"] += 1
            logger.info(f"✅ Validación exitosa: {len(valid_data)} símbolos válidos")
        else:
            logger.error("❌ Validación fallida: No hay símbolos válidos")
        
        return valid_data
    
    def _repair_symbol_data(self, symbol: str, data: Any) -> Optional[pd.DataFrame]:
        """
        Intenta reparar datos de un símbolo.
        
        Args:
            symbol: Símbolo a reparar
            data: Datos a reparar
            
        Returns:
            DataFrame reparado o None si no se puede reparar
        """
        try:
            if isinstance(data, dict):
                # Convertir dict a DataFrame
                df = pd.DataFrame([data])
                if not df.empty:
                    return df
            
            elif isinstance(data, list) and data:
                # Convertir lista a DataFrame
                df = pd.DataFrame(data)
                if not df.empty:
                    return df
            
            # Intentar crear DataFrame vacío con columnas estándar
            df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            return df
            
        except Exception as e:
            logger.error(f"❌ No se pudo reparar {symbol}: {e}")
            return None
    
    async def _update_cache(self, data: Dict[str, pd.DataFrame], source: str):
        """Actualiza el caché con datos válidos."""
        async with self._cache_lock:
            self._cache = CacheEntry(
                data=data,
                timestamp=time.time(),
                source=source,
                validation_passed=True
            )
            logger.info(f"💾 Caché actualizado desde {source}")
    
    async def _get_cached_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Obtiene datos del caché si están vigentes."""
        async with self._cache_lock:
            if self._cache is None:
                return None
            
            cache_age = time.time() - self._cache.timestamp
            
            if cache_age <= self.cache_valid_seconds:
                self.stats["cache_hits"] += 1
                logger.info(f"💾 Cache hit: {self._cache.source} (edad: {cache_age:.1f}s)")
                return self._cache.data
            else:
                logger.info(f"⏰ Cache expirado: {self._cache.source} (edad: {cache_age:.1f}s)")
                self._cache = None
                return None
    
    async def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos de mercado usando la estrategia de fuentes y fallbacks.
        
        Returns:
            Dict con datos de mercado válidos (puede estar vacío)
        """
        try:
            # 1. Intentar obtener del caché
            cached_data = await self._get_cached_data()
            if cached_data:
                return cached_data
            
            # 2. Intentar fuentes según estrategia
            data = {}
            source_used = "none"
            
            if self.fallback_strategy == FallbackStrategy.EXTERNAL_TO_REALTIME_TO_DATAFEED:
                # Intentar ExternalAdapter primero
                data = await self._get_external_data()
                if data:
                    source_used = "external"
                else:
                    # Fallback a RealTimeLoader
                    data = await self._get_realtime_data()
                    if data:
                        source_used = "realtime"
                        self.stats["fallbacks"] += 1
                    else:
                        # Fallback final a DataFeed
                        data = await self._get_datafeed_data()
                        if data:
                            source_used = "datafeed"
                            self.stats["fallbacks"] += 2
                        else:
                            logger.error("❌ Todas las fuentes de datos fallaron")
            
            elif self.fallback_strategy == FallbackStrategy.REALTIME_TO_DATAFEED:
                # Intentar RealTimeLoader primero
                data = await self._get_realtime_data()
                if data:
                    source_used = "realtime"
                else:
                    # Fallback a DataFeed
                    data = await self._get_datafeed_data()
                    if data:
                        source_used = "datafeed"
                        self.stats["fallbacks"] += 1
            
            elif self.fallback_strategy == FallbackStrategy.DATAFEED_ONLY:
                # Solo DataFeed
                data = await self._get_datafeed_data()
                if data:
                    source_used = "datafeed"
            
            # 3. Validar y reparar datos
            validated_data = await self._validate_data(data)
            
            # 4. Actualizar caché si hay datos válidos
            if validated_data:
                await self._update_cache(validated_data, source_used)
                return validated_data
            else:
                logger.warning("⚠️ No se obtuvieron datos válidos después de validación")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error en get_market_data: {e}", exc_info=True)
            return {}
    
    async def validate_and_repair(self, data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Valida y repara datos de mercado.
        
        Args:
            data: Datos a validar y reparar
            
        Returns:
            Dict con datos validados y reparados
        """
        try:
            return await self._validate_data(data)
        except Exception as e:
            logger.error(f"❌ Error en validate_and_repair: {e}")
            return {}
    
    async def get_data_with_fallback(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos con lógica de fallback simplificada.
        Método expuesto para compatibilidad.
        
        Returns:
            Dict con datos de mercado
        """
        return await self.get_market_data()
    
    async def refresh_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fuerza la actualización de datos (ignora caché).
        
        Returns:
            Dict con datos de mercado actualizados
        """
        try:
            # Limpiar caché
            async with self._cache_lock:
                self._cache = None
                logger.info("🗑️ Caché limpiado para actualización forzada")
            
            # Obtener nuevos datos
            return await self.get_market_data()
            
        except Exception as e:
            logger.error(f"❌ Error en refresh_data: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de operación."""
        return {
            "symbols": self.symbols,
            "strategy": self.fallback_strategy.value,
            "cache_valid_seconds": self.cache_valid_seconds,
            "stats": self.stats.copy(),
            "cache_status": {
                "has_cache": self._cache is not None,
                "cache_source": self._cache.source if self._cache else None,
                "cache_age_seconds": time.time() - self._cache.timestamp if self._cache else 0
            }
        }
    
    async def close(self):
        """Cierra conexiones y recursos."""
        try:
            if self.realtime_loader:
                await self.realtime_loader.close()
            if self.data_feed:
                await self.data_feed.close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.error(f"❌ Error cerrando MarketDataManager: {e}")


# Función de conveniencia para compatibilidad con main.py
async def get_market_data_with_fallback() -> Dict[str, pd.DataFrame]:
    """
    Función de conveniencia para obtener datos con fallback.
    Mantiene compatibilidad con el código existente en main.py
    """
    manager = MarketDataManager()
    try:
        data = await manager.get_market_data()
        return data
    finally:
        await manager.close()