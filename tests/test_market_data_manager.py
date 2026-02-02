#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para MarketDataManager.

Testea todas las funcionalidades del gestor de datos de mercado.
"""

import asyncio
import time
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from system.market_data_manager import (
    MarketDataManager, 
    FallbackStrategy, 
    CacheEntry,
    get_market_data_with_fallback
)
from core.unified_validation import UnifiedValidator


class TestMarketDataManager:
    """Suite de pruebas para MarketDataManager."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.config = {
            "SYMBOLS": ["BTCUSDT", "ETHUSDT"],
            "VALIDATION_RETRIES": 3,
            "CACHE_VALID_SECONDS": 30,
            "FALLBACK_STRATEGY": "external->realtime->datafeed"
        }
        self.manager = MarketDataManager(self.config)
    
    def test_initialization(self):
        """Prueba la inicialización del gestor."""
        assert self.manager.symbols == ["BTCUSDT", "ETHUSDT"]
        assert self.manager.validation_retries == 3
        assert self.manager.cache_valid_seconds == 30
        assert self.manager.fallback_strategy == FallbackStrategy.EXTERNAL_TO_REALTIME_TO_DATAFEED
        assert self.manager.stats["attempts"] == 0
        assert self.manager._cache is None
    
    def test_fallback_strategies(self):
        """Prueba las diferentes estrategias de fallback."""
        strategies = [
            ("external->realtime->datafeed", FallbackStrategy.EXTERNAL_TO_REALTIME_TO_DATAFEED),
            ("realtime->datafeed", FallbackStrategy.REALTIME_TO_DATAFEED),
            ("datafeed_only", FallbackStrategy.DATAFEED_ONLY)
        ]
        
        for strategy_str, expected_enum in strategies:
            config = {"FALLBACK_STRATEGY": strategy_str}
            manager = MarketDataManager(config)
            assert manager.fallback_strategy == expected_enum
    
    def test_cache_entry(self):
        """Prueba la creación de entradas de caché."""
        data = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        entry = CacheEntry(
            data=data,
            timestamp=1234567890.0,
            source="test",
            validation_passed=True
        )
        
        assert entry.data == data
        assert entry.timestamp == 1234567890.0
        assert entry.source == "test"
        assert entry.validation_passed is True
    
    @pytest.mark.asyncio
    async def test_get_cached_data_hit(self):
        """Prueba el hit de caché."""
        # Crear datos de caché
        cached_data = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        self.manager._cache = CacheEntry(
            data=cached_data,
            timestamp=time.time() - 10,  # 10 segundos atrás
            source="test",
            validation_passed=True
        )
        
        result = await self.manager._get_cached_data()
        assert result == cached_data
        assert self.manager.stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_get_cached_data_expired(self):
        """Prueba el miss de caché por expiración."""
        # Crear datos de caché expirado
        self.manager._cache = CacheEntry(
            data={"BTCUSDT": pd.DataFrame({"close": [100.0]})},
            timestamp=time.time() - 100,  # 100 segundos atrás (expirado)
            source="test",
            validation_passed=True
        )
        
        result = await self.manager._get_cached_data()
        assert result is None
        assert self.manager._cache is None  # Debe limpiarse
    
    @pytest.mark.asyncio
    async def test_validate_data_empty(self):
        """Prueba la validación de datos vacíos."""
        result = await self.manager._validate_data({})
        assert result == {}
        # No se incrementa attempts porque no se llama a validate_market_data_structure
        assert self.manager.stats["validation_failures"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_data_valid(self):
        """Prueba la validación de datos válidos."""
        # Crear datos válidos
        valid_data = {
            "BTCUSDT": pd.DataFrame({
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0]
            })
        }
        
        with patch.object(UnifiedValidator, 'validate_market_data_structure', return_value=(True, "Valid")):
            with patch.object(UnifiedValidator, 'validate_symbol_data_required', return_value=(valid_data, "Valid")):
                result = await self.manager._validate_data(valid_data)
        
        assert result == valid_data
        assert self.manager.stats["successes"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_data_with_repair(self):
        """Prueba la validación con reparación automática."""
        # Crear datos inválidos que pueden repararse
        invalid_data = {
            "BTCUSDT": {"close": 100.0}  # Dict en lugar de DataFrame
        }
        
        with patch.object(UnifiedValidator, 'validate_market_data_structure', return_value=(True, "Valid")):
            with patch.object(UnifiedValidator, 'validate_symbol_data_required', return_value=({}, "Invalid")):
                result = await self.manager._validate_data(invalid_data)
        
        # Debe intentar reparar y crear un DataFrame
        assert "BTCUSDT" in result
        assert isinstance(result["BTCUSDT"], pd.DataFrame)
        assert self.manager.stats["repaired"] == 1
    
    @pytest.mark.asyncio
    async def test_get_external_data_success(self):
        """Prueba la obtención de datos de ExternalAdapter exitosa."""
        # Mock del ExternalAdapter
        mock_external = Mock()
        mock_loader = AsyncMock()
        mock_loader.get_market_data.return_value = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        mock_external.get_component.return_value = mock_loader
        
        self.manager.external_adapter = mock_external
        
        result = await self.manager._get_external_data()
        
        assert "BTCUSDT" in result
        assert isinstance(result["BTCUSDT"], pd.DataFrame)
        assert self.manager.stats["successes"] == 0  # No cuenta aquí, solo en validate_data
    
    @pytest.mark.asyncio
    async def test_get_external_data_failure(self):
        """Prueba la obtención de datos de ExternalAdapter fallida."""
        # Mock que lanza excepción
        mock_external = Mock()
        mock_external.get_component.side_effect = Exception("Connection failed")
        
        self.manager.external_adapter = mock_external
        
        result = await self.manager._get_external_data()
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_realtime_data_success(self):
        """Prueba la obtención de datos de RealTimeLoader exitosa."""
        # Mock del RealTimeLoader
        mock_loader = AsyncMock()
        mock_loader.get_realtime_data.return_value = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        
        self.manager.realtime_loader = mock_loader
        
        result = await self.manager._get_realtime_data()
        
        assert "BTCUSDT" in result
        assert isinstance(result["BTCUSDT"], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_get_datafeed_data_success(self):
        """Prueba la obtención de datos de DataFeed exitosa."""
        # Mock del DataFeed
        mock_feed = AsyncMock()
        mock_feed.get_market_data.return_value = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        
        self.manager.data_feed = mock_feed
        
        result = await self.manager._get_datafeed_data()
        
        assert "BTCUSDT" in result
        assert isinstance(result["BTCUSDT"], pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_get_market_data_with_cache(self):
        """Prueba la obtención de datos usando caché."""
        # Crear datos en caché
        cached_data = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        self.manager._cache = CacheEntry(
            data=cached_data,
            timestamp=time.time() - 10,
            source="test",
            validation_passed=True
        )
        
        # Mock de fuentes para que no se llamen
        with patch.object(self.manager, '_get_external_data', return_value={}) as mock_external:
            with patch.object(self.manager, '_get_realtime_data', return_value={}) as mock_realtime:
                with patch.object(self.manager, '_get_datafeed_data', return_value={}) as mock_datafeed:
                    result = await self.manager.get_market_data()
        
        assert result == cached_data
        assert self.manager.stats["cache_hits"] == 1
        # Las fuentes no deben ser llamadas
        mock_external.assert_not_called()
        mock_realtime.assert_not_called()
        mock_datafeed.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_market_data_with_fallback(self):
        """Prueba la obtención de datos con fallback completo."""
        # Mock de fuentes fallando en cascada
        with patch.object(self.manager, '_get_external_data', return_value={}) as mock_external:
            with patch.object(self.manager, '_get_realtime_data', return_value={}) as mock_realtime:
                with patch.object(self.manager, '_get_datafeed_data', return_value={}) as mock_datafeed:
                    with patch.object(self.manager, '_validate_data', return_value={}) as mock_validate:
                        result = await self.manager.get_market_data()
        
        assert result == {}
        # No se incrementa fallbacks porque no se llama a las fuentes reales
        assert self.manager.stats["fallbacks"] == 0
        mock_external.assert_called_once()
        mock_realtime.assert_called_once()
        mock_datafeed.assert_called_once()
        mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_market_data_external_success(self):
        """Prueba la obtención de datos con éxito en ExternalAdapter."""
        external_data = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
        
        with patch.object(self.manager, '_get_external_data', return_value=external_data) as mock_external:
            with patch.object(self.manager, '_validate_data', return_value=external_data) as mock_validate:
                with patch.object(self.manager, '_update_cache') as mock_cache:
                    result = await self.manager.get_market_data()
        
        assert result == external_data
        mock_external.assert_called_once()
        mock_validate.assert_called_once_with(external_data)
        mock_cache.assert_called_once_with(external_data, "external")
    
    @pytest.mark.asyncio
    async def test_refresh_data(self):
        """Prueba la actualización forzada de datos."""
        # Crear datos en caché
        self.manager._cache = CacheEntry(
            data={"BTCUSDT": pd.DataFrame({"close": [100.0]})},
            timestamp=time.time() - 10,
            source="test",
            validation_passed=True
        )
        
        # Mock para obtener nuevos datos
        new_data = {"BTCUSDT": pd.DataFrame({"close": [101.0]})}
        
        with patch.object(self.manager, 'get_market_data', return_value=new_data) as mock_get:
            result = await self.manager.refresh_data()
        
        assert result == new_data
        assert self.manager._cache is None  # Caché debe limpiarse
        mock_get.assert_called_once()
    
    def test_get_stats(self):
        """Prueba la obtención de estadísticas."""
        stats = self.manager.get_stats()
        
        assert "symbols" in stats
        assert "strategy" in stats
        assert "cache_valid_seconds" in stats
        assert "stats" in stats
        assert "cache_status" in stats
        
        assert stats["symbols"] == ["BTCUSDT", "ETHUSDT"]
        assert stats["strategy"] == "external->realtime->datafeed"
        assert stats["cache_valid_seconds"] == 30
        assert stats["cache_status"]["has_cache"] is False
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Prueba el cierre del gestor."""
        # Mock de componentes
        mock_realtime = AsyncMock()
        mock_datafeed = AsyncMock()
        
        self.manager.realtime_loader = mock_realtime
        self.manager.data_feed = mock_datafeed
        
        await self.manager.close()
        
        mock_realtime.close.assert_called_once()
        mock_datafeed.close.assert_called_once()


class TestConvenienceFunction:
    """Pruebas para la función de conveniencia."""
    
    @pytest.mark.asyncio
    async def test_get_market_data_with_fallback(self):
        """Prueba la función de conveniencia."""
        with patch('system.market_data_manager.MarketDataManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.get_market_data.return_value = {"BTCUSDT": pd.DataFrame({"close": [100.0]})}
            mock_manager_class.return_value = mock_manager
            
            result = await get_market_data_with_fallback()
            
            # Comparar DataFrames de forma segura
            assert "BTCUSDT" in result
            assert isinstance(result["BTCUSDT"], pd.DataFrame)
            assert result["BTCUSDT"]["close"].iloc[0] == 100.0
            mock_manager.get_market_data.assert_called_once()
            mock_manager.close.assert_called_once()


if __name__ == "__main__":
    # Ejecutar pruebas
    pytest.main([__file__, "-v"])