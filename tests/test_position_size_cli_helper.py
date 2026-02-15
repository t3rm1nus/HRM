# -*- coding: utf-8 -*-
"""
Tests para PositionSizeCLIHelper

Este módulo prueba:
1. Lectura de balance desde balance_cache
2. Cálculo de qty con allocation porcentual
3. Redondeo al mínimo decimal permitido
4. Validación de min_qty
5. Marcado de señales inválidas
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.position_size_cli_helper import PositionSizeCLIHelper, PositionSizeResult


class MockPortfolioManager:
    """Mock del PortfolioManager para testing"""
    
    def __init__(self, balances=None):
        self._balance_cache = balances or {
            "BTC": 0.01549,
            "ETH": 0.385,
            "USDT": 3000.0
        }
        self.portfolio = {
            "BTCUSDT": {"position": 0.01549, "free": 0.01549},
            "ETHUSDT": {"position": 0.385, "free": 0.385},
            "USDT": {"free": 3000.0}
        }
    
    async def get_asset_balance_async(self, asset: str) -> float:
        return self._balance_cache.get(asset, 0.0)
    
    async def get_balances_async(self) -> dict:
        return self._balance_cache.copy()


@pytest.fixture
def mock_pm():
    """Fixture para PortfolioManager mock"""
    return MockPortfolioManager()


@pytest.fixture
def helper(mock_pm):
    """Fixture para PositionSizeCLIHelper"""
    return PositionSizeCLIHelper(mock_pm, min_order_value=2.0)


class TestPositionSizeCLIHelper:
    """Test suite para PositionSizeCLIHelper"""
    
    def test_initialization(self, mock_pm):
        """Test que el helper se inicializa correctamente"""
        helper = PositionSizeCLIHelper(mock_pm, min_order_value=5.0, default_allocation_pct=0.15)
        
        assert helper.portfolio_manager == mock_pm
        assert helper.min_order_value == 5.0
        assert helper.default_allocation_pct == 0.15
    
    def test_get_asset_precision(self, helper):
        """Test obtención de precisión por asset"""
        assert helper.get_asset_precision("BTC") == 6
        assert helper.get_asset_precision("ETH") == 5
        assert helper.get_asset_precision("USDT") == 2
        assert helper.get_asset_precision("BTCUSDT") == 6  # Debe normalizar
        assert helper.get_asset_precision("UNKNOWN") == 6  # Default
    
    def test_round_to_precision(self, helper):
        """Test redondeo a precisión correcta"""
        # BTC - 6 decimales
        assert helper.round_to_precision(0.0012345678, "BTC") == 0.001234
        
        # ETH - 5 decimales
        assert helper.round_to_precision(0.012345678, "ETH") == 0.01234
        
        # USDT - 2 decimales
        assert helper.round_to_precision(123.456, "USDT") == 123.45
    
    def test_calculate_min_qty_threshold(self, helper):
        """Test cálculo de threshold mínimo"""
        # BTC a $50,000, min $2
        # min_qty = 2 / 50000 = 0.00004 -> redondeado a 0.00004
        min_qty = helper.calculate_min_qty_threshold(50000.0, "BTC", 2.0)
        assert min_qty > 0
        assert min_qty == 0.00004
        
        # ETH a $3,000, min $2
        # min_qty = 2 / 3000 = 0.000666... -> redondeado a 0.00066
        min_qty = helper.calculate_min_qty_threshold(3000.0, "ETH", 2.0)
        assert min_qty == 0.00066
    
    @pytest.mark.asyncio
    async def test_get_balance_from_cache(self, helper, mock_pm):
        """Test lectura de balance desde cache"""
        balance = await helper.get_balance_from_cache("BTC")
        assert balance == 0.01549
        
        balance = await helper.get_balance_from_cache("ETH")
        assert balance == 0.385
        
        balance = await helper.get_balance_from_cache("USDT")
        assert balance == 3000.0
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_buy_valid(self, helper):
        """Test cálculo de posición BUY válida"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=50000.0,
            allocation_pct=0.10,  # 10% de $3000 = $300
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is True
        assert result.qty > 0
        # $300 / $50,000 = 0.006 BTC -> redondeado a 6 decimales
        assert result.qty == 0.006
        assert result.order_value_usd == 300.0
        assert result.balance_used == 3000.0
        assert result.allocation_pct == 0.10
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_sell_valid(self, helper):
        """Test cálculo de posición SELL válida"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="sell",
            current_price=50000.0,
            allocation_pct=0.50,  # 50% de 0.01549 BTC
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is True
        assert result.qty > 0
        # 0.01549 * 0.50 = 0.007745 -> redondeado a 0.007745
        assert result.qty == 0.007745
        assert result.balance_used == 0.01549
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_buy_invalid_qty_too_small(self, helper):
        """Test que qty muy pequeño marca señal como inválida"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=50000.0,
            allocation_pct=0.0001,  # 0.01% de $3000 = $0.30 (muy pequeño)
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is False
        assert result.qty == 0.0  # No se permite orden con qty inválido
        assert result.rejection_reason is not None
        assert "QTY_BELOW_MIN" in result.rejection_reason or "ORDER_VALUE_TOO_SMALL" in result.rejection_reason
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_buy_insufficient_balance(self, helper):
        """Test que balance insuficiente marca señal como inválida"""
        # Crear helper con balance muy bajo
        pm_low = MockPortfolioManager({"BTC": 0.0, "ETH": 0.0, "USDT": 0.0})
        helper_low = PositionSizeCLIHelper(pm_low, min_order_value=2.0)
        
        result = await helper_low.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=50000.0,
            allocation_pct=0.10,
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert "INSUFFICIENT_BALANCE" in result.rejection_reason
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_price(self, helper):
        """Test que precio inválido marca señal como inválida"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=0.0,  # Precio inválido
            allocation_pct=0.10,
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert "INVALID_PRICE" in result.rejection_reason
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_side(self, helper):
        """Test que lado inválido marca señal como inválida"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="invalid",  # Lado inválido
            current_price=50000.0,
            allocation_pct=0.10,
            min_order_value=2.0
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is False
        assert result.rejection_reason is not None
        assert "INVALID_SIDE" in result.rejection_reason
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_custom_balance(self, helper):
        """Test uso de balance custom (override)"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=50000.0,
            allocation_pct=0.10,
            min_order_value=2.0,
            custom_balance=5000.0  # Override del balance
        )
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is True
        assert result.balance_used == 5000.0  # Usó el balance custom
        assert result.qty == 0.01  # $500 / $50,000
    
    @pytest.mark.asyncio
    async def test_batch_calculate(self, helper):
        """Test cálculo batch de múltiples señales"""
        
        class MockSignal:
            def __init__(self, symbol, side, confidence=0.5):
                self.symbol = symbol
                self.side = side
                self.confidence = confidence
        
        signals = [
            MockSignal("BTCUSDT", "buy", 0.10),
            MockSignal("ETHUSDT", "buy", 0.10),
        ]
        
        market_data = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0
        }
        
        results = await helper.batch_calculate(signals, market_data)
        
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert results["BTCUSDT"].is_valid is True
        assert results["ETHUSDT"].is_valid is True
    
    def test_format_result_for_cli(self, helper):
        """Test formateo de resultado para CLI"""
        result = PositionSizeResult(
            is_valid=True,
            qty=0.006,
            qty_raw=0.00612345,
            order_value_usd=300.0,
            balance_used=3000.0,
            allocation_pct=0.10,
            min_qty_threshold=0.00004,
            metadata={
                "symbol": "BTCUSDT",
                "side": "buy",
                "current_price": 50000.0,
                "precision": 6
            }
        )
        
        formatted = helper.format_result_for_cli(result)
        
        assert "POSITION SIZE CALCULATION RESULT" in formatted
        assert "BTCUSDT" in formatted
        assert "VALID" in formatted
        assert "0.006" in formatted
    
    @pytest.mark.asyncio
    async def test_validate_signal_for_order(self, helper):
        """Test validación completa de señal"""
        
        class MockSignal:
            def __init__(self):
                self.symbol = "BTCUSDT"
                self.side = "buy"
                self.confidence = 0.10
                self.metadata = {"strategy": "test"}
        
        signal = MockSignal()
        result = await helper.validate_signal_for_order(signal, 50000.0)
        
        assert isinstance(result, PositionSizeResult)
        assert result.is_valid is True
        assert result.qty > 0
    
    @pytest.mark.asyncio
    async def test_eth_precision_and_rounding(self, helper):
        """Test específico para ETH con su precisión de 5 decimales"""
        result = await helper.calculate_position_size(
            symbol="ETHUSDT",
            side="buy",
            current_price=3000.0,
            allocation_pct=0.10,  # 10% de $3000 = $300
            min_order_value=2.0
        )
        
        assert result.is_valid is True
        # $300 / $3000 = 0.1 ETH
        # Precisión ETH es 5 decimales, pero 0.1 ya está en precisión correcta
        assert result.qty == 0.1
        
        # Verificar que el qty se redondeó correctamente
        assert result.metadata["precision"] == 5
    
    @pytest.mark.asyncio
    async def test_result_to_dict(self, helper):
        """Test conversión de resultado a diccionario"""
        result = await helper.calculate_position_size(
            symbol="BTCUSDT",
            side="buy",
            current_price=50000.0,
            allocation_pct=0.10,
            min_order_value=2.0
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert "is_valid" in data
        assert "qty" in data
        assert "order_value_usd" in data
        assert "metadata" in data


class TestIntegrationWithOrderIntentBuilder:
    """Tests de integración con OrderIntentBuilder"""
    
    @pytest.mark.asyncio
    async def test_helper_integration(self):
        """Test que el helper se integra correctamente con OrderIntentBuilder"""
        from l1_operational.order_intent_builder import OrderIntentBuilder
        from l1_operational.position_manager import PositionManager
        
        # Crear mocks
        mock_pm = MagicMock(spec=PositionManager)
        mock_pm.portfolio = MockPortfolioManager()
        
        # Crear OrderIntentBuilder
        builder = OrderIntentBuilder(
            position_manager=mock_pm,
            config={"MIN_ORDER_USDT": 2.0, "COOLDOWN_SECONDS": 36},
            paper_mode=True
        )
        
        # Verificar que el builder tiene acceso al helper
        assert hasattr(builder, '_calculate_order_quantity_async')
        
        # Crear mock signal
        mock_signal = MagicMock()
        mock_signal.symbol = "BTCUSDT"
        mock_signal.side = "buy"
        mock_signal.confidence = 0.10
        
        # Test async calculation
        result = await builder._calculate_order_quantity_async(
            signal=mock_signal,
            current_price=50000.0,
            position_qty=0.0,
            available_usdt=3000.0
        )
        
        assert isinstance(result, PositionSizeResult)


# =============================================================================
# Test de CLI
# =============================================================================

def test_cli_main():
    """Test ejecución del CLI"""
    from utils.position_size_cli_helper import main
    
    # Ejecutar el main async
    result = asyncio.run(main())
    assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
