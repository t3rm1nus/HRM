"""
Tests unitarios para ErrorRecoveryManager

Valida que el gestor de recuperación de errores maneje correctamente
todos los tipos de error sin crash y con límites de retry adecuados.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

from system.error_recovery_manager import ErrorRecoveryManager, ErrorType, RecoveryActionType
from system.models import RecoveryAction
from core.exceptions import HRMException


class TestErrorRecoveryManager:
    """Suite de tests para ErrorRecoveryManager."""

    def setup_method(self):
        """Configura el gestor de recovery para cada test."""
        self.recovery_manager = ErrorRecoveryManager()

    def test_classify_error_data_quality(self):
        """Test clasificación de errores de calidad de datos."""
        # Errores de calidad de datos
        data_errors = [
            ValueError("market_data is empty"),
            RuntimeError("No hay market_data en el estado"),
            ValueError("Empty market data"),
            RuntimeError("Data validation failed")
        ]
        
        for error in data_errors:
            error_type = self.recovery_manager.classify_error(error)
            assert error_type == ErrorType.DATA_QUALITY, f"Error {error} no clasificado como DATA_QUALITY"

    def test_classify_error_ml_framework(self):
        """Test clasificación de errores de frameworks ML."""
        # Errores de frameworks ML
        ml_errors = [
            RuntimeError("TensorFlow session failed"),
            RuntimeError("PyTorch CUDA error"),
            RuntimeError("truth value of a DataFrame is ambiguous"),
            RuntimeError("Keras model loading failed")
        ]
        
        for error in ml_errors:
            error_type = self.recovery_manager.classify_error(error)
            # truth value of DataFrame debe clasificarse como DATA_QUALITY, no ML_FRAMEWORK
            if "truth value of a DataFrame" in str(error):
                assert error_type == ErrorType.DATA_QUALITY, f"Error {error} debe clasificarse como DATA_QUALITY"
            else:
                assert error_type == ErrorType.ML_FRAMEWORK, f"Error {error} no clasificado como ML_FRAMEWORK"

    def test_classify_error_state_corruption(self):
        """Test clasificación de errores de corrupción de estado."""
        # Errores de corrupción de estado
        state_errors = [
            KeyError("state not found"),
            AttributeError("state has no attribute"),
            KeyError("missing key in state"),
            AttributeError("invalid attribute access")
        ]
        
        for error in state_errors:
            error_type = self.recovery_manager.classify_error(error)
            assert error_type == ErrorType.STATE_CORRUPTION, f"Error {error} no clasificado como STATE_CORRUPTION"

    def test_classify_error_network(self):
        """Test clasificación de errores de red."""
        # Errores de red
        network_errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Request timeout"),
            ConnectionError("API connection lost"),
            TimeoutError("Network timeout")
        ]
        
        for error in network_errors:
            error_type = self.recovery_manager.classify_error(error)
            assert error_type == ErrorType.NETWORK, f"Error {error} no clasificado como NETWORK"

    def test_classify_error_unknown(self):
        """Test clasificación de errores desconocidos."""
        # Errores desconocidos - deben ser genéricos y no coincidir con otros tipos
        unknown_errors = [
            Exception("Some error"),
            ValueError("Invalid operation"),
            TypeError("Type conversion error"),
            IndexError("Index out of range")
        ]
        
        for error in unknown_errors:
            error_type = self.recovery_manager.classify_error(error)
            assert error_type == ErrorType.UNKNOWN, f"Error {error} no clasificado como UNKNOWN"

    def test_should_retry_data_quality(self):
        """Test límites de retry para errores de calidad de datos."""
        error = ValueError("market_data is empty")
        
        # Debe permitir retry hasta el límite
        for i in range(3):
            should_retry = self.recovery_manager.should_retry(error)
            assert should_retry == True, f"Debería permitir retry en intento {i+1}"
            # Simular que el error ocurre de nuevo
            self.recovery_manager.error_counts[f"data_quality_ValueError"] = i + 1
        
        # Después del límite, no debe permitir retry
        should_retry = self.recovery_manager.should_retry(error)
        assert should_retry == False, "No debería permitir retry después del límite"

    def test_should_retry_network(self):
        """Test límites de retry para errores de red."""
        error = ConnectionError("Connection failed")
        
        # Debe permitir retry hasta el límite (5 intentos)
        for i in range(5):
            should_retry = self.recovery_manager.should_retry(error)
            assert should_retry == True, f"Debería permitir retry en intento {i+1}"
            # Simular que el error ocurre de nuevo
            self.recovery_manager.error_counts[f"network_ConnectionError"] = i + 1
        
        # Después del límite, no debe permitir retry
        should_retry = self.recovery_manager.should_retry(error)
        assert should_retry == False, "No debería permitir retry después del límite"

    def test_get_recovery_wait_time_data_quality(self):
        """Test tiempos de espera para errores de calidad de datos."""
        error = ValueError("market_data is empty")
        wait_time = self.recovery_manager.get_recovery_wait_time(error)
        assert wait_time == 5, f"Tiempo de espera incorrecto para DATA_QUALITY: {wait_time}"

    def test_get_recovery_wait_time_network_with_backoff(self):
        """Test backoff exponencial para errores de red."""
        error = ConnectionError("Connection failed")
        
        # Primer intento
        self.recovery_manager.error_counts["network_ConnectionError"] = 0
        wait_time = self.recovery_manager.get_recovery_wait_time(error)
        assert wait_time == 30, f"Tiempo de espera incorrecto para primer intento: {wait_time}"
        
        # Segundo intento (30 * 2^1 = 60)
        self.recovery_manager.error_counts["network_ConnectionError"] = 1
        wait_time = self.recovery_manager.get_recovery_wait_time(error)
        assert wait_time == 60, f"Tiempo de espera incorrecto para segundo intento: {wait_time}"
        
        # Tercer intento (30 * 2^2 = 120)
        self.recovery_manager.error_counts["network_ConnectionError"] = 2
        wait_time = self.recovery_manager.get_recovery_wait_time(error)
        assert wait_time == 120, f"Tiempo de espera incorrecto para tercer intento: {wait_time}"

    def test_get_recovery_wait_time_max_limit(self):
        """Test límite máximo de tiempo de espera para errores de red."""
        error = ConnectionError("Connection failed")
        
        # Muchos intentos (30 * 2^10 = 30720, pero debe estar limitado a 300)
        self.recovery_manager.error_counts["network_ConnectionError"] = 10
        wait_time = self.recovery_manager.get_recovery_wait_time(error)
        assert wait_time == 300, f"Tiempo de espera debe estar limitado a 300s: {wait_time}"

    @pytest.mark.asyncio
    async def test_handle_cycle_error_data_quality_success(self):
        """Test manejo de error de calidad de datos con recovery exitoso."""
        error = ValueError("market_data is empty")
        state = {"market_data": {}}
        
        # Mock para simular recovery exitoso
        with patch.object(self.recovery_manager, 'recover_from_data_error', return_value=AsyncMock(return_value=True)):
            recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)
            
            assert recovery_action.action == RecoveryActionType.RETRY
            assert recovery_action.success == True
            assert recovery_action.wait_seconds == 5

    @pytest.mark.asyncio
    async def test_handle_cycle_error_data_quality_failure(self):
        """Test manejo de error de calidad de datos con recovery fallido."""
        error = ValueError("market_data is empty")
        state = {"market_data": {}}

        # Mock para simular recovery fallido
        with patch.object(self.recovery_manager, 'recover_from_data_error', return_value=AsyncMock(return_value=False)):
            recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)

            assert recovery_action.action == RecoveryActionType.SKIP_CYCLE
            assert recovery_action.success == False

    @pytest.mark.asyncio
    async def test_handle_cycle_error_ml_framework(self):
        """Test manejo de error de frameworks ML."""
        error = RuntimeError("TensorFlow session failed")
        state = {"market_data": {}}
        
        # Mock para simular recovery exitoso
        with patch.object(self.recovery_manager, 'recover_from_ml_framework_error', return_value=AsyncMock(return_value=True)):
            recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)
            
            assert recovery_action.action == RecoveryActionType.RETRY
            assert recovery_action.success == True
            assert recovery_action.wait_seconds == 10  # Tiempo fijo para ML

    @pytest.mark.asyncio
    async def test_handle_cycle_error_network_max_retries(self):
        """Test manejo de error de red con máximo de retries."""
        error = ConnectionError("Connection failed")
        state = {"market_data": {}}
        
        # Simular que ya se alcanzó el máximo de retries
        self.recovery_manager.error_counts["network_ConnectionError"] = 5
        
        recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)
        
        assert recovery_action.action == RecoveryActionType.SKIP_CYCLE
        assert recovery_action.success == False

    @pytest.mark.asyncio
    async def test_handle_cycle_error_unknown(self):
        """Test manejo de error desconocido."""
        error = Exception("Unknown error")
        state = {"market_data": {}}
        
        recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)
        
        assert recovery_action.action == RecoveryActionType.SKIP_CYCLE
        assert recovery_action.wait_seconds == 30
        assert recovery_action.success == False

    @pytest.mark.asyncio
    async def test_recover_from_data_error_success(self):
        """Test recovery exitoso de errores de datos."""
        # Mock para simular éxito en obtención y sanitización de datos
        with patch.object(self.recovery_manager, '_get_fresh_market_data', return_value=AsyncMock(return_value={"BTCUSDT": {"close": 50000.0}})):
            with patch('system.error_recovery_manager.sanitize_market_data', return_value={"BTCUSDT": {"close": 50000.0}}):
                result = await self.recovery_manager.recover_from_data_error()
                assert result == True

    @pytest.mark.asyncio
    async def test_recover_from_data_error_failure(self):
        """Test recovery fallido de errores de datos."""
        # Mock para simular fallo en obtención de datos
        with patch.object(self.recovery_manager, '_get_fresh_market_data', return_value=AsyncMock(return_value=None)):
            result = await self.recovery_manager.recover_from_data_error()
            assert result == False

    @pytest.mark.asyncio
    async def test_recover_from_ml_framework_error_success(self):
        """Test recovery exitoso de errores de frameworks ML."""
        # Mock para simular éxito en limpieza y re-inicialización
        with patch.object(self.recovery_manager, '_cleanup_ml_resources', return_value=AsyncMock()):
            with patch.object(self.recovery_manager, '_reinitialize_ml_frameworks', return_value=AsyncMock(return_value=True)):
                result = await self.recovery_manager.recover_from_ml_framework_error()
                assert result == True

    @pytest.mark.asyncio
    async def test_recover_from_ml_framework_error_failure(self):
        """Test recovery fallido de errores de frameworks ML."""
        # Mock para simular fallo en re-inicialización
        with patch.object(self.recovery_manager, '_cleanup_ml_resources', return_value=AsyncMock()):
            with patch.object(self.recovery_manager, '_reinitialize_ml_frameworks', return_value=AsyncMock(return_value=False)):
                result = await self.recovery_manager.recover_from_ml_framework_error()
                assert result == False

    @pytest.mark.asyncio
    async def test_recover_from_state_corruption_success(self):
        """Test recovery exitoso de errores de corrupción de estado."""
        # Mock para simular éxito en validación y reparación
        with patch.object(self.recovery_manager, '_validate_state_structure', return_value=True):
            result = await self.recovery_manager.recover_from_state_corruption()
            assert result == True

    @pytest.mark.asyncio
    async def test_recover_from_state_corruption_failure(self):
        """Test recovery fallido de errores de corrupción de estado."""
        # Mock para simular fallo en validación y reparación
        with patch.object(self.recovery_manager, '_validate_state_structure', return_value=False):
            with patch.object(self.recovery_manager, '_repair_state', return_value=AsyncMock(return_value=False)):
                result = await self.recovery_manager.recover_from_state_corruption()
                assert result == False

    def test_reset_error_counters(self):
        """Test reinicio de contadores de errores."""
        # Simular algunos errores
        self.recovery_manager.error_counts["data_quality_ValueError"] = 2
        self.recovery_manager.error_counts["network_ConnectionError"] = 3
        
        # Reiniciar contadores de un tipo específico
        self.recovery_manager.reset_error_counters(ErrorType.DATA_QUALITY)
        
        # Verificar que el contador se reinició a 0, no que se eliminó
        assert self.recovery_manager.error_counts["data_quality_ValueError"] == 0
        assert self.recovery_manager.error_counts["network_ConnectionError"] == 3
        
        # Reiniciar todos los contadores
        self.recovery_manager.reset_error_counters()
        assert len(self.recovery_manager.error_counts) == 0

    def test_get_error_statistics(self):
        """Test obtención de estadísticas de errores."""
        # Simular algunos errores
        self.recovery_manager.error_counts["data_quality_ValueError"] = 2
        self.recovery_manager.error_counts["network_ConnectionError"] = 1
        
        stats = self.recovery_manager.get_error_statistics()
        
        assert "error_counts" in stats
        assert "data_quality_ValueError" in stats["error_counts"]
        assert "network_ConnectionError" in stats["error_counts"]
        assert "max_retries_config" in stats
        assert "base_wait_times" in stats

    @pytest.mark.asyncio
    async def test_handle_cycle_error_with_exception_during_recovery(self):
        """Test manejo de excepciones durante el proceso de recovery."""
        error = ValueError("market_data is empty")
        state = {"market_data": {}}
        
        # Mock que lanza excepción durante el recovery
        with patch.object(self.recovery_manager, 'recover_from_data_error', side_effect=Exception("Recovery failed")):
            recovery_action = await self.recovery_manager.handle_cycle_error(error, state, 1)
            
            assert recovery_action.action == RecoveryActionType.SHUTDOWN
            assert recovery_action.success == False
            assert recovery_action.wait_seconds == 60

    def test_error_handling_without_crash(self):
        """Test que el sistema no crashea con cualquier tipo de error."""
        test_errors = [
            ValueError("Test error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
            AttributeError("Attribute error"),
            ConnectionError("Connection error"),
            TimeoutError("Timeout error"),
            Exception("Generic exception"),
            TypeError("Type error"),
            IndexError("Index error"),
            OSError("OS error")
        ]
        
        for error in test_errors:
            try:
                error_type = self.recovery_manager.classify_error(error)
                should_retry = self.recovery_manager.should_retry(error)
                wait_time = self.recovery_manager.get_recovery_wait_time(error)
                
                # Verificar que los métodos no crashen
                assert error_type is not None
                assert isinstance(should_retry, bool)
                assert isinstance(wait_time, int)
                
            except Exception as e:
                pytest.fail(f"El método clasify_error crashó con error: {e}")


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"])