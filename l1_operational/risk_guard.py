# l1_operational/risk_guard.py
"""Validador de riesgo hard-coded para L1"""

import logging
from typing import Dict, Optional
from .models import Signal, ValidationResult, RiskAlert
from .config import RISK_LIMITS, PORTFOLIO_LIMITS
import time

from core.logging import logger

class RiskGuard:
    """Guardián de riesgo con validaciones hard-coded"""
    
    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_start_balance = 10000.0  # Mock balance inicial
        self.current_positions = {}  # symbol -> position_size
        self.account_balance = 10000.0  # Mock balance
        
    def validate_signal(self, signal: Signal) -> ValidationResult:
        """
        Validación completa de una señal
        Aplica TODAS las reglas de seguridad hard-coded
        """
        logger.info(f"Validating signal {signal.signal_id}")
        
        validations = [
            self._validate_basic_params,
            self._validate_stop_loss,
            self._validate_order_size,
            self._validate_account_balance,
            self._validate_position_limits,
            self._validate_daily_drawdown,
        ]
        
        warnings = []
        for validation_func in validations:
            try:
                result = validation_func(signal)
                if not result.is_valid:
                    logger.warning(f"Validation failed: {result.reason}")
                    return result
                
                if result.warnings:
                    warnings.extend(result.warnings)
                    
            except Exception as e:
                logger.error(f"Validation error in {validation_func.__name__}: {e}")
                return ValidationResult(False, f"Validation error: {str(e)}")
        
        return ValidationResult(True, "All validations passed", warnings=warnings)
    
    def _validate_basic_params(self, signal: Signal) -> ValidationResult:
        """Validación de parámetros básicos"""
        if not signal.signal_id:
            return ValidationResult(False, "Missing signal_id")
        if not signal.symbol:
            return ValidationResult(False, "Missing symbol")
        if signal.side not in ['buy', 'sell']:
            return ValidationResult(False, f"Invalid side: {signal.side}")
        if signal.qty <= 0:
            return ValidationResult(False, f"Invalid quantity: {signal.qty}")
        if signal.order_type not in ['market', 'limit']:
            return ValidationResult(False, f"Invalid order_type: {signal.order_type}")
        if signal.order_type == 'limit' and not signal.price:
            return ValidationResult(False, "Limit order requires price")
        return ValidationResult(True, "Basic parameters valid")
    
    def _validate_stop_loss(self, signal: Signal) -> ValidationResult:
        """Validación de stop loss obligatorio"""
        if not signal.stop_loss:
            return ValidationResult(False, "Stop loss is mandatory")
        if signal.side == 'buy' and signal.stop_loss >= (signal.price or 50000):
            return ValidationResult(False, "Buy stop loss must be below entry price")
        if signal.side == 'sell' and signal.stop_loss <= (signal.price or 50000):
            return ValidationResult(False, "Sell stop loss must be above entry price")
        return ValidationResult(True, "Stop loss valid")
    
    def _validate_order_size(self, signal: Signal) -> ValidationResult:
        """Validación de tamaño de orden"""
        symbol_base = signal.symbol.split('/')[0] if '/' in signal.symbol else signal.symbol
        
        if symbol_base == 'BTC':
            if signal.qty > RISK_LIMITS["MAX_ORDER_SIZE_BTC"]:
                return ValidationResult(False, f"BTC order too large: {signal.qty} > {RISK_LIMITS['MAX_ORDER_SIZE_BTC']}")
        elif symbol_base == 'ETH':
            if signal.qty > RISK_LIMITS["MAX_ORDER_SIZE_ETH"]:
                return ValidationResult(False, f"ETH order too large: {signal.qty} > {RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")
        
        estimated_usdt_value = signal.qty * (signal.price or (50000 if symbol_base == 'BTC' else 3000))
        if estimated_usdt_value > RISK_LIMITS["MAX_ORDER_SIZE_USDT"]:
            return ValidationResult(False, f"Order value too large: ${estimated_usdt_value:.2f} > ${RISK_LIMITS['MAX_ORDER_SIZE_USDT']}")
        if estimated_usdt_value < RISK_LIMITS["MIN_ORDER_SIZE_USDT"]:
            return ValidationResult(False, f"Order value too small: ${estimated_usdt_value:.2f} < ${RISK_LIMITS['MIN_ORDER_SIZE_USDT']}")
        return ValidationResult(True, "Order size valid")
    
    def _validate_account_balance(self, signal: Signal) -> ValidationResult:
        """Validación de balance de cuenta"""
        if self.account_balance < PORTFOLIO_LIMITS["MIN_ACCOUNT_BALANCE_USDT"]:
            return ValidationResult(False, f"Insufficient account balance: ${self.account_balance}")
        estimated_cost = signal.qty * (signal.price or (50000 if "BTC" in signal.symbol else 3000))
        if estimated_cost > self.account_balance * 0.9:
            return ValidationResult(False, "Order cost too high relative to balance")
        return ValidationResult(True, "Account balance sufficient")
    
    def _validate_position_limits(self, signal: Signal) -> ValidationResult:
        """Validación de límites de posición"""
        symbol_base = signal.symbol.split('/')[0] if '/' in signal.symbol else signal.symbol
        current_position = self.current_positions.get(symbol_base, 0.0)
        position_change = signal.qty if signal.side == 'buy' else -signal.qty
        new_position = abs(current_position + position_change)
        
        if symbol_base == 'BTC':
            max_exposure = PORTFOLIO_LIMITS["MAX_PORTFOLIO_EXPOSURE_BTC"] * self.account_balance / 50000
            if new_position > max_exposure:
                return ValidationResult(False, f"BTC position limit exceeded: {new_position} > {max_exposure}")
        elif symbol_base == 'ETH':
            max_exposure = PORTFOLIO_LIMITS["MAX_PORTFOLIO_EXPOSURE_ETH"] * self.account_balance / 3000
            if new_position > max_exposure:
                return ValidationResult(False, f"ETH position limit exceeded: {new_position} > {max_exposure}")
        return ValidationResult(True, "Position limits OK")
    
    def _validate_daily_drawdown(self, signal: Signal) -> ValidationResult:
        """Validación de drawdown diario"""
        daily_drawdown = (self.daily_start_balance - (self.daily_start_balance + self.daily_pnl)) / self.daily_start_balance
        if daily_drawdown > PORTFOLIO_LIMITS["MAX_DAILY_DRAWDOWN"]:
            return ValidationResult(False, f"Daily drawdown limit exceeded: {daily_drawdown:.2%}")
        warnings = []
        if daily_drawdown > PORTFOLIO_LIMITS["MAX_DAILY_DRAWDOWN"] * 0.8:
            warnings.append(f"Approaching daily drawdown limit: {daily_drawdown:.2%}")
        return ValidationResult(True, "Daily drawdown OK", warnings=warnings)
    
    def update_position(self, symbol: str, qty_change: float):
        """Actualiza posición después de ejecución"""
        symbol_base = symbol.split('/')[0] if '/' in symbol else symbol
        self.current_positions[symbol_base] = self.current_positions.get(symbol_base, 0.0) + qty_change
        logger.info(f"Position updated: {symbol_base} = {self.current_positions[symbol_base]}")
    
    def update_daily_pnl(self, pnl_change: float):
        """Actualiza PnL diario"""
        self.daily_pnl += pnl_change
        logger.info(f"Daily PnL updated: {self.daily_pnl}")
    
    def reset_daily_metrics(self):
        """Reset métricas diarias (llamar cada día)"""
        self.daily_pnl = 0.0
        self.daily_start_balance = self.account_balance
        logger.info("Daily metrics reset")
