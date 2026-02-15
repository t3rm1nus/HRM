"""
SmartCooldownManager - Gestión inteligente de tiempos de enfriamiento para señales de trading

Características:
1. NO bloquea señales L3 con confianza > 0.65
2. Tiempos de cooldown diferenciados por tipo de operación
3. Registro de actividad por símbolo
4. Auto-reset después de tiempo sin actividad
"""

import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from core.logging import logger


class SmartCooldownManager:
    """
    Gestor inteligente de cooldowns que permite excepciones para señales de alta calidad.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa el SmartCooldownManager.
        
        Args:
            config: Configuración del sistema con parámetros de cooldown
        """
        self.config = config
        
        # Configuración de cooldown por defecto
        self.base_cooldown_seconds = config.get("COOLDOWN_SECONDS", 36)  # Reducido de 60 a 36 (3 ciclos de 12s)
        self.high_confidence_threshold = config.get("HIGH_CONFIDENCE_THRESHOLD", 0.65)
        self.inactivity_reset_hours = config.get("INACTIVITY_RESET_HOURS", 6)
        
        # Tiempos de cooldown diferenciados por régimen y confianza (en segundos)
        # TRENDING con confianza >0.6: 2 ciclos (24s)
        # TRENDING con confianza <0.6: 3 ciclos (36s)
        # RANGE: 4 ciclos (48s)
        # Confianza <0.4: 5 ciclos (60s)
        self.cooldown_times = {
            "trending_high": 24,  # 2 ciclos
            "trending_medium": 36,  # 3 ciclos
            "range": 48,  # 4 ciclos
            "low_confidence": 60,  # 5 ciclos
            "buy": config.get("BUY_COOLDOWN_SECONDS", 24),  # Más corto para compras
            "sell": config.get("SELL_COOLDOWN_SECONDS", 36),  # Normal para ventas
            "default": self.base_cooldown_seconds
        }
        
        # Registro de actividad
        self.last_trade_time: Dict[str, float] = {}
        self.last_signal_time: Dict[str, float] = {}
        self.trade_counts: Dict[str, int] = {}
        
        logger.info(f"✅ SmartCooldownManager inicializado")
        logger.info(f"   - High confidence threshold: {self.high_confidence_threshold}")
        logger.info(f"   - Base cooldown: {self.base_cooldown_seconds}s")
        logger.info(f"   - Inactivity reset: {self.inactivity_reset_hours}h")
        logger.info(f"   - Cooldown por régimen: trending_high=24s, trending_medium=36s, range=48s, low_confidence=60s")

    def record_trade(self, symbol: str) -> None:
        """
        Registra una operación ejecutada para un símbolo.
        
        Args:
            symbol: Símbolo de trading (ej: 'BTCUSDT')
        """
        current_time = time.time()
        self.last_trade_time[symbol] = current_time
        self.trade_counts[symbol] = self.trade_counts.get(symbol, 0) + 1
        
        logger.debug(f"[COOLDOWN] Trade recorded for {symbol} at {current_time}")

    def record_signal(self, symbol: str) -> None:
        """
        Registra una señal recibida (aunque no se ejecute).
        
        Args:
            symbol: Símbolo de trading
        """
        self.last_signal_time[symbol] = time.time()

    def should_execute_signal(self, symbol: str, signal_action: str = None, 
                             signal_confidence: float = None, l3_regime: str = None) -> bool:
        """
        Determina si se debe ejecutar una señal basándose en reglas inteligentes.
        
        Reglas:
        1. ✅ Señales L3 con confianza > threshold NO se bloquean
        2. ✅ Cooldown adaptativo por régimen y confianza
        3. ✅ Cooldown más corto para operaciones BUY que SELL
        4. ✅ Reset automático después de inactividad prolongada
        5. ✅ Primera operación del día siempre permitida
        
        Args:
            symbol: Símbolo de trading
            signal_action: Acción de la señal ('buy' o 'sell')
            signal_confidence: Confianza de la señal (0.0-1.0)
            l3_regime: Régimen L3 (trending, range, etc.)
            
        Returns:
            bool: True si se puede ejecutar, False si está en cooldown
        """
        # REGLA 1: Señales de alta confianza (> 0.65) SIEMPRE pasan
        if signal_confidence and signal_confidence > self.high_confidence_threshold:
            logger.info(f"🎯 High confidence signal ({signal_confidence:.2f}) - bypassing cooldown for {symbol}")
            return True
        
        # Obtener último tiempo de trade
        last_trade = self.last_trade_time.get(symbol)
        
        # REGLA 2: Si es la primera operación o no hay registro, permitir
        if last_trade is None:
            return True
        
        # REGLA 3: Reset después de inactividad prolongada
        inactivity_hours = (time.time() - last_trade) / 3600
        if inactivity_hours > self.inactivity_reset_hours:
            logger.info(f"🔄 Cooldown reset for {symbol} after {inactivity_hours:.1f}h inactivity")
            del self.last_trade_time[symbol]  # Reset
            return True
        
        # Calcular cooldown adaptativo por régimen y confianza
        cooldown_seconds = self._get_adaptive_cooldown(signal_confidence, l3_regime, signal_action)
        
        # Verificar si ha pasado el tiempo de cooldown
        time_since_last_trade = time.time() - last_trade
        can_execute = time_since_last_trade >= cooldown_seconds
        
        if not can_execute:
            remaining = cooldown_seconds - time_since_last_trade
            logger.info(f"⏱️ Cooldown active for {symbol}: {remaining:.0f}s remaining (regime: {l3_regime or 'unknown'}, conf: {signal_confidence:.2f})")
        else:
            logger.debug(f"✅ Cooldown OK for {symbol}: {time_since_last_trade:.0f}s since last trade (regime: {l3_regime or 'unknown'}, conf: {signal_confidence:.2f})")
        
        return can_execute
    
    def _get_adaptive_cooldown(self, signal_confidence: float, l3_regime: str, signal_action: str) -> float:
        """
        Obtiene cooldown adaptativo basado en régimen L3 y confianza de señal.
        
        - TRENDING con confianza >0.6: 2 ciclos (24s)
        - TRENDING con confianza <0.6: 3 ciclos (36s)
        - RANGE: 4 ciclos (48s)
        - Confianza <0.4: 5 ciclos (60s)
        """
        # Si no hay información, usar cooldown por defecto
        if signal_confidence is None and l3_regime is None:
            return self.cooldown_times.get(
                signal_action.lower() if signal_action else "default",
                self.cooldown_times["default"]
            )
        
        # Priorizar confianza baja
        if signal_confidence is not None and signal_confidence < 0.4:
            return self.cooldown_times["low_confidence"]
        
        # Si hay régimen L3
        if l3_regime is not None:
            l3_regime = l3_regime.lower()
            
            if "trend" in l3_regime or "trending" in l3_regime:
                if signal_confidence is not None and signal_confidence > 0.6:
                    return self.cooldown_times["trending_high"]
                else:
                    return self.cooldown_times["trending_medium"]
            elif "range" in l3_regime:
                return self.cooldown_times["range"]
        
        # Fallback a cooldown por acción
        return self.cooldown_times.get(
            signal_action.lower() if signal_action else "default",
            self.cooldown_times["default"]
        )

    def get_cooldown_status(self, symbol: str) -> Dict[str, any]:
        """
        Obtiene el estado actual del cooldown para un símbolo.
        
        Args:
            symbol: Símbolo de trading
            
        Returns:
            Dict con información de estado
        """
        last_trade = self.last_trade_time.get(symbol)
        
        if last_trade is None:
            return {
                "symbol": symbol,
                "cooldown_active": False,
                "time_since_last_trade": None,
                "remaining_seconds": 0,
                "trade_count": self.trade_counts.get(symbol, 0),
                "status": "READY"
            }
        
        time_since_last = time.time() - last_trade
        cooldown_seconds = self.cooldown_times["default"]
        remaining = max(0, cooldown_seconds - time_since_last)
        
        return {
            "symbol": symbol,
            "cooldown_active": remaining > 0,
            "time_since_last_trade": time_since_last,
            "remaining_seconds": remaining,
            "trade_count": self.trade_counts.get(symbol, 0),
            "status": "COOLDOWN_ACTIVE" if remaining > 0 else "READY"
        }

    def reset_cooldown(self, symbol: Optional[str] = None) -> None:
        """
        Resetea el cooldown para un símbolo específico o todos.
        
        Args:
            symbol: Símbolo específico a resetear (None para todos)
        """
        if symbol:
            if symbol in self.last_trade_time:
                del self.last_trade_time[symbol]
                logger.info(f"🔄 Cooldown manually reset for {symbol}")
        else:
            self.last_trade_time.clear()
            logger.info("🔄 All cooldowns manually reset")

    def get_summary(self) -> Dict[str, any]:
        """
        Obtiene un resumen del estado de todos los cooldowns.
        
        Returns:
            Dict con resumen de actividad
        """
        now = time.time()
        active_cooldowns = 0
        ready_symbols = []
        
        for symbol, last_time in self.last_trade_time.items():
            time_since = now - last_time
            cooldown_seconds = self.cooldown_times["default"]
            if time_since < cooldown_seconds:
                active_cooldowns += 1
            else:
                ready_symbols.append(symbol)
        
        return {
            "total_symbols_tracked": len(self.last_trade_time),
            "active_cooldowns": active_cooldowns,
            "ready_symbols": ready_symbols,
            "total_trades": sum(self.trade_counts.values()),
            "high_confidence_threshold": self.high_confidence_threshold
        }

    def cleanup_old_records(self, max_age_hours: int = 24) -> int:
        """
        Limpia registros antiguos para mantener la memoria.
        
        Args:
            max_age_hours: Edad máxima en horas antes de eliminar
            
        Returns:
            int: Número de registros eliminados
        """
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        
        removed = 0
        symbols_to_remove = []
        
        for symbol, last_time in self.last_trade_time.items():
            if now - last_time > max_age_seconds:
                symbols_to_remove.append(symbol)
                removed += 1
        
        for symbol in symbols_to_remove:
            del self.last_trade_time[symbol]
            if symbol in self.trade_counts:
                del self.trade_counts[symbol]
        
        if removed > 0:
            logger.info(f"🧹 Cleaned up {removed} old cooldown records (> {max_age_hours}h)")
        
        return removed