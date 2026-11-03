# -*- coding: utf-8 -*-
"""
OrderFlowSignalGenerator - Generador de señales L1 basado EXCLUSIVAMENTE en bid_ask_imbalance.

Basado en análisis de order flow, bid_ask_imbalance es la única métrica
con edge predictivo real (r=0.17 en 1-5 minutos).

LÓGICA DE SEÑAL:
- BUY: imbalance > +0.30 Y se mantiene > +0.20 durante 3+ lecturas consecutivas
- SELL: imbalance < -0.30 Y se mantiene < -0.20 durante 3+ lecturas consecutivas (solo si hay posición larga)
- HOLD: cualquier otro caso
"""

import asyncio
from typing import Dict, Any, Optional, List, Deque
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.logging import logger


class SignalAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ImbalanceTrend(Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"


@dataclass
class SignalResult:
    """Resultado de una señal de order flow"""
    action: str  # 'buy' | 'sell' | 'hold'
    confidence: float  # 0.0 - 1.0
    reason: str
    imbalance_current: float
    imbalance_trend: str
    consecutive_readings: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reason': self.reason,
            'imbalance_current': self.imbalance_current,
            'imbalance_trend': self.imbalance_trend,
            'consecutive_readings': self.consecutive_readings
        }


class OrderFlowSignalGenerator:
    """
    Genera señales L1 basadas EXCLUSIVAMENTE en bid_ask_imbalance.
    
    Integración:
    - Lee datos del OrderFlowCapture via get_current_features(symbol)
    - Mantiene buffer interno de últimas 10 lecturas por símbolo
    - Prioridad sobre L1 clásico (RSI/MACD) cuando confianza > 0.50
    """
    
    # Umbrales de señal
    THRESHOLD_STRONG_BUY = 0.30   # Umbral para iniciar señal BUY
    THRESHOLD_WEAK_BUY = 0.20     # Umbral mínimo para mantener señal BUY
    THRESHOLD_STRONG_SELL = -0.30 # Umbral para iniciar señal SELL
    THRESHOLD_WEAK_SELL = -0.20   # Umbral mínimo para mantener señal SELL
    
    # Configuración de buffer y lecturas
    REQUIRED_CONSECUTIVE_READINGS = 3  # 3 lecturas = ~15 segundos (cada 5s)
    MAX_BUFFER_SIZE = 10
    
    # Máxima confianza para dejar margen a L2/L3
    MAX_CONFIDENCE = 0.80
    CONFIDENCE_SCALING_FACTOR = 0.50  # divisor para calcular confianza
    
    def __init__(self, order_flow_capture=None):
        """
        Inicializa el generador de señales.
        
        Args:
            order_flow_capture: Instancia de OrderFlowCapture (opcional, usa global si None)
        """
        self._order_flow_capture = order_flow_capture
        
        # Buffers internos: {symbol: deque de lecturas}
        self._imbalance_buffers: Dict[str, Deque[float]] = {}
        self._timestamps: Dict[str, Deque[datetime]] = {}
        
        # Estado de posiciones para saber si podemos vender
        self._has_long_position: Dict[str, bool] = {}
        
        # Lock para thread safety
        self._lock = asyncio.Lock()
        
        logger.info("🌊 OrderFlowSignalGenerator inicializado")
        logger.info(f"   Umbral BUY: >{self.THRESHOLD_STRONG_BUY} (mantener >{self.THRESHOLD_WEAK_BUY})")
        logger.info(f"   Umbral SELL: <{self.THRESHOLD_STRONG_SELL} (mantener <{self.THRESHOLD_WEAK_SELL})")
        logger.info(f"   Lecturas consecutivas requeridas: {self.REQUIRED_CONSECUTIVE_READINGS}")
    
    def _get_order_flow_capture(self):
        """Obtiene la instancia de OrderFlowCapture (inyectada o global)"""
        if self._order_flow_capture is not None:
            return self._order_flow_capture
        
        # Intentar obtener la instancia global
        try:
            from l1_operational.order_flow_capture import get_order_flow_capture
            return get_order_flow_capture()
        except ImportError:
            logger.error("❌ No se pudo importar get_order_flow_capture")
            return None
    
    def _get_or_create_buffer(self, symbol: str) -> Deque[float]:
        """Obtiene o crea el buffer de imbalance para un símbolo"""
        if symbol not in self._imbalance_buffers:
            self._imbalance_buffers[symbol] = deque(maxlen=self.MAX_BUFFER_SIZE)
            self._timestamps[symbol] = deque(maxlen=self.MAX_BUFFER_SIZE)
            self._has_long_position[symbol] = False
        return self._imbalance_buffers[symbol]
    
    def _calculate_trend(self, buffer: Deque[float]) -> ImbalanceTrend:
        """Calcula la tendencia del imbalance basado en últimas lecturas"""
        if len(buffer) < 3:
            return ImbalanceTrend.STABLE
        
        # Comparar promedio de últimas 2 vs anteriores
        recent = list(buffer)[-2:]
        previous = list(buffer)[-4:-2] if len(buffer) >= 4 else list(buffer)[:-2]
        
        if not previous:
            return ImbalanceTrend.STABLE
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        diff = recent_avg - previous_avg
        
        if diff > 0.05:
            return ImbalanceTrend.RISING
        elif diff < -0.05:
            return ImbalanceTrend.FALLING
        else:
            return ImbalanceTrend.STABLE
    
    def _count_consecutive_above(self, buffer: Deque[float], threshold: float) -> int:
        """Cuenta lecturas consecutivas por encima de un umbral (desde el final)"""
        count = 0
        for value in reversed(buffer):
            if value > threshold:
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_below(self, buffer: Deque[float], threshold: float) -> int:
        """Cuenta lecturas consecutivas por debajo de un umbral (desde el final)"""
        count = 0
        for value in reversed(buffer):
            if value < threshold:
                count += 1
            else:
                break
        return count
    
    def _calculate_confidence(self, imbalance: float) -> float:
        """
        Calcula la confianza de la señal.
        
        Fórmula: min(|imbalance| / 0.50, 1.0) * 0.80
        Máximo 0.80 para dejar margen a L2/L3.
        """
        abs_imbalance = abs(imbalance)
        normalized = min(abs_imbalance / self.CONFIDENCE_SCALING_FACTOR, 1.0)
        confidence = normalized * self.MAX_CONFIDENCE
        return round(confidence, 4)
    
    def update_position_state(self, symbol: str, has_long_position: bool):
        """
        Actualiza el estado de posición para un símbolo.
        Debe llamarse desde main.py antes de get_signal() para señales SELL.
        
        Args:
            symbol: Símbolo a actualizar
            has_long_position: True si hay posición larga abierta
        """
        self._has_long_position[symbol] = has_long_position
    
    async def get_signal(self, symbol: str) -> SignalResult:
        """
        Genera una señal basada en bid_ask_imbalance.
        
        Args:
            symbol: Símbolo a analizar (ej: 'BTCUSDT')
            
        Returns:
            SignalResult con action, confidence, reason, etc.
        """
        async with self._lock:
            # Obtener datos actuales del OrderFlowCapture
            order_flow = self._get_order_flow_capture()
            
            if order_flow is None:
                logger.warning(f"⚠️ OrderFlowCapture no disponible para {symbol}")
                return SignalResult(
                    action=SignalAction.HOLD.value,
                    confidence=0.0,
                    reason="OrderFlowCapture no disponible",
                    imbalance_current=0.0,
                    imbalance_trend=ImbalanceTrend.STABLE.value
                )
            
            # Obtener features actuales
            features = order_flow.get_current_features(symbol)
            
            if features is None:
                logger.warning(f"⚠️ No hay features disponibles para {symbol}")
                return SignalResult(
                    action=SignalAction.HOLD.value,
                    confidence=0.0,
                    reason="No hay datos de order flow disponibles",
                    imbalance_current=0.0,
                    imbalance_trend=ImbalanceTrend.STABLE.value
                )
            
            # Extraer imbalance actual
            imbalance = features.get('bid_ask_imbalance', 0.0)
            
            # Actualizar buffer
            buffer = self._get_or_create_buffer(symbol)
            buffer.append(imbalance)
            self._timestamps[symbol].append(datetime.utcnow())
            
            # Calcular tendencia
            trend = self._calculate_trend(buffer)
            
            # Calcular confianza base
            confidence = self._calculate_confidence(imbalance)
            
            # Verificar condiciones de señal
            has_long = self._has_long_position.get(symbol, False)
            
            # SEÑAL BUY: imbalance > +0.30 Y se mantiene > +0.20 durante 3+ lecturas
            if imbalance > self.THRESHOLD_STRONG_BUY:
                consecutive = self._count_consecutive_above(buffer, self.THRESHOLD_WEAK_BUY)
                
                if consecutive >= self.REQUIRED_CONSECUTIVE_READINGS:
                    reason = (f"Presión compradora fuerte: imbalance={imbalance:.3f} "
                             f"({consecutive} lecturas consecutivas >{self.THRESHOLD_WEAK_BUY})")
                    
                    logger.info(f"🌊 ORDERFLOW SIGNAL: {symbol} BUY | imbalance={imbalance:.3f} | conf={confidence:.2f}")
                    
                    return SignalResult(
                        action=SignalAction.BUY.value,
                        confidence=confidence,
                        reason=reason,
                        imbalance_current=imbalance,
                        imbalance_trend=trend.value,
                        consecutive_readings=consecutive
                    )
                else:
                    # Cumple umbral fuerte pero no las lecturas consecutivas
                    return SignalResult(
                        action=SignalAction.HOLD.value,
                        confidence=confidence * 0.5,  # Confianza reducida
                        reason=f"Imbalance fuerte ({imbalance:.3f}) pero solo {consecutive}/{self.REQUIRED_CONSECUTIVE_READINGS} lecturas consecutivas",
                        imbalance_current=imbalance,
                        imbalance_trend=trend.value,
                        consecutive_readings=consecutive
                    )
            
            # SEÑAL SELL: imbalance < -0.30 Y se mantiene < -0.20 durante 3+ lecturas Y hay posición larga
            elif imbalance < self.THRESHOLD_STRONG_SELL:
                consecutive = self._count_consecutive_below(buffer, self.THRESHOLD_WEAK_SELL)
                
                if consecutive >= self.REQUIRED_CONSECUTIVE_READINGS:
                    if has_long:
                        reason = (f"Presión vendedora fuerte: imbalance={imbalance:.3f} "
                                 f"({consecutive} lecturas consecutivas <{self.THRESHOLD_WEAK_SELL}) | Cierre de long")
                        
                        logger.info(f"🌊 ORDERFLOW SIGNAL: {symbol} SELL | imbalance={imbalance:.3f} | conf={confidence:.2f}")
                        
                        return SignalResult(
                            action=SignalAction.SELL.value,
                            confidence=confidence,
                            reason=reason,
                            imbalance_current=imbalance,
                            imbalance_trend=trend.value,
                            consecutive_readings=consecutive
                        )
                    else:
                        # No hay posición larga para cerrar
                        return SignalResult(
                            action=SignalAction.HOLD.value,
                            confidence=confidence,
                            reason=f"Presión vendedora fuerte pero no hay posición larga abierta",
                            imbalance_current=imbalance,
                            imbalance_trend=trend.value,
                            consecutive_readings=consecutive
                        )
                else:
                    # Cumple umbral fuerte pero no las lecturas consecutivas
                    return SignalResult(
                        action=SignalAction.HOLD.value,
                        confidence=confidence * 0.5,
                        reason=f"Imbalance débil ({imbalance:.3f}) pero solo {consecutive}/{self.REQUIRED_CONSECUTIVE_READINGS} lecturas consecutivas",
                        imbalance_current=imbalance,
                        imbalance_trend=trend.value,
                        consecutive_readings=consecutive
                    )
            
            # HOLD: Ninguna condición de señal cumplida
            else:
                # Determinar razón según el valor del imbalance
                if abs(imbalance) < 0.10:
                    reason = f"Mercado equilibrado (imbalance={imbalance:.3f})"
                elif imbalance > 0:
                    reason = f"Presión compradora débil (imbalance={imbalance:.3f})"
                else:
                    reason = f"Presión vendedora débil (imbalance={imbalance:.3f})"
                
                return SignalResult(
                    action=SignalAction.HOLD.value,
                    confidence=0.0,
                    reason=reason,
                    imbalance_current=imbalance,
                    imbalance_trend=trend.value,
                    consecutive_readings=0
                )
    
    def get_buffer_stats(self, symbol: str) -> Dict[str, Any]:
        """Retorna estadísticas del buffer para un símbolo (para debugging)"""
        buffer = self._imbalance_buffers.get(symbol, deque())
        
        if not buffer:
            return {
                'symbol': symbol,
                'buffer_size': 0,
                'avg_imbalance': 0.0,
                'min_imbalance': 0.0,
                'max_imbalance': 0.0
            }
        
        values = list(buffer)
        return {
            'symbol': symbol,
            'buffer_size': len(values),
            'avg_imbalance': sum(values) / len(values),
            'min_imbalance': min(values),
            'max_imbalance': max(values),
            'latest_imbalance': values[-1] if values else 0.0
        }


# Instancia global para acceso desde main.py
_order_flow_signal_generator: Optional[OrderFlowSignalGenerator] = None


def get_order_flow_signal_generator() -> Optional[OrderFlowSignalGenerator]:
    """Retorna la instancia global de OrderFlowSignalGenerator"""
    return _order_flow_signal_generator


def set_order_flow_signal_generator(generator: OrderFlowSignalGenerator):
    """Establece la instancia global"""
    global _order_flow_signal_generator
    _order_flow_signal_generator = generator
    logger.info("✅ OrderFlowSignalGenerator registrado globalmente")


def create_order_flow_signal_generator(order_flow_capture=None) -> OrderFlowSignalGenerator:
    """
    Factory function para crear el generador de señales.
    
    Args:
        order_flow_capture: Instancia de OrderFlowCapture (opcional)
        
    Returns:
        OrderFlowSignalGenerator inicializado
    """
    generator = OrderFlowSignalGenerator(order_flow_capture=order_flow_capture)
    set_order_flow_signal_generator(generator)
    return generator
