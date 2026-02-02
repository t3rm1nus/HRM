# core/incremental_signal_verifier.py - Incremental Signal Verification System
import asyncio
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from core.logging import logger
from l2_tactic.models import TacticalSignal
import logging

@dataclass
class SignalVerificationResult:
    """Result of signal verification"""
    signal_id: str
    symbol: str
    original_signal: str
    verified_signal: str
    confidence_change: float
    latency_ms: float
    market_conditions: Dict[str, Any]
    verification_timestamp: datetime
    is_valid: bool
    reason: str



class IncrementalSignalVerifier:
    """
    Verificador incremental de señales con manejo especial para HOLD.

    HOLD signals son NEUTRALES - no pasan por validación binaria VALID/INVALID.
    HOLD no contamina métricas ni rejection counters.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.verified_signals = {}
        self.hold_signals = set()  # Track HOLD signals separately
        self.logger = logging.getLogger(__name__)

    def verify_signal(self, signal: Dict) -> bool:
        """
        Verificar señal con lógica especial para HOLD.

        HOLD = neutral_state - NO pasa por validación binaria.
        HOLD no incrementa intent_count ni rejection counters.
        Solo BUY/SELL signals pasan por validación normal.
        """
        signal_id = signal.get('signal_id', 'unknown')
        action = signal.get('action', '').lower()
        confidence = signal.get('confidence', 0.0)

        # HOLD = neutral_state - NO validación binaria, NO métricas contaminadas
        if action == 'hold':
            self.logger.debug(f"😐 HOLD Signal {signal_id} - neutral state (no validation, no metrics)")
            # Track HOLD signals separately (no contaminar verified_signals)
            self.hold_signals.add(signal_id)
            return True  # HOLD es aceptado como estado neutral

        # BUY/SELL signals pasan por validación binaria normal
        if confidence < self.min_confidence:
            self.logger.info(f"❌ INVALID Signal {signal_id}: Confidence too low: {confidence}")
            self.verified_signals[signal_id] = False
            return False

        # Otras validaciones para BUY/SELL
        if not self._validate_signal_structure(signal):
            self.logger.info(f"❌ INVALID Signal {signal_id}: Invalid structure")
            self.verified_signals[signal_id] = False
            return False

        self.logger.info(f"✅ Signal {signal_id} verified successfully (confidence: {confidence})")
        self.verified_signals[signal_id] = True
        return True

    def _validate_signal_structure(self, signal: Dict) -> bool:
        """Validar estructura básica de señal BUY/SELL."""
        required_fields = ['action', 'symbol', 'confidence']
        return all(field in signal for field in required_fields)

    def get_verified_signals(self) -> Dict[str, bool]:
        """Obtener diccionario de señales verificadas."""
        return self.verified_signals.copy()

    def clear_verified_signals(self):
        """Limpiar el registro de señales verificadas."""
        self.verified_signals.clear()
        self.hold_signals.clear()

    def get_verification_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas de verificación limpias - HOLD no contamina métricas.

        Returns:
            Dict con métricas que excluyen HOLD signals
        """
        total_verified = len(self.verified_signals)
        total_valid = sum(1 for v in self.verified_signals.values() if v)
        total_invalid = total_verified - total_valid
        total_hold = len(self.hold_signals)

        # Métricas limpias: solo BUY/SELL signals
        if total_verified > 0:
            success_rate = total_valid / total_verified
        else:
            success_rate = 0.0

        return {
            'total_signals': total_verified + total_hold,  # Total incluyendo HOLD
            'total_verified': total_verified,  # Solo BUY/SELL verificados
            'total_valid': total_valid,       # BUY/SELL válidos
            'total_invalid': total_invalid,   # BUY/SELL inválidos
            'total_hold': total_hold,         # HOLD signals (neutrales)
            'success_rate': success_rate,     # Solo BUY/SELL
            'hold_rate': total_hold / (total_verified + total_hold) if (total_verified + total_hold) > 0 else 0.0
        }

    async def submit_signal_for_verification(self, signal: TacticalSignal, market_data: Dict[str, Any]) -> Optional[SignalVerificationResult]:
        """
        Submit a signal for incremental verification.

        Args:
            signal: TacticalSignal object to verify
            market_data: Market data for verification context

        Returns:
            SignalVerificationResult if verification completes, None if async processing
        """
        try:
            # Convert TacticalSignal to dict for verification
            signal_dict = {
                'signal_id': f"{getattr(signal, 'symbol', 'UNKNOWN')}_{getattr(signal, 'side', 'hold')}_{int(time.time() * 1000)}",
                'action': getattr(signal, 'side', 'hold'),  # Map 'side' to 'action' for compatibility
                'symbol': getattr(signal, 'symbol', 'UNKNOWN'),
                'confidence': getattr(signal, 'confidence', 0.5),
                'strength': getattr(signal, 'strength', 0.5),
                'side': getattr(signal, 'side', 'hold')
            }

            # Verify the signal
            is_valid = self.verify_signal(signal_dict)

            # Create verification result
            result = SignalVerificationResult(
                signal_id=signal_dict['signal_id'],
                symbol=signal_dict['symbol'],
                original_signal=getattr(signal, 'side', 'hold'),
                verified_signal=signal_dict['action'] if is_valid else 'rejected',
                confidence_change=0.0,  # No confidence adjustment for now
                latency_ms=0.0,  # Synchronous for now
                market_conditions=self._extract_market_conditions(market_data),
                verification_timestamp=datetime.now(),
                is_valid=is_valid,
                reason="HOLD signals auto-approved" if signal_dict['action'].lower() == 'hold' else f"Confidence check: {signal_dict['confidence']}"
            )

            self.logger.info(f"✅ Signal verification result: {result.signal_id} -> {'VALID' if result.is_valid else 'INVALID'}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error in signal verification: {e}")
            return None

    def _extract_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic market conditions for verification context."""
        try:
            conditions = {}
            for symbol, data in market_data.items():
                if isinstance(data, pd.DataFrame) and len(data) > 0:
                    current_price = data['close'].iloc[-1] if 'close' in data.columns else 0.0
                    conditions[symbol] = {
                        'price': float(current_price),
                        'timestamp': datetime.now().isoformat()
                    }
                elif isinstance(data, dict) and 'close' in data:
                    conditions[symbol] = {
                        'price': float(data['close']),
                        'timestamp': datetime.now().isoformat()
                    }
            return conditions
        except Exception as e:
            self.logger.error(f"Error extracting market conditions: {e}")
            return {}

    async def start_verification_loop(self):
        """Start the verification processing loop (placeholder for future async processing)."""
        self.logger.info("Signal verification loop started (placeholder)")

    async def stop_verification_loop(self):
        """Stop the verification processing loop."""
        self.logger.info("Signal verification loop stopped")


# Global verifier instance
_verifier_instance: Optional['IncrementalSignalVerifier'] = None

def get_signal_verifier() -> IncrementalSignalVerifier:
    """Get global signal verifier instance"""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = IncrementalSignalVerifier()
    return _verifier_instance

async def start_signal_verification():
    """Start the global signal verification system"""
    verifier = get_signal_verifier()
    await verifier.start_verification_loop()

async def stop_signal_verification():
    """Stop the global signal verification system"""
    verifier = get_signal_verifier()
    await verifier.stop_verification_loop()
