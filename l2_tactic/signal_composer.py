# l2_tactic/signal_composer.py
from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from .config import *  # Import L2 config settings
from .models import TacticalSignal
from datetime import datetime
from core.logging import logger
logger.info("l2_tactic.signal_composer")
import pandas as pd

class SignalComposer:
    """
    Combina señales tácticas de múltiples fuentes (IA, técnicas, patrones).
    - Weighted voting / score average por símbolo+lado
    - Resolución de conflictos BUY vs SELL en el mismo símbolo
    - Ajuste dinámico de pesos por performance histórica (inyectada vía metrics)
    """

    def __init__(self, config: dict, metrics: Optional[object] = None):
        self.cfg = config
        self.metrics = metrics

        # Valores por defecto si no existen en el config
        config_defaults = {
            "ai_model_weight": 0.50,
            "technical_weight": 0.30,
            "pattern_weight": 0.20,
            "min_signal_strength": 0.10,
            "conflict_tie_threshold": 0.05,
            "keep_both_when_far": False
        }

        # Configuración de señales por defecto
        signals_defaults = {
            "min_confidence": 0.3
        }

        # Pesos base por fuente
        self.w_ai = config.get("ai_model_weight", config_defaults["ai_model_weight"])
        self.w_tech = config.get("technical_weight", config_defaults["technical_weight"])
        self.w_pattern = config.get("pattern_weight", config_defaults["pattern_weight"])

        # Filtros de calidad / mínimos para aceptar la señal compuesta
        signals_config = config.get("signals", {})
        self.min_conf = signals_config.get("min_confidence", signals_defaults["min_confidence"])
        self.min_strength = config.get("min_signal_strength", config_defaults["min_signal_strength"])

        # Ajuste de umbrales de conflicto para ser más permisivos con señales fuertes
        self.conflict_tie_threshold = config.get("conflict_tie_threshold", config_defaults["conflict_tie_threshold"]) * 0.8
        self.keep_both_when_far = True  # Forzar keep_both para señales con alta confianza
        self.high_conf_threshold = 0.7   # Nueva: umbral para señales de alta confianza

    def _process_signal_dict(self, signal_dict: dict) -> Optional[TacticalSignal]:
        """Helper to process dictionary signals"""
        try:
            # Ensure numeric values are properly converted
            if 'strength' in signal_dict:
                signal_dict['strength'] = float(signal_dict['strength'])
            if 'confidence' in signal_dict:
                signal_dict['confidence'] = float(signal_dict['confidence'])
            
            # Convert features to proper types
            if 'features' in signal_dict and isinstance(signal_dict['features'], dict):
                features = {}
                for k, v in signal_dict['features'].items():
                    if isinstance(v, (int, float)):
                        features[k] = float(v)
                signal_dict['features'] = features
            
            return TacticalSignal(**signal_dict)
        except Exception as e:
            logger.error(f"❌ Error procesando señal dict: {e}")
            return None

    # --- MÉTODO CORREGIDO ---
    def compose(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        if not signals:
            logger.warning("⚠️ No hay señales para componer")
            return []

        # Log incoming signals for debugging
        logger.debug(f"🔄 Señales entrantes: {len(signals)}")
        for s in signals:
            logger.debug(f"  - {s.symbol} {s.side}: strength={s.strength:.3f} conf={s.confidence:.3f} source={s.source}")

        signals_by_symbol = {}
        for signal in signals:
            # Acepta dicts o TacticalSignal
            if isinstance(signal, dict):
                signal = self._process_signal_dict(signal)
                if signal is None:
                    continue
            elif not isinstance(signal, TacticalSignal):
                logger.error(f"❌ Señal no reconocida: {signal}")
                continue

            # Asignar source por defecto si no existe
            if not hasattr(signal, 'source') or signal.source is None:
                signal.source = 'unknown'

            if not hasattr(signal, 'symbol') or not hasattr(signal, 'side'):
                logger.error(f"❌ Señal inválida: {signal}")
                continue

            symbol = signal.symbol
            signals_by_symbol.setdefault(symbol, []).append(signal)

        composed_signals = []
        for symbol, sym_signals in signals_by_symbol.items():
            total_weight = 0.0
            weighted_strength = 0.0
            weighted_confidence = 0.0
            features = {}

            for signal in sym_signals:
                weight = self._get_dynamic_weight(signal)
                total_weight += weight
                weighted_strength += signal.strength * weight
                weighted_confidence += signal.confidence * weight
                if hasattr(signal, "features") and signal.features:
                    features.update(signal.features)

            if total_weight > 0:
                avg_strength = weighted_strength / total_weight
                avg_confidence = weighted_confidence / total_weight
                dominant_signal = max(sym_signals, key=lambda s: self._get_dynamic_weight(s))

                # Create TacticalSignal with all required fields
                # Get current price from market data or features
                current_price = None
                if hasattr(dominant_signal, 'price') and dominant_signal.price:
                    current_price = dominant_signal.price
                elif 'close' in features:
                    current_price = features['close']
                
                # Get latest price from features
                current_price = None
                if 'close' in features:
                    current_price = features['close']
                elif hasattr(dominant_signal, 'price') and dominant_signal.price:
                    current_price = dominant_signal.price
                
                composed_signal = TacticalSignal(
                    symbol=symbol,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    side=dominant_signal.side,
                    type="market",
                    signal_type='tactical',
                    features=features,  # Include all collected features
                    timestamp=pd.Timestamp.now(),
                    source='composed',
                    price=current_price,
                    quantity=0.0,  # Let order manager calculate based on portfolio
                    stop_loss=None,  # Risk overlay should set this
                    take_profit=None,  # Risk overlay should set this
                    metadata={
                        'composed_from': [s.source for s in sym_signals],
                        'technical_indicators': {k: v for k, v in features.items() 
                                              if isinstance(v, (int, float))},
                        'source': 'composed',
                        'signal_type': dominant_signal.side
                    }
                )
                composed_signals.append(composed_signal)
                logger.debug(f"✅ Señal compuesta para {symbol}: side={dominant_signal.side}, strength={avg_strength:.3f}, confidence={avg_confidence:.3f}")

        # Resolver conflictos y filtrar por thresholds
        filtered_signals = self._resolve_conflicts_and_filter(composed_signals)
        
        # Convert to order format
        final_signals = []
        for signal in filtered_signals:
            final_signals.append(signal.to_order_signal())
            
        logger.info(f"✅ Señales compuestas generadas: {len(final_signals)}")
        return final_signals

    # --- Métodos auxiliares ---
    def _get_dynamic_weight(self, signal: TacticalSignal) -> float:
        logger.debug(f"Calculando peso para señal: source={signal.source}, confidence={signal.confidence}")
        base_weight = 1.0
        if signal.source == 'ai':
            base_weight *= 1.5
            if signal.side == 'hold':  # Baja peso para holds
                base_weight *= 0.2
        elif signal.source == 'technical':
            base_weight *= 1.0
        elif signal.source == 'risk':
            base_weight *= 2.0
        weight = base_weight * signal.confidence
        return max(weight, 0.01)

    def _resolve_conflicts_and_filter(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        """
        Resuelve conflictos BUY vs SELL y filtra por umbrales de calidad.
        """
        signals_by_symbol = {}
        for s in signals:
            if s and s.confidence >= self.min_conf and s.strength >= self.min_strength:
                signals_by_symbol.setdefault(s.symbol, {}).setdefault(s.side, []).append(s)  # Usar listas para múltiples señales por side

        final = []
        for symbol, signals_in_conflict in signals_by_symbol.items():
            # Promediar señales por side
            for side in ['buy', 'sell', 'hold']:
                sigs = signals_in_conflict.get(side, [])
                if sigs:
                    avg_strength = sum(s.strength for s in sigs) / len(sigs)
                    avg_confidence = sum(s.confidence for s in sigs) / len(sigs)
                    if avg_confidence >= self.min_conf and avg_strength >= self.min_strength:
                        # Get price from features if available
                        current_price = None
                        if sigs[0].features and 'close' in sigs[0].features:
                            current_price = sigs[0].features['close']

                        final.append(TacticalSignal(
                            symbol=symbol,
                            strength=avg_strength,
                            confidence=avg_confidence,
                            side=side,
                            type='market',  # Required field
                            signal_type='tactical',  # Required field
                            source='composed',
                            features=sigs[0].features if sigs[0].features else {},
                            timestamp=pd.Timestamp.now(),
                            price=current_price,  # Include price
                            quantity=0.0,  # Let order manager calculate based on portfolio
                            stop_loss=None,  # Risk overlay should set this
                            take_profit=None,  # Risk overlay should set this
                            metadata={
                                'composed_from': [s.source for s in sigs],
                                'technical_indicators': {k: v for k, v in (sigs[0].features or {}).items() 
                                                      if isinstance(v, (int, float))},
                                'source': 'composed',
                                'signal_type': side
                            }
                        ))

            # Resolver conflictos si hay buy y sell
            buy = next((s for s in final if s.side == 'buy' and s.symbol == symbol), None)
            sell = next((s for s in final if s.side == 'sell' and s.symbol == symbol), None)
            hold = next((s for s in final if s.side == 'hold' and s.symbol == symbol), None)

            if buy and sell:
                b_score = buy.strength * buy.confidence
                s_score = sell.strength * sell.confidence
                if abs(b_score - s_score) > self.conflict_tie_threshold * 5 and self.keep_both_when_far:
                    continue  # Mantener ambos
                elif b_score > s_score + self.conflict_tie_threshold:
                    final = [s for s in final if s.side != 'sell' or s.symbol != symbol]
                elif s_score > b_score + self.conflict_tie_threshold:
                    final = [s for s in final if s.side != 'buy' or s.symbol != symbol]
                else:
                    # Empate: Mantener el de mayor confianza, o descartar ambos
                    if buy.confidence > sell.confidence:
                        final = [s for s in final if s.side != 'sell' or s.symbol != symbol]
                    elif sell.confidence > buy.confidence:
                        final = [s for s in final if s.side != 'buy' or s.symbol != symbol]
                    else:
                        final = [s for s in final if s.side not in ['buy', 'sell'] or s.symbol != symbol]
                        logger.debug(f"Conflict tie for {symbol}; discarding buy/sell")

            # Si solo hold y no buy/sell, mantener si confianza alta
            if hold and not (buy or sell) and hold.confidence >= self.min_conf:
                logger.debug(f"Conservando hold para {symbol} con confianza {hold.confidence:.3f}")

        # Validación final de señales
        validated_signals = []
        for signal in final:
            # Aceptar señales de alta confianza incluso con menor fuerza
            if signal.confidence >= self.high_conf_threshold:
                validated_signals.append(signal)
                logger.debug(f"✅ Señal validada por alta confianza: {signal.symbol} {signal.side}")
            # Para el resto, aplicar criterios estándar
            elif signal.confidence >= self.min_conf and signal.strength >= self.min_strength:
                validated_signals.append(signal)
                logger.debug(f"✅ Señal validada por criterios estándar: {signal.symbol} {signal.side}")
            else:
                logger.debug(f"❌ Señal rechazada: {signal.symbol} {signal.side} (conf={signal.confidence:.3f}, strength={signal.strength:.3f})")

        logger.info(f"Final signals after composition and conflict resolution: {len(validated_signals)}")
        return validated_signals