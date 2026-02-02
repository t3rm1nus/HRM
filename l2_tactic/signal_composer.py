# l2_tactic/signal_composer.py
from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from .config import *  # Import L2 config settings
from .models import TacticalSignal
from .utils import safe_float
from .similarity_detector import SignalSimilarityDetector, SimilarityConfig, SimilarityAlgorithm
from datetime import datetime
from core.logging import logger
logger.info("l2_tactic.signal_composer")
import pandas as pd
import numpy as np

class SignalComposer:
    """
    Combina señales tácticas de múltiples fuentes (IA, técnicas, patrones).
    - Weighted voting / score average por símbolo+lado
    - Resolución de conflictos BUY vs SELL en el mismo símbolo
    - Ajuste dinámico de pesos por performance histórica (inyectada vía metrics)
    - HIERARCHICAL OVERRIDE: L3 strategic signals can veto L1/L2 when confidence > 0.7
    """
    def compose_signal(self, symbol: str, base_signal: TacticalSignal, indicators: Dict, state: Dict, l3_signal: Optional[TacticalSignal] = None) -> Optional[TacticalSignal]:
        """
        Compone una señal táctica final combinando la señal base con indicadores técnicos

        Args:
            symbol: El símbolo para el que se genera la señal
            base_signal: La señal base (típicamente de FinRL)
            indicators: Indicadores técnicos calculados
            state: Estado del sistema

        Returns:
            TacticalSignal opcional (None si la señal es rechazada)
        """
        # DEBUG: Log signal processing
        logger.info(f"🎯 SIGNAL COMPOSER INPUT for {symbol}: side={getattr(base_signal, 'side', 'no_side')}, conf={getattr(base_signal, 'confidence', 'no_conf'):.3f}, strength={getattr(base_signal, 'strength', 'no_strength'):.3f}")

        # Validar entrada básica
        if not symbol:
            logger.error("❌ Símbolo vacío en compose_signal")
            return None

        # Validar señal base
        if not base_signal:
            logger.error(f"❌ Señal base vacía para {symbol}")
            return None

        if not isinstance(base_signal, TacticalSignal):
            logger.error(f"❌ Señal base no es TacticalSignal para {symbol}: {type(base_signal)}")
            return None

        # Validar que la señal base tenga los campos requeridos
        required_attrs = ['side', 'confidence', 'strength']
        missing_attrs = [attr for attr in required_attrs if not hasattr(base_signal, attr)]
        if missing_attrs:
            logger.error(f"❌ Señal base incompleta para {symbol}, faltan: {missing_attrs}")
            return None

        # Validar valores de confianza y fuerza usando safe_float
        try:
            confidence = safe_float(base_signal.confidence)
            strength = safe_float(base_signal.strength)
            if np.isnan(confidence) or np.isnan(strength):
                logger.error(f"❌ Valores NaN en señal base para {symbol}: conf={base_signal.confidence}, strength={base_signal.strength}")
                return None
        except (ValueError, TypeError):
            logger.error(f"❌ Valores inválidos en señal base para {symbol}: conf={base_signal.confidence}, strength={base_signal.strength}")
            return None

        # DEBUG: Log threshold check
        logger.info(f"🔍 SIGNAL COMPOSER THRESHOLD CHECK for {symbol}: conf={confidence:.3f} >= {self.min_conf:.3f}, strength={strength:.3f} >= {self.min_strength:.3f}")
        logger.info(f"🔍 SIGNAL COMPOSER CURRENT THRESHOLDS: min_conf={self.min_conf}, min_strength={self.min_strength}")

        # VALIDACIÓN CORREGIDA: Permitir HOLD con buena confidence independientemente de strength
        if base_signal.side == 'hold':
            # Para señales HOLD: mantener si confidence >= 0.3 (relajado vs 0.7 para permitir diversidad)
            if confidence >= 0.3:
                logger.debug(f"✅ HOLD mantenido por buena confidence: {confidence:.2f} (strength: {strength:.2f})")
            else:
                logger.debug(f"❌ HOLD rechazado por baja confidence: {confidence:.2f} < 0.3")
                return None
        else:
            # Para señales BUY/SELL: aplicar criterios normales de fuerza y confianza
            if confidence < self.min_conf or strength < self.min_strength:
                logger.debug(f"❌ BUY/SELL rechazada por baja confianza/fuerza: conf={confidence:.2f}/{self.min_conf:.2f}, strength={strength:.2f}/{self.min_strength:.2f}")
                return None

        # Validar side
        if not hasattr(base_signal, 'side') or base_signal.side not in ['buy', 'sell', 'hold']:
            logger.error(f"❌ Side inválido para {symbol}: {getattr(base_signal, 'side', 'None')}")
            return None

        # Enriquecer con indicadores técnicos si están disponibles
        enhanced_features = base_signal.features.copy() if hasattr(base_signal, 'features') and base_signal.features else {}

        if indicators:
            # Agregar indicadores técnicos a las features
            for key, value in indicators.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    enhanced_features[key] = safe_float(value)
                elif hasattr(value, 'iloc') and len(value) > 0:  # Series
                    try:
                        last_val = safe_float(value.iloc[-1])
                        if not np.isnan(last_val):
                            enhanced_features[key] = last_val
                    except (IndexError, ValueError, TypeError):
                        pass

        # Crear señal compuesta con features enriquecidas
        composed_signal = TacticalSignal(
            symbol=symbol,
            side=base_signal.side,
            strength=strength,
            confidence=confidence,
            signal_type=getattr(base_signal, 'signal_type', base_signal.side),
            source='composed',
            features=enhanced_features,
            timestamp=pd.Timestamp.now(),
            metadata={
                'original_source': getattr(base_signal, 'source', 'unknown'),
                'composed_from': 'base_signal + indicators',
                'indicators_count': len(indicators) if indicators else 0
            }
        )

        # 🛡️ PRESERVAR STOP-LOSS de la señal base
        if hasattr(base_signal, 'stop_loss') and base_signal.stop_loss is not None:
            composed_signal.stop_loss = base_signal.stop_loss

        # 🔄 INTEGRATE CONVERGENCE-BASED PROFIT TAKING
        convergence_score = 0.5  # Default neutral convergence
        if hasattr(base_signal, 'features') and base_signal.features:
            # Extract convergence from various possible sources
            convergence_score = safe_float(base_signal.features.get('l1_l2_agreement', 0.5))
            convergence_score = safe_float(base_signal.features.get('convergence', convergence_score))
            convergence_score = safe_float(base_signal.features.get('signal_convergence', convergence_score))

            # Ensure convergence is within valid range
            convergence_score = max(0.0, min(1.0, convergence_score))

            # Add convergence as a direct attribute for order manager access
            composed_signal.convergence = convergence_score

            logger.debug(f"🔄 CONVERGENCE INTEGRATED: {symbol} {base_signal.side} convergence={convergence_score:.3f}")

        # 🎯 INTEGRATE STAGGERED PROFIT-TAKING METADATA
        if hasattr(base_signal, 'features') and base_signal.features:
            rsi_value = safe_float(base_signal.features.get('rsi', 50.0))
            convergence_strength = convergence_score  # Use the extracted convergence score

            # Get current price for profit target calculation
            current_price_for_pt = None
            if hasattr(base_signal, 'price') and base_signal.price:
                current_price_for_pt = safe_float(base_signal.price)
            elif 'close' in base_signal.features and base_signal.features['close']:
                current_price_for_pt = safe_float(base_signal.features['close'])

            # Fallback to default prices if needed
            if not current_price_for_pt:
                if symbol == 'BTCUSDT':
                    current_price_for_pt = 110000.0
                elif symbol == 'ETHUSDT':
                    current_price_for_pt = 4300.0
                else:
                    current_price_for_pt = 1000.0

            # Calculate profit targets using signal generator's method
            try:
                from .tactical_signal_processor import L2TacticProcessor
                # Create a temporary instance to access the method
                temp_processor = L2TacticProcessor.__new__(L2TacticProcessor)
                profit_targets = temp_processor._calculate_profit_targets(
                    current_price_for_pt, base_signal.side, rsi_value, convergence_strength
                )

                # Add profit-taking metadata to composed signal
                if hasattr(composed_signal, 'metadata'):
                    composed_signal.metadata = composed_signal.metadata or {}
                    composed_signal.metadata.update({
                        'profit_taking_levels': profit_targets,
                        'profit_taking_rsi': rsi_value,
                        'profit_taking_convergence': convergence_strength,
                        'profit_taking_calculated': True
                    })

                # INTEGRATE MULTI-LEVEL PROFIT TAKING: Add profit targets as direct attributes
                if profit_targets and len(profit_targets) >= 3:
                    composed_signal.profit_target_1 = profit_targets[0]  # First profit target
                    composed_signal.profit_target_2 = profit_targets[1]  # Second profit target
                    composed_signal.profit_target_3 = profit_targets[2]  # Third profit target

                    logger.info(f"🎯 MULTI-LEVEL PROFIT TARGETS SET: {symbol} {base_signal.side}")
                    logger.info(f"   Target 1: {profit_targets[0]:.2f}")
                    logger.info(f"   Target 2: {profit_targets[1]:.2f}")
                    logger.info(f"   Target 3: {profit_targets[2]:.2f}")

                logger.info(f"🎯 PROFIT-TAKING INTEGRATED: {len(profit_targets)} levels for {symbol} {base_signal.side} @ {current_price_for_pt:.2f} (RSI={rsi_value:.1f}, conv={convergence_strength:.3f})")

            except Exception as e:
                logger.warning(f"⚠️ Could not calculate profit targets for {symbol}: {e}")
                # Add fallback metadata
                if hasattr(composed_signal, 'metadata'):
                    composed_signal.metadata = composed_signal.metadata or {}
                    composed_signal.metadata.update({
                        'profit_taking_levels': [],
                        'profit_taking_error': str(e),
                        'profit_taking_calculated': False
                    })

        # 🔴 HIERARCHICAL OVERRIDE: L3 VETO WHEN CONFIDENCE > 0.7
        if l3_signal and hasattr(l3_signal, 'confidence') and safe_float(l3_signal.confidence) > 0.7:
            l3_conf = safe_float(l3_signal.confidence)
            l3_side = getattr(l3_signal, 'side', 'hold')
            logger.info(f"🔴 HIERARCHICAL OVERRIDE: L3 VETO ACTIVATED for {symbol}!")
            logger.info(f"   L3 Signal: {l3_side} (conf={l3_conf:.3f})")
            logger.info(f"   L1/L2 Original: {composed_signal.side} (conf={confidence:.3f})")

            # L3 OVERRIDE: Replace the entire signal with L3 decision
            composed_signal.side = l3_side
            composed_signal.confidence = l3_conf
            composed_signal.strength = getattr(l3_signal, 'strength', 0.8)  # L3 signals get high strength
            composed_signal.source = 'l3_override'
            composed_signal.signal_type = 'strategic'

            # Update metadata to reflect override
            if hasattr(composed_signal, 'metadata'):
                composed_signal.metadata = composed_signal.metadata or {}
                composed_signal.metadata.update({
                    'l3_override': True,
                    'original_l1_l2_side': base_signal.side,
                    'original_l1_l2_confidence': confidence,
                    'l3_veto_reason': f'L3 confidence {l3_conf:.3f} > 0.7'
                })

            logger.info(f"🔴 L3 VETO RESULT: {symbol} {l3_side} (conf={l3_conf:.3f}) - L1/L2 overridden")

        logger.debug(f"✅ Señal compuesta para {symbol}: {composed_signal.side} (conf={composed_signal.confidence:.2f}, strength={composed_signal.strength:.2f})")
        return composed_signal

    def __init__(self, config: SignalConfig, metrics: Optional[object] = None):
        self.cfg = config
        self.metrics = metrics

        # Valores por defecto si no existen en el config
        config_defaults = {
            "ai_model_weight": 0.45,      # Peso principal a la IA
            "technical_weight": 0.35,      # Técnico como apoyo
            "pattern_weight": 0.20,        # Patrones como confirmación
            "min_signal_strength": 0.10,   # Más permisivo
            "conflict_tie_threshold": 0.15, # Mayor margen para conflictos
            "keep_both_when_far": True     # Mantener señales opuestas si son fuertes
        }

        # Usar valores de la configuración o defaults
        self.w_ai = getattr(config, "ai_model_weight", config_defaults["ai_model_weight"])
        self.w_tech = getattr(config, "technical_weight", config_defaults["technical_weight"])
        self.w_pattern = getattr(config, "pattern_weight", config_defaults["pattern_weight"])

        # ✅ FIXED: Use correct attribute names from L2Config
        # 🛠️ CRITICAL FIX: More permissive thresholds to allow legitimate signals through
        # Lowered from 0.4/0.15 to 0.3/0.1 to allow initial positions and fix the bottleneck
        self.min_conf = 0.3  # REQUIRES CONFIDENCE >= 0.3 (lowered further to pass signals with ~0.49 conf)
        self.min_strength = 0.1  # REQUIRES STRENGTH >= 0.1 (lowered further to pass signals with ~0.21 strength)

        # ✅ FIXED: Use proper attribute access for SignalConfig dataclass
        self.conflict_tie_threshold = getattr(config, "conflict_tie_threshold", config_defaults["conflict_tie_threshold"])

        # Inicializar histórico de señales
        self._last_signals = {}
        self.keep_both_when_far = True  # Forzar keep_both para señales con alta confianza
        self.high_conf_threshold = 0.7   # Nueva: umbral para señales de alta confianza

        # Initialize similarity detector for signal processing
        similarity_config = SimilarityConfig(
            algorithm=SimilarityAlgorithm.FEATURE_WEIGHTED,
            threshold=0.7,  # Medium threshold for similarity detection
            enable_duplicate_filtering=True,
            enable_pattern_recognition=True,
            enable_clustering=False,  # Disable clustering by default for performance
            time_window_minutes=30
        )
        self.similarity_detector = SignalSimilarityDetector(similarity_config)
        logger.info("🔍 Similarity detector initialized in SignalComposer")

    def _process_signal_dict(self, signal_dict: dict) -> Optional[TacticalSignal]:
        """Helper to process dictionary signals"""
        try:
            # Ensure numeric values are properly converted using safe_float
            if 'strength' in signal_dict:
                signal_dict['strength'] = safe_float(signal_dict['strength'])
            if 'confidence' in signal_dict:
                signal_dict['confidence'] = safe_float(signal_dict['confidence'])

            # Convert features to proper types
            if 'features' in signal_dict and isinstance(signal_dict['features'], dict):
                features = {}
                for k, v in signal_dict['features'].items():
                    if isinstance(v, (int, float)):
                        features[k] = safe_float(v)
                signal_dict['features'] = features
            
            return TacticalSignal(**signal_dict)
        except Exception as e:
            logger.error(f"❌ Error procesando señal dict: {e}")
            return None

    # --- MÉTODO CORREGIDO ---
    def compose(self, signals: List[TacticalSignal], state: Dict = None) -> List[TacticalSignal]:
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
                # 🛠️ DIFERENCIAR LÓGICA POR TIPO DE SEÑAL (CRÍTICO FIX)
                # En lugar de promediar buy/sell vs hold, procesar por separado

                # 1. Procesar señales HOLD primero (lógica especial)
                hold_signals = [s for s in sym_signals if s.side.lower() == 'hold']
                if hold_signals:
                    # Para HOLD: seleccionar la señal con mayor confidence (sin requerir strength)
                    best_hold_signal = max(hold_signals, key=lambda s: getattr(s, 'confidence', 0.5))
                    hold_confidence = getattr(best_hold_signal, 'confidence', 0.5)

                    # ✅ Aplicar lógica diferenciada para HOLD: solo requiere confidence >= 0.3
                    if hold_confidence >= 0.3:
                        logger.debug(f"✅ HOLD mantenido por confidence: conf={hold_confidence:.3f} (strength ignorado para HOLD)")
                        side = 'hold'
                        avg_strength = getattr(best_hold_signal, 'strength', 0.1)  # Mantener original
                        avg_confidence = hold_confidence
                    else:
                        # Skip HOLD signals con baja confidence
                        logger.debug(f"❌ HOLD rechazado: confidence {hold_confidence:.3f} < 0.3")
                        continue
                else:
                    # 2. Procesar señales BUY/SELL (lógica normal)
                    buy_signals = [s for s in sym_signals if s.side.lower() == 'buy']
                    sell_signals = [s for s in sym_signals if s.side.lower() == 'sell']

                    # Calcular fuerza por lado
                    buy_strength = sum(s.strength * self._get_dynamic_weight(s) for s in buy_signals) / total_weight if buy_signals else 0
                    sell_strength = sum(s.strength * self._get_dynamic_weight(s) for s in sell_signals) / total_weight if sell_signals else 0

                    # Decidir el lado dominante
                    side = 'buy' if buy_strength > sell_strength else 'sell'
                    strength_diff = abs(buy_strength - sell_strength)

                    # Si la diferencia es pequeña, mantener la señal original en lugar de ignorar
                    if strength_diff < self.conflict_tie_threshold:
                        logger.debug(f"Señales {symbol} muy cercanas (diff={strength_diff:.3f}), manteniendo señal original")
                        # Usar la señal dominante por confianza en lugar de ignorar
                        all_signals = buy_signals + sell_signals
                        if all_signals:
                            dominant_signal = max(all_signals, key=lambda s: s.confidence * self._get_dynamic_weight(s))
                            side = dominant_signal.side
                            avg_strength = dominant_signal.strength
                            avg_confidence = dominant_signal.confidence
                        else:
                            continue  # No hay señales para procesar
                    else:
                        # Usar señales del lado dominante
                        relevant_signals = buy_signals if side == 'buy' else sell_signals
                        if not relevant_signals:
                            continue

                        # Calcular métricas finales
                        dominant_signal = max(relevant_signals, key=lambda s: s.strength * self._get_dynamic_weight(s))
                        avg_strength = max(buy_strength, sell_strength)
                        avg_confidence = sum(s.confidence * self._get_dynamic_weight(s) for s in relevant_signals) / sum(self._get_dynamic_weight(s) for s in relevant_signals)

                        # 🛡️ EMERGENCY OVERRIDE: Check for initial positions to allow lower thresholds
                        has_positions = False
                        if state and 'portfolio' in state:
                            portfolio = state['portfolio']
                            for sym, data in portfolio.items():
                                if isinstance(data, dict) and abs(data.get('position', 0.0)) > 0:
                                    has_positions = True
                                    break

                        # Aplicar validación final para BUY/SELL con override para posiciones iniciales
                        passes_validation = (avg_strength >= self.min_strength and avg_confidence >= self.min_conf) or \
                                           (not has_positions and avg_confidence > 0.45 and avg_strength > 0.15)

                        if not passes_validation:
                            logger.debug(f"❌ BUY/SELL rechazado: strength={avg_strength:.3f}/{self.min_strength:.3f}, conf={avg_confidence:.3f}/{self.min_conf:.3f}, has_positions={has_positions}")
                            continue

                # Get current price from market data or features
                current_price = None
                if 'close' in features and features['close']:
                    current_price = safe_float(features['close'])
                elif hasattr(dominant_signal, 'price') and dominant_signal.price:
                    current_price = safe_float(dominant_signal.price)
                
                # Fallback a precios por defecto si no se encuentra precio
                if not current_price:
                    if symbol == 'BTCUSDT':
                        current_price = 110000.0  # Precio fallback BTC
                    elif symbol == 'ETHUSDT':
                        current_price = 4300.0    # Precio fallback ETH
                    else:
                        current_price = 1000.0    # Precio fallback genérico
                    logger.warning(f"⚠️ Usando precio fallback para {symbol}: {current_price}")
                
                # VALIDACIÓN DIFERENCIADA POR TIPO DE SEÑAL (CRÍTICO FIX)
                signal_passes_validation = False

                if side == 'hold':
                    # Para HOLD: solo requiere confidence >= 0.3 (acepta low strength en mercados laterales)
                    if avg_confidence >= 0.3:
                        signal_passes_validation = True
                        logger.debug(f"✅ HOLD validado por confidence: conf={avg_confidence:.3f} (strength={avg_strength:.3f} - OK para mercado lateral)")
                    else:
                        logger.debug(f"❌ HOLD rechazado: confidence {avg_confidence:.3f} < 0.3")
                else:
                    # Para BUY/SELL: requiere tanto strength como confidence
                    if avg_strength >= self.min_strength and avg_confidence >= self.min_conf:
                        signal_passes_validation = True
                        logger.debug(f"✅ BUY/SELL validado: strength={avg_strength:.3f} >= {self.min_strength:.3f}, conf={avg_confidence:.3f} >= {self.min_conf:.3f}")
                    else:
                        logger.debug(f"❌ BUY/SELL rechazado: strength={avg_strength:.3f}/{self.min_strength:.3f}, conf={avg_confidence:.3f}/{self.min_conf:.3f}")

                if not signal_passes_validation:
                    continue  # Skip esta señal, no la proceses

                # ENHANCED POSITION SIZING WITH CONVERGENCE AND TECHNICAL STRENGTH
                quantity = self._calculate_enhanced_position_size(
                    symbol, side, avg_strength, avg_confidence, features, state, current_price
                )

                # Calcular stop-loss y take-profit
                stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                    current_price, side, avg_strength, avg_confidence, features
                )

                logger.info(f"🛡️ SL/TP para {symbol}: SL={stop_loss:.2f}, TP={take_profit:.2f}")

                # Calculate convergence score for the composed signal
                convergence_score = 0.5  # Default neutral convergence
                if features:
                    # Extract convergence from various possible sources
                    convergence_score = safe_float(features.get('l1_l2_agreement', 0.5))
                    convergence_score = safe_float(features.get('convergence', convergence_score))
                    convergence_score = safe_float(features.get('signal_convergence', convergence_score))
                    convergence_score = max(0.0, min(1.0, convergence_score))

                composed_signal = TacticalSignal(
                    symbol=symbol,
                    side=side,  # Usar el lado dominante calculado anteriormente
                    strength=avg_strength,
                    confidence=avg_confidence,
                    type="market",
                    signal_type='tactical',
                    features=features,  # Include all collected features
                    timestamp=pd.Timestamp.now(),
                    source='composed',
                    price=current_price,
                    quantity=quantity,  # Usar la cantidad calculada y asegurarnos que no es 0
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'composed_from': [s.source for s in sym_signals],
                        'technical_indicators': {k: v for k, v in features.items()
                                            if isinstance(v, (int, float))},
                        'source': 'composed',
                        'signal_type': 'composed',
                        'convergence_score': convergence_score
                    }
                )

                # Add convergence as a direct attribute for order manager access
                composed_signal.convergence = convergence_score
                composed_signals.append(composed_signal)
                logger.debug(f"✅ Señal compuesta para {symbol}: side={dominant_signal.side}, strength={avg_strength:.3f}, confidence={avg_confidence:.3f}")

                # 🏆 ENHANCED: Also create signals for the non-dominant side if they have high confidence
                # This allows conflict resolution to properly prioritize high-confidence signals
                non_dominant_signals = sell_signals if side == 'buy' else buy_signals
                if non_dominant_signals:
                    # Calculate metrics for non-dominant side
                    non_dominant_strength = sum(s.strength * self._get_dynamic_weight(s) for s in non_dominant_signals) / total_weight
                    non_dominant_confidence = sum(s.confidence * self._get_dynamic_weight(s) for s in non_dominant_signals) / sum(self._get_dynamic_weight(s) for s in non_dominant_signals)

                    # Only create non-dominant signal if it has high confidence (>= 0.8)
                    if non_dominant_confidence >= 0.8 and non_dominant_strength >= self.min_strength:
                        non_dominant_side = 'sell' if side == 'buy' else 'buy'

                        # Calculate quantity for non-dominant signal
                        if non_dominant_side == 'sell':
                            current_position = 0.0
                            try:
                                portfolio = state.get("portfolio", {})
                                symbol_data = portfolio.get(symbol, {})
                                if isinstance(symbol_data, dict):
                                    current_position = abs(symbol_data.get("position", 0.0))
                            except:
                                current_position = 0.0

                            if current_position > 0:
                                sell_percentage = min(0.5, 0.1 + (non_dominant_strength * non_dominant_confidence * 0.4))
                                quantity = current_position * sell_percentage
                            else:
                                quantity = 0
                        else:
                            base_quantity = 0.05 if symbol == 'BTCUSDT' else 0.5
                            base_multiplier = 0.15
                            quantity = base_quantity * base_multiplier * non_dominant_strength * non_dominant_confidence

                        if quantity > 0:
                            # Calculate stop-loss and take-profit
                            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                                current_price, non_dominant_side, non_dominant_strength, non_dominant_confidence, features
                            )

                            non_dominant_signal = TacticalSignal(
                                symbol=symbol,
                                side=non_dominant_side,
                                strength=non_dominant_strength,
                                confidence=non_dominant_confidence,
                                type="market",
                                signal_type='tactical',
                                features=features,
                                timestamp=pd.Timestamp.now(),
                                source='composed',
                                price=current_price,
                                quantity=quantity,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                metadata={
                                    'composed_from': [s.source for s in non_dominant_signals],
                                    'technical_indicators': {k: v for k, v in features.items()
                                                        if isinstance(v, (int, float))},
                                    'source': 'composed',
                                    'signal_type': 'non_dominant_high_conf',
                                    'convergence_score': convergence_score
                                }
                            )

                            # Add convergence as a direct attribute for order manager access
                            non_dominant_signal.convergence = convergence_score
                            composed_signals.append(non_dominant_signal)
                            logger.info(f"🏆 HIGH CONFIDENCE NON-DOMINANT: Added {non_dominant_side} signal (conf={non_dominant_confidence:.3f}) for conflict resolution")

        # Resolver conflictos y filtrar por thresholds
        filtered_signals = self._resolve_conflicts_and_filter(composed_signals, state)

        # 🔍 INTEGRATE SIMILARITY DETECTOR: Apply similarity-based filtering and prioritization
        try:
            logger.info(f"🔍 Applying similarity detection to {len(filtered_signals)} signals")
            processed_signals, analysis_results = self.similarity_detector.process_signals(filtered_signals)

            # Log similarity analysis results
            if analysis_results:
                logger.info(f"🔍 Similarity Analysis: {analysis_results.get('original_count', 0)} → {analysis_results.get('prioritized_count', 0)} signals")
                logger.info(f"   Groups found: {analysis_results.get('similarity_groups', 0)}")

                market_patterns = analysis_results.get('market_patterns', {})
                if market_patterns:
                    regime = market_patterns.get('market_regime', 'unknown')
                    volatility = market_patterns.get('volatility_pattern', 'normal')
                    momentum = market_patterns.get('momentum_pattern', 'neutral')
                    logger.info(f"📊 Market Patterns: regime={regime}, volatility={volatility}, momentum={momentum}")

            filtered_signals = processed_signals

        except Exception as e:
            logger.error(f"❌ Error in similarity detection integration: {e}")
            # Continue with original filtered signals if similarity detection fails

        logger.info(f"✅ Señales compuestas generadas: {len(filtered_signals)}")
        return filtered_signals

    # --- Métodos auxiliares ---
    def normalize_features(self, features: dict, symbol: str) -> dict:
        """Normaliza features según el activo"""
        if not features:
            return {}
            
        normalized = {}
        if symbol == 'BTCUSDT':
            normalized.update({
                'rsi': features.get('rsi', 50) / 100,
                'macd': features.get('macd', 0) / 100,  # normalizar por rango típico de BTC
                'vol_zscore': features.get('vol_zscore', 0) / 3  # normalizar por desvíos estándar
            })
        elif symbol == 'ETHUSDT':
            normalized.update({
                'rsi': features.get('rsi', 50) / 100,
                'macd': features.get('macd', 0) / 50,  # normalizar por rango típico de ETH
                'vol_zscore': features.get('vol_zscore', 0) / 3
            })
        return normalized

    def _get_dynamic_weight(self, signal: TacticalSignal) -> float:
        """Calcula peso dinámico basado en la fuente y calidad de la señal"""
        logger.debug(f"Calculando peso para señal: source={signal.source}, confidence={signal.confidence}")
        base_weight = 1.0

        # Por fuente
        if signal.source == 'ai':
            base_weight *= 1.2  # Reducir peso de IA
            # 🛠️ AJUSTE: Extra weight for high-confidence L2 signals (> 0.85)
            if signal.confidence > 0.85:
                base_weight *= 1.5  # Additional 50% weight for very confident L2 signals
                logger.info(f"🚀 HIGH CONFIDENCE L2 BOOST: {signal.symbol} {signal.side} conf={signal.confidence:.3f} → weight ×1.5")
            if signal.side == 'hold':  # Baja peso para holds
                base_weight *= 0.2
        elif signal.source == 'technical':
            base_weight *= 1.3  # Aumentar peso técnico
        elif signal.source == 'risk':
            base_weight *= 2.0  # Mantener peso de riesgo alto
            
        # Por indicador
        if signal.features:
            normalized = self.normalize_features(signal.features, signal.symbol)
            rsi = signal.features.get('rsi', 50)
            macd = signal.features.get('macd', 0)
            
            if abs(rsi - 50) > 20:
                base_weight *= 1.2  # Aumentar peso cuando RSI está en extremos
            if abs(macd) > 10:
                base_weight *= 1.1  # Aumentar peso cuando MACD es fuerte
                
        return max(base_weight * signal.confidence, 0.01)

    def _resolve_conflicts_and_filter(self, signals: List[TacticalSignal], state: Dict = None) -> List[TacticalSignal]:
        """
        Resuelve conflictos BUY vs SELL y filtra por umbrales de calidad.
        APLICA LÓGICA DIFERENCIADA: HOLD solo requiere confidence >= 0.5, BUY/SELL requieren confidence + strength
        """
        signals_by_symbol = {}

        # ✅ CRÍTICO FIX: FILTRO DIFERENCIADO POR TIPO DE SEÑAL
        for s in signals:
            if not s:
                continue

            # 🔍 LÓGICA DIFERENCIADA PARA PASAR FILTROS INICIALES
            signal_passes_filter = False

            if s.side.lower() == 'hold':
                # Para HOLD: solo requiere confidence >= 0.5 (acepta low strength en mercados laterales)
                if getattr(s, 'confidence', 0.0) >= 0.5:
                    signal_passes_filter = True
                    logger.debug(f"✅ HOLD pasa filtro inicial por confidence: conf={getattr(s, 'confidence', 0.0):.3f} (strength={getattr(s, 'strength', 0.0):.3f} - OK para mercado lateral)")
                else:
                    logger.debug(f"❌ HOLD rechazada por baja confidence: conf={getattr(s, 'confidence', 0.0):.3f} < 0.5")
            else:
                # Para BUY/SELL: requiere tanto confidence como strength
                confidence = getattr(s, 'confidence', 0.0)
                strength = getattr(s, 'strength', 0.0)
                if confidence >= self.min_conf and strength >= self.min_strength:
                    signal_passes_filter = True
                    logger.debug(f"✅ BUY/SELL pasan filtro inicial: strength={strength:.3f} >= {self.min_strength:.3f}, conf={confidence:.3f} >= {self.min_conf:.3f}")
                else:
                    logger.debug(f"❌ BUY/SELL rechazada: strength={strength:.3f}/{self.min_strength:.3f}, conf={confidence:.3f}/{self.min_conf:.3f}")

            if signal_passes_filter:
                signals_by_symbol.setdefault(s.symbol, {}).setdefault(s.side, []).append(s)

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
                        if sigs[0].features and 'close' in sigs[0].features and sigs[0].features['close']:
                            current_price = safe_float(sigs[0].features['close'])
                        
                        # Fallback a precios por defecto si no se encuentra precio
                        if not current_price:
                            if symbol == 'BTCUSDT':
                                current_price = 110000.0  # Precio fallback BTC
                            elif symbol == 'ETHUSDT':
                                current_price = 4300.0    # Precio fallback ETH
                            else:
                                current_price = 1000.0    # Precio fallback genérico
                            logger.warning(f"⚠️ Usando precio fallback para {symbol}: {current_price}")

                        # Calcular cantidad base según el símbolo - MÁS AGRESIVA
                        if side == 'sell':
                            # For SELL signals, use percentage of current position with confidence-based scaling
                            current_position = 0.0
                            try:
                                portfolio = state.get("portfolio", {})
                                symbol_data = portfolio.get(symbol, {})
                                if isinstance(symbol_data, dict):
                                    current_position = abs(symbol_data.get("position", 0.0))
                            except:
                                current_position = 0.0

                            if current_position > 0:
                                # ENHANCED CONFIDENCE-BASED SIZING: Scale up for high confidence signals (0.7+)
                                # UPDATED: More aggressive scaling for high confidence signals
                                confidence_multiplier = 1.0
                                if avg_confidence >= 0.9:
                                    confidence_multiplier = 2.5  # 2.5x for very high confidence
                                elif avg_confidence >= 0.8:
                                    confidence_multiplier = 2.0  # 2.0x for high confidence
                                elif avg_confidence >= 0.7:
                                    confidence_multiplier = 1.5  # 1.5x for moderate-high confidence

                                # Base sell percentage with confidence scaling - INCREASED BASE
                                base_sell_percentage = 0.15 + (avg_strength * avg_confidence * 0.5)  # Increased from 0.1 to 0.15, from 0.4 to 0.5
                                sell_percentage = min(0.7, base_sell_percentage * confidence_multiplier)  # Increased max from 0.5 to 0.7
                                calculated_quantity = current_position * sell_percentage
                                logger.info(f"💰 ENHANCED SELL POSITION SIZING (conflict): {symbol} position={current_position:.6f}, conf={avg_confidence:.3f}→{confidence_multiplier:.1f}x, base={base_sell_percentage:.1%}, final_selling={sell_percentage:.1%} = {calculated_quantity:.6f}")
                            else:
                                # No position to sell, skip signal
                                continue
                        else:
                            # For BUY signals, use fixed quantity with confidence-based scaling
                            base_quantity = 0.05 if symbol == 'BTCUSDT' else 0.5  # BTC: 0.05, ETH: 0.5

                            # ENHANCED CONFIDENCE-BASED SIZING: Scale up for high confidence signals (0.7+)
                            confidence_multiplier = 1.0
                            if avg_confidence >= 0.9:
                                confidence_multiplier = 2.5  # 2.5x for very high confidence
                            elif avg_confidence >= 0.8:
                                confidence_multiplier = 2.0  # 2.0x for high confidence
                            elif avg_confidence >= 0.7:
                                confidence_multiplier = 1.5  # 1.5x for moderate-high confidence

                            # Ajustar por fuerza de la señal y multiplicador de confianza
                            calculated_quantity = base_quantity * (1 + avg_strength) * avg_confidence * confidence_multiplier
                        
                        # Calcular stop-loss y take-profit
                        stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                            current_price, side, avg_strength, avg_confidence, sigs[0].features or {}
                        )
                        
                        # Calculate convergence score for the final signal
                        convergence_score = 0.5  # Default neutral convergence
                        signal_features = sigs[0].features if sigs[0].features else {}
                        if signal_features:
                            # Extract convergence from various possible sources
                            convergence_score = safe_float(signal_features.get('l1_l2_agreement', 0.5))
                            convergence_score = safe_float(signal_features.get('convergence', convergence_score))
                            convergence_score = safe_float(signal_features.get('signal_convergence', convergence_score))
                            convergence_score = max(0.0, min(1.0, convergence_score))

                        final_signal = TacticalSignal(
                            symbol=symbol,
                            strength=avg_strength,
                            confidence=avg_confidence,
                            side=side,
                            type='market',  # Required field
                            signal_type='tactical',  # Required field
                            source='composed',
                            features=signal_features,
                            timestamp=pd.Timestamp.now(),
                            price=current_price,  # Include price
                            quantity=calculated_quantity,  # Use calculated quantity
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            metadata={
                                'composed_from': [s.source for s in sigs],
                                'technical_indicators': {k: v for k, v in signal_features.items()
                                                      if isinstance(v, (int, float))},
                                'source': 'composed',
                                'signal_type': side,
                                'convergence_score': convergence_score
                            }
                        )

                        # Add convergence as a direct attribute for order manager access
                        final_signal.convergence = convergence_score
                        final.append(final_signal)

            # Resolver conflictos si hay buy y sell
            buy = next((s for s in final if s.side == 'buy' and s.symbol == symbol), None)
            sell = next((s for s in final if s.side == 'sell' and s.symbol == symbol), None)
            hold = next((s for s in final if s.side == 'hold' and s.symbol == symbol), None)

            if buy and sell:
                # 🏆 HIGH-CONFIDENCE PRIORITY: Check for high-confidence signals (>= 0.8)
                buy_conf = getattr(buy, 'confidence', 0.5)
                sell_conf = getattr(sell, 'confidence', 0.5)

                logger.info(f"🎯 CONFLICT DETECTED: BUY(conf={buy_conf:.3f}) vs SELL(conf={sell_conf:.3f}) for {symbol}")

                # If one signal has high confidence (>= 0.8) and the other doesn't, prioritize the high-confidence one
                if buy_conf >= 0.8 and sell_conf < 0.8:
                    final = [s for s in final if s.side != 'sell' or s.symbol != symbol]
                    logger.info(f"🎯 HIGH CONFIDENCE PRIORITY: BUY wins (conf={buy_conf:.3f} >= 0.8) over SELL (conf={sell_conf:.3f})")
                    # Skip to next iteration
                elif sell_conf >= 0.8 and buy_conf < 0.8:
                    final = [s for s in final if s.side != 'buy' or s.symbol != symbol]
                    logger.info(f"🎯 HIGH CONFIDENCE PRIORITY: SELL wins (conf={sell_conf:.3f} >= 0.8) over BUY (conf={buy_conf:.3f})")
                    # Skip to next iteration
                else:
                    # Both have similar confidence levels, use traditional scoring
                    # Calcular scores base
                    b_score = buy.strength * buy.confidence
                    s_score = sell.strength * sell.confidence

                    # Añadir peso por volumen
                    if buy.features and 'vol_zscore' in buy.features:
                        vol_weight = 1 + abs(buy.features['vol_zscore']) if buy.features['vol_zscore'] > 0 else 1
                        b_score *= vol_weight
                        logger.debug(f"Buy vol_weight: {vol_weight:.3f}")

                    if sell.features and 'vol_zscore' in sell.features:
                        vol_weight = 1 + abs(sell.features['vol_zscore']) if sell.features['vol_zscore'] > 0 else 1
                        s_score *= vol_weight
                        logger.debug(f"Sell vol_weight: {vol_weight:.3f}")

                    # Añadir peso por tendencia MACD
                    if buy.features and 'macd' in buy.features:
                        macd_weight = 1.2 if buy.features['macd'] > 0 else 0.8
                        b_score *= macd_weight
                        logger.debug(f"Buy macd_weight: {macd_weight:.3f}")

                    if sell.features and 'macd' in sell.features:
                        macd_weight = 1.2 if sell.features['macd'] < 0 else 0.8
                        s_score *= macd_weight
                        logger.debug(f"Sell macd_weight: {macd_weight:.3f}")

                    # Añadir peso por RSI
                    if buy.features and 'rsi' in buy.features:
                        rsi = buy.features['rsi']
                        rsi_weight = 1.3 if rsi < 30 else (1.1 if rsi < 40 else 1.0)
                        b_score *= rsi_weight
                        logger.debug(f"Buy rsi_weight: {rsi_weight:.3f}")

                    if sell.features and 'rsi' in sell.features:
                        rsi = sell.features['rsi']
                        rsi_weight = 1.3 if rsi > 70 else (1.1 if rsi > 60 else 1.0)
                        s_score *= rsi_weight
                        logger.debug(f"Sell rsi_weight: {rsi_weight:.3f}")

                    logger.debug(f"Final scores - Buy: {b_score:.3f}, Sell: {s_score:.3f}")

                    if abs(b_score - s_score) > self.conflict_tie_threshold * 5 and self.keep_both_when_far:
                        continue  # Mantener ambos si la diferencia es muy grande
                    elif b_score > s_score + self.conflict_tie_threshold:
                        final = [s for s in final if s.side != 'sell' or s.symbol != symbol]
                        logger.info(f"Resolviendo conflicto a favor de BUY - scores: {b_score:.3f} vs {s_score:.3f}")
                    elif s_score > b_score + self.conflict_tie_threshold:
                        final = [s for s in final if s.side != 'buy' or s.symbol != symbol]
                        logger.info(f"Resolviendo conflicto a favor de SELL - scores: {s_score:.3f} vs {b_score:.3f}")
                    else:
                        # En caso de empate técnico, usar tendencia
                        if buy.features and sell.features:
                            buy_trend = buy.features.get('macd', 0) > 0 and buy.features.get('rsi', 50) < 50
                            sell_trend = sell.features.get('macd', 0) < 0 and sell.features.get('rsi', 50) > 50

                            if buy_trend and not sell_trend:
                                final = [s for s in final if s.side != 'sell' or s.symbol != symbol]
                                logger.info(f"Empate resuelto por tendencia a favor de BUY")
                            elif sell_trend and not buy_trend:
                                final = [s for s in final if s.side != 'buy' or s.symbol != symbol]
                                logger.info(f"Empate resuelto por tendencia a favor de SELL")
                            else:
                                final = [s for s in final if s.side not in ['buy', 'sell'] or s.symbol != symbol]
                                logger.info(f"Empate sin tendencia clara - descartando ambas señales")

            # Si solo hold y no buy/sell, mantener si confianza alta
            if hold and not (buy or sell) and hold.confidence >= self.min_conf:
                logger.debug(f"Conservando hold para {symbol} con confianza {hold.confidence:.3f}")

        # Validación final de señales
        validated_signals = []
        for signal in final:
            logger.info(f"🔍 VALIDATING SIGNAL: {signal.symbol} {signal.side} conf={signal.confidence:.3f} strength={signal.strength:.3f} (high_threshold={self.high_conf_threshold:.3f}, min_conf={self.min_conf:.3f}, min_strength={self.min_strength:.3f})")
            # Aceptar señales de alta confianza incluso con menor fuerza
            if signal.confidence >= self.high_conf_threshold:
                validated_signals.append(signal)
                logger.info(f"✅ Señal validada por alta confianza: {signal.symbol} {signal.side}")
            # Para el resto, aplicar criterios estándar
            elif signal.confidence >= self.min_conf and signal.strength >= self.min_strength:
                validated_signals.append(signal)
                logger.info(f"✅ Señal validada por criterios estándar: {signal.symbol} {signal.side}")
            else:
                logger.warning(f"❌ Señal rechazada: {signal.symbol} {signal.side} (conf={signal.confidence:.3f}, strength={signal.strength:.3f})")

        logger.info(f"Final signals after composition and conflict resolution: {len(validated_signals)}")
        return validated_signals

    def _calculate_stop_loss_take_profit(self, price: float, side: str, strength: float, confidence: float, features: dict) -> tuple[float, float]:
        """
        Calcula stop-loss y take-profit basado en volatilidad, fuerza de señal y confianza
        """
        if not price or price <= 0:
            logger.warning(f"⚠️ Precio inválido para SL/TP: {price}")
            return None, None
            
        # Configuración base
        base_stop_pct = 0.02  # 2% stop-loss base
        base_tp_pct = 0.04    # 4% take-profit base (2:1 ratio)
        
        # Ajustar por volatilidad si está disponible
        volatility_multiplier = 1.0
        if 'vol_zscore' in features:
            vol_zscore = features['vol_zscore']
            # Aumentar stop si hay alta volatilidad
            volatility_multiplier = 1.0 + abs(vol_zscore) * 0.5
            
        # Ajustar por confianza de la señal
        confidence_multiplier = 1.0 + (1.0 - confidence) * 0.5  # Stop más amplio si menos confianza
        
        # Ajustar por fuerza de la señal
        strength_multiplier = 1.0 + (1.0 - strength) * 0.3  # Stop más amplio si menos fuerza
        
        # Calcular stop-loss final
        final_stop_pct = base_stop_pct * volatility_multiplier * confidence_multiplier * strength_multiplier
        final_stop_pct = min(final_stop_pct, 0.05)  # Máximo 5% stop-loss
        
        # Calcular take-profit con ratio riesgo/beneficio
        risk_reward_ratio = 2.0  # 2:1 por defecto
        if confidence > 0.8:
            risk_reward_ratio = 2.5  # Mejor ratio para señales de alta confianza
        elif confidence < 0.5:
            risk_reward_ratio = 1.5  # Ratio más conservador para señales de baja confianza
            
        final_tp_pct = final_stop_pct * risk_reward_ratio
        
        # Aplicar según el lado de la operación
        if side.lower() == 'buy':
            stop_loss = price * (1 - final_stop_pct)
            take_profit = price * (1 + final_tp_pct)
        else:  # sell
            stop_loss = price * (1 + final_stop_pct)
            take_profit = price * (1 - final_tp_pct)
            
        return stop_loss, take_profit

    def _calculate_enhanced_position_size(self, symbol: str, side: str, strength: float, confidence: float,
                                        features: dict, state: dict, current_price: float) -> float:
        """
        Calculate enhanced position size using convergence and technical strength.

        Args:
            symbol: Trading symbol
            side: Signal side (buy/sell)
            strength: Signal strength (0.0-1.0)
            confidence: Signal confidence (0.0-1.0)
            features: Technical indicators and features
            state: System state including portfolio and market data
            current_price: Current market price

        Returns:
            Calculated position quantity
        """
        try:
            # Extract convergence metrics from features
            l1_l2_agreement = features.get('l1_l2_agreement', 0.5)

            # Calculate technical strength score
            try:
                from core.technical_indicators import calculate_technical_strength_score
                # Create a DataFrame with the latest features for strength calculation
                indicators_df = pd.DataFrame([features])
                technical_strength = calculate_technical_strength_score(indicators_df, symbol)
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate technical strength for {symbol}: {e}")
                technical_strength = 0.5  # Neutral fallback

            # Get market data for enhanced sizing
            market_data = state.get("market_data", {})

            # Use PortfolioManager's enhanced sizing method if available
            try:
                from core.portfolio_manager import PortfolioManager
                # Create a temporary portfolio manager instance for sizing
                temp_pm = PortfolioManager.__new__(PortfolioManager)
                temp_pm.mode = getattr(state.get('portfolio_manager', {}), 'mode', 'simulated')

                # Calculate base position size first
                base_position_size = self._calculate_base_position_size(symbol, side, strength, confidence, current_price)

                # Apply convergence and technical enhancements
                enhanced_size = temp_pm.calculate_convergence_technical_position_size(
                    symbol=symbol,
                    base_position_size=base_position_size,
                    l1_l2_agreement=l1_l2_agreement,
                    technical_strength_score=technical_strength,
                    market_data=market_data
                )

                if enhanced_size > 0:
                    # Convert USD size back to quantity
                    quantity = enhanced_size / current_price if current_price > 0 else 0
                    logger.info(f"🔄 ENHANCED POSITION SIZING for {symbol} {side}: base=${base_position_size:.2f}, enhanced=${enhanced_size:.2f}, quantity={quantity:.6f}")
                    return quantity

            except Exception as e:
                logger.warning(f"⚠️ Enhanced sizing failed for {symbol}, using base calculation: {e}")

            # Fallback to base calculation
            return self._calculate_base_position_size(symbol, side, strength, confidence, current_price)

        except Exception as e:
            logger.error(f"❌ Error in enhanced position sizing for {symbol}: {e}")
            return self._calculate_base_position_size(symbol, side, strength, confidence, current_price)

    def _calculate_base_position_size(self, symbol: str, side: str, strength: float, confidence: float, current_price: float) -> float:
        """
        Calculate base position size using traditional logic.

        Args:
            symbol: Trading symbol
            side: Signal side (buy/sell)
            strength: Signal strength (0.0-1.0)
            confidence: Signal confidence (0.0-1.0)
            current_price: Current market price

        Returns:
            Position quantity
        """
        try:
            if side.lower() == 'sell':
                # For SELL signals, use percentage of current position
                current_position = 0.0
                try:
                    portfolio = state.get("portfolio", {}) if 'state' in locals() else {}
                    symbol_data = portfolio.get(symbol, {})
                    if isinstance(symbol_data, dict):
                        current_position = abs(symbol_data.get("position", 0.0))
                except:
                    current_position = 0.0

                if current_position > 0:
                    # Enhanced confidence-based scaling
                    confidence_multiplier = 1.0
                    if confidence >= 0.9:
                        confidence_multiplier = 2.5
                    elif confidence >= 0.8:
                        confidence_multiplier = 2.0
                    elif confidence >= 0.7:
                        confidence_multiplier = 1.5

                    base_sell_percentage = 0.1 + (strength * confidence * 0.4)
                    sell_percentage = min(0.5, base_sell_percentage * confidence_multiplier)
                    quantity = current_position * sell_percentage

                    logger.info(f"💰 BASE SELL POSITION SIZING: {symbol} position={current_position:.6f}, conf={confidence:.3f}→{confidence_multiplier:.1f}x, selling {sell_percentage:.1%} = {quantity:.6f}")
                    return quantity
                else:
                    logger.warning(f"⚠️ No position to sell for {symbol}")
                    return 0.0
            else:
                # For BUY signals, use fixed quantity with scaling
                base_quantity = 0.05 if symbol == 'BTCUSDT' else 0.5

                # Enhanced confidence-based scaling
                confidence_multiplier = 1.0
                if confidence >= 0.9:
                    confidence_multiplier = 2.5
                elif confidence >= 0.8:
                    confidence_multiplier = 2.0
                elif confidence >= 0.7:
                    confidence_multiplier = 1.5

                quantity = base_quantity * (1 + strength) * confidence * confidence_multiplier

                logger.info(f"💰 BASE BUY POSITION SIZING: {symbol} base={base_quantity:.4f}, strength={strength:.3f}, conf={confidence:.3f}→{confidence_multiplier:.1f}x, final={quantity:.6f}")
                return quantity

        except Exception as e:
            logger.error(f"❌ Error calculating base position size for {symbol}: {e}")
            return 0.01  # Minimum fallback quantity

    def compose_signals_with_rebalance_awareness(
        self,
        l2_signals: list,
        l3_context: dict,
        portfolio_state: dict
    ) -> list:
        """
        Componer señales con awareness de necesidad de rebalanceo.
        """
        # Verificar si portfolio necesita liquidez urgente
        usdt_pct = portfolio_state.get('usdt_percentage', 0)
        target_usdt_pct = l3_context.get('asset_allocation', {}).get('USDT', 0.65)

        if usdt_pct < 0.05 and target_usdt_pct > 0.50:
            logger.warning(f"🚨 CRITICAL LIQUIDITY: {usdt_pct*100:.1f}% << {target_usdt_pct*100:.1f}%")
            logger.warning("🔄 FORCING SELL SIGNALS for rebalance")

            # Forzar señales SELL para recuperar liquidez
            for signal in l2_signals:
                if signal['symbol'] in ['BTCUSDT', 'ETHUSDT']:
                    # Convertir a SELL si tiene posición
                    if portfolio_state.get(f"{signal['symbol']}_position", 0) > 0:
                        signal['action'] = 'sell'
                        signal['confidence'] = 0.75  # Alta confianza para rebalanceo
                        signal['reason'] = 'forced_rebalance_liquidity'
                        logger.info(f"💰 FORCED SELL: {signal['symbol']} for liquidity recovery")

        return l2_signals
