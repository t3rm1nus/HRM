# l2_tactic/signal_composer.py
from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from .config import *  # Import L2 config settings
from .models import TacticalSignal
from .utils import safe_float
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
    """
    def compose_signal(self, symbol: str, base_signal: TacticalSignal, indicators: Dict, state: Dict) -> Optional[TacticalSignal]:
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
        try:
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
            logger.debug(f"🔍 THRESHOLD CHECK for {symbol}: conf={confidence:.3f} >= {self.min_conf:.3f}, strength={strength:.3f} >= {self.min_strength:.3f}")

            # Validar fuerza y confianza de la señal base
            if confidence < self.min_conf or strength < self.min_strength:
                logger.debug(f"Señal {symbol} rechazada por baja confianza/fuerza: {confidence:.2f}/{strength:.2f}")
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
            
            logger.debug(f"✅ Señal compuesta para {symbol}: {composed_signal.side} (conf={confidence:.2f}, strength={strength:.2f})")
            return composed_signal
            
        except Exception as e:
            logger.error(f"❌ Error componiendo señal para {symbol}: {e}", exc_info=True)
            return None

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
        # FURTHER LOWER THRESHOLDS TO ALLOW MORE SIGNALS THROUGH
        self.min_conf = getattr(config, "min_signal_confidence", 0.05)  # Lower from 0.1 to 0.05
        self.min_strength = getattr(config, "min_signal_strength", 0.005)  # Lower from 0.01 to 0.005

        # ✅ FIXED: Use proper attribute access for SignalConfig dataclass
        self.conflict_tie_threshold = getattr(config, "conflict_tie_threshold", config_defaults["conflict_tie_threshold"])
        
        # Inicializar histórico de señales
        self._last_signals = {}
        self.keep_both_when_far = True  # Forzar keep_both para señales con alta confianza
        self.high_conf_threshold = 0.7   # Nueva: umbral para señales de alta confianza

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
                # Separar señales por lado (buy/sell)
                buy_signals = [s for s in sym_signals if s.side.lower() == 'buy']
                sell_signals = [s for s in sym_signals if s.side.lower() == 'sell']
                
                # Calcular fuerza por lado
                buy_strength = sum(s.strength * self._get_dynamic_weight(s) for s in buy_signals) / total_weight
                sell_strength = sum(s.strength * self._get_dynamic_weight(s) for s in sell_signals) / total_weight
                
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
                
                # Solo generar señal si supera umbrales mínimos
                if avg_strength >= self.min_strength and avg_confidence >= self.min_conf:
                    # Calcular cantidad base según el símbolo
                    base_quantity = 0.01 if symbol == 'BTCUSDT' else 0.1  # ETH y otros
                    
                    # Ajustar por fuerza de la señal
                    quantity = base_quantity * (1 + avg_strength)
                    
                    # Calcular stop-loss y take-profit
                    stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                        current_price, side, avg_strength, avg_confidence, features
                    )
                    
                    logger.info(f"💰 Calculando cantidad para {symbol}: base={base_quantity:.4f}, strength={avg_strength:.3f}, final={quantity:.4f}")
                    logger.info(f"🛡️ SL/TP para {symbol}: SL={stop_loss:.2f}, TP={take_profit:.2f}")
                    
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
                            'signal_type': 'composed'
                        }
                )
                composed_signals.append(composed_signal)
                logger.debug(f"✅ Señal compuesta para {symbol}: side={dominant_signal.side}, strength={avg_strength:.3f}, confidence={avg_confidence:.3f}")

        # Resolver conflictos y filtrar por thresholds
        filtered_signals = self._resolve_conflicts_and_filter(composed_signals)
        
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

                        # Calcular cantidad base según el símbolo
                        base_quantity = 0.01 if symbol == 'BTCUSDT' else 0.1  # ETH y otros
                        # Ajustar por fuerza de la señal
                        calculated_quantity = base_quantity * (1 + avg_strength)
                        
                        # Calcular stop-loss y take-profit
                        stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                            current_price, side, avg_strength, avg_confidence, sigs[0].features or {}
                        )
                        
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
                            quantity=calculated_quantity,  # Use calculated quantity
                            stop_loss=stop_loss,
                            take_profit=take_profit,
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
