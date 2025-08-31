"""
Multi-Timeframe Technical Analysis
=================================
Análisis técnico en múltiples timeframes para L2_tactic
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.logging import logger
from ..models import TacticalSignal

class MultiTimeframeTechnical:
    """
    Generador de señales técnicas multi-timeframe
    """
    
    def __init__(self, config):
        self.config = config
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.indicators = ['rsi', 'macd', 'bollinger', 'sma', 'ema']
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales técnicas para múltiples timeframes
        """
        signals = []
        
        try:
            # Procesar cada símbolo
            universe = getattr(self.config.signals, 'universe', ['BTC/USDT', 'ETH/USDT'])
            
            for symbol in universe:
                if symbol == 'USDT':  # Skip stablecoin
                    continue
                    
                symbol_data = market_data.get(symbol, {})
                if not symbol_data:
                    continue
                
                # Generar señales por timeframe
                tf_signals = await self._analyze_timeframes(symbol, symbol_data)
                signals.extend(tf_signals)
            
            logger.info(f"📊 Señales técnicas multi-timeframe generadas: {len(signals)}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error en análisis técnico multi-timeframe: {e}")
            return []
    
    async def _analyze_timeframes(self, symbol: str, data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Analiza múltiples timeframes para un símbolo
        """
        signals = []
        
        try:
            # Obtener datos OHLCV
            ohlcv = data.get('ohlcv', {})
            indicators = data.get('indicators', {})
            
            if not ohlcv:
                return signals
            
            # Análisis RSI
            rsi_signal = self._analyze_rsi(symbol, indicators)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # Análisis MACD
            macd_signal = self._analyze_macd(symbol, indicators)
            if macd_signal:
                signals.append(macd_signal)
            
            # Análisis Bollinger Bands
            bb_signal = self._analyze_bollinger(symbol, indicators, ohlcv)
            if bb_signal:
                signals.append(bb_signal)
            
            # Análisis de tendencia (SMA/EMA)
            trend_signal = self._analyze_trend(symbol, indicators, ohlcv)
            if trend_signal:
                signals.append(trend_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error analizando timeframes para {symbol}: {e}")
            return []
    
    def _analyze_rsi(self, symbol: str, indicators: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Análisis RSI
        """
        try:
            rsi = indicators.get('rsi')
            if rsi is None:
                return None
            
            # Señal de sobrecompra/sobreventa
            if rsi < 30:  # Sobreventa
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='rsi_oversold',
                    strength=min((30 - rsi) / 10, 1.0),  # Strength based on how oversold
                    confidence=0.7,
                    side='buy',
                    features={'rsi': rsi, 'condition': 'oversold'},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'RSI', 'threshold': 30}
                )
            elif rsi > 70:  # Sobrecompra
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='rsi_overbought',
                    strength=min((rsi - 70) / 20, 1.0),  # Strength based on how overbought
                    confidence=0.7,
                    side='sell',
                    features={'rsi': rsi, 'condition': 'overbought'},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'RSI', 'threshold': 70}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en análisis RSI: {e}")
            return None
    
    def _analyze_macd(self, symbol: str, indicators: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Análisis MACD
        """
        try:
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            macd_histogram = indicators.get('macd_histogram')
            
            if None in [macd, macd_signal, macd_histogram]:
                return None
            
            # Señal de cruce MACD
            if macd > macd_signal and macd_histogram > 0:  # Bullish crossover
                strength = min(abs(macd - macd_signal) * 100, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='macd_bullish',
                    strength=strength,
                    confidence=0.6,
                    side='buy',
                    features={'macd': macd, 'signal': macd_signal, 'histogram': macd_histogram},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'MACD', 'type': 'bullish_crossover'}
                )
            elif macd < macd_signal and macd_histogram < 0:  # Bearish crossover
                strength = min(abs(macd - macd_signal) * 100, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='macd_bearish',
                    strength=strength,
                    confidence=0.6,
                    side='sell',
                    features={'macd': macd, 'signal': macd_signal, 'histogram': macd_histogram},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'MACD', 'type': 'bearish_crossover'}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en análisis MACD: {e}")
            return None
    
    def _analyze_bollinger(self, symbol: str, indicators: Dict[str, Any], ohlcv: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Análisis Bollinger Bands
        """
        try:
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            bb_middle = indicators.get('bb_middle')
            close = ohlcv.get('close')
            
            if None in [bb_upper, bb_lower, bb_middle, close]:
                return None
            
            # Señales de Bollinger Bands
            if close <= bb_lower:  # Price at lower band - potential buy
                strength = min((bb_lower - close) / bb_lower * 100, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='bb_oversold',
                    strength=strength,
                    confidence=0.5,
                    side='buy',
                    features={'close': close, 'bb_lower': bb_lower, 'bb_middle': bb_middle},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'Bollinger', 'type': 'lower_touch'}
                )
            elif close >= bb_upper:  # Price at upper band - potential sell
                strength = min((close - bb_upper) / bb_upper * 100, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='bb_overbought',
                    strength=strength,
                    confidence=0.5,
                    side='sell',
                    features={'close': close, 'bb_upper': bb_upper, 'bb_middle': bb_middle},
                    timestamp=datetime.now().timestamp(),
                    metadata={'indicator': 'Bollinger', 'type': 'upper_touch'}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en análisis Bollinger: {e}")
            return None
    
    def _analyze_trend(self, symbol: str, indicators: Dict[str, Any], ohlcv: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        Análisis de tendencia con SMA/EMA
        """
        try:
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            ema_12 = indicators.get('ema_12')
            ema_26 = indicators.get('ema_26')
            close = ohlcv.get('close')
            
            if close is None:
                return None
            
            # Señal de cruce de medias móviles (Golden/Death Cross)
            if sma_20 and sma_50:
                if sma_20 > sma_50 and close > sma_20:  # Golden cross + price above
                    strength = min((close - sma_20) / sma_20 * 100, 1.0)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='trend_bullish',
                        strength=strength,
                        confidence=0.6,
                        side='buy',
                        features={'close': close, 'sma_20': sma_20, 'sma_50': sma_50},
                        timestamp=datetime.now().timestamp(),
                        metadata={'indicator': 'SMA_Cross', 'type': 'golden_cross'}
                    )
                elif sma_20 < sma_50 and close < sma_20:  # Death cross + price below
                    strength = min((sma_20 - close) / sma_20 * 100, 1.0)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='trend_bearish',
                        strength=strength,
                        confidence=0.6,
                        side='sell',
                        features={'close': close, 'sma_20': sma_20, 'sma_50': sma_50},
                        timestamp=datetime.now().timestamp(),
                        metadata={'indicator': 'SMA_Cross', 'type': 'death_cross'}
                    )
            
            # Señal EMA
            if ema_12 and ema_26:
                if ema_12 > ema_26 and close > ema_12:  # EMA bullish
                    strength = min((close - ema_12) / ema_12 * 50, 0.8)  # Moderate strength
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='ema_bullish',
                        strength=strength,
                        confidence=0.5,
                        side='buy',
                        features={'close': close, 'ema_12': ema_12, 'ema_26': ema_26},
                        timestamp=datetime.now().timestamp(),
                        metadata={'indicator': 'EMA_Cross', 'type': 'bullish'}
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en análisis de tendencia: {e}")
            return None

# Función helper para consenso multi-timeframe
async def resample_and_consensus(data: Dict[str, Any], timeframes: List[str]) -> Dict[str, Any]:
    """
    Resampling y consenso entre múltiples timeframes
    """
    try:
        consensus = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'confidence': 0.0,
            'timeframes_analyzed': len(timeframes)
        }
        
        # Simple consensus based on available data
        # En una implementación real, aquí harías resampling de los datos OHLCV
        for tf in timeframes:
            tf_data = data.get(f'tf_{tf}', {})
            if tf_data:
                # Analyze each timeframe (simplified)
                if tf_data.get('trend', 'neutral') == 'bullish':
                    consensus['bullish_signals'] += 1
                elif tf_data.get('trend', 'neutral') == 'bearish':
                    consensus['bearish_signals'] += 1
                else:
                    consensus['neutral_signals'] += 1
        
        # Calculate overall confidence
        total_signals = sum([consensus['bullish_signals'], consensus['bearish_signals'], consensus['neutral_signals']])
        if total_signals > 0:
            max_signals = max(consensus['bullish_signals'], consensus['bearish_signals'], consensus['neutral_signals'])
            consensus['confidence'] = max_signals / total_signals
        
        return consensus
        
    except Exception as e:
        logger.error(f"❌ Error en consenso multi-timeframe: {e}")
        return {'confidence': 0.0, 'timeframes_analyzed': 0}
