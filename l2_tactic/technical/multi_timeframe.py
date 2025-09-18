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
        # Definir umbrales ajustados
        self.rsi_overbought = 60  # Reducido de 70
        self.rsi_oversold = 40   # Aumentado de 30
        self.macd_strength_factor = 50  # Reducido de 100 para mayor sensibilidad
        self.bb_strength_factor = 50   # Reducido de 100 para mayor sensibilidad
        self.sma_strength_factor = 50  # Reducido de 100 para mayor sensibilidad

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula indicadores técnicos para un DataFrame."""
        results = {}
        
        if data.empty:
            return results
            
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            results['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            results['macd'] = exp1 - exp2
            results['macd_signal'] = results['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            results['bb_upper'] = sma + (std * 2)
            results['bb_lower'] = sma - (std * 2)
            results['bb_mid'] = sma
            
            # ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            results['atr'] = true_range.rolling(window=14).mean()
            
            # ADX (Average Directional Index)
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean())
            minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            results['adx'] = dx.ewm(alpha=1/14).mean()
            
            # Momentum
            results['mom'] = data['close'].diff(10)
            
            # Volume
            results['volume'] = data['volume']
            
            # Moving Averages
            results['close_sma'] = data['close'].rolling(window=20).mean()
            results['close_ema'] = data['close'].ewm(span=20, adjust=False).mean()
            
            # Volatilidad
            returns = data['close'].pct_change()
            results['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Trend (usando la pendiente del SMA)
            results['trend'] = results['close_sma'].diff()
            
            # Momentum adicional (ROC - Rate of Change)
            results['momentum'] = data['close'].pct_change(periods=10) * 100
            
        except Exception as e:
            logger.error(f"❌ Error calculando indicadores técnicos: {e}")
            
        return results
        
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame], technical_indicators: Dict[str, pd.DataFrame]) -> List[TacticalSignal]:
        """
        Genera señales técnicas para múltiples timeframes
        """
        signals = []
        for symbol in market_data:
            try:
                data = technical_indicators.get(symbol, pd.DataFrame())
                if not isinstance(data, pd.DataFrame) or data.empty:
                    logger.warning(f"⚠️ Datos técnicos vacíos o inválidos para {symbol}: {data.shape if isinstance(data, pd.DataFrame) else type(data)}")
                    continue

                # Obtener la última fila de indicadores
                latest = data.iloc[-1]
                rsi = float(latest.get('rsi', 50.0))
                macd = float(latest.get('macd', 0.0))
                macd_signal = float(latest.get('macd_signal', 0.0))
                bollinger_upper = float(latest.get('bollinger_upper', 0.0))
                bollinger_lower = float(latest.get('bollinger_lower', 0.0))
                sma_20 = float(latest.get('sma_20', 0.0))
                sma_50 = float(latest.get('sma_50', 0.0))
                ema_12 = float(latest.get('ema_12', 0.0))
                ema_26 = float(latest.get('ema_26', 0.0))
                close = float(latest.get('close', 0.0))

                logger.debug(f"Indicadores para {symbol}: rsi={rsi:.2f}, macd={macd:.2f}, macd_signal={macd_signal:.2f}, "
                            f"bb_upper={bollinger_upper:.2f}, bb_lower={bollinger_lower:.2f}, "
                            f"sma_20={sma_20:.2f}, sma_50={sma_50:.2f}, ema_12={ema_12:.2f}, "
                            f"ema_26={ema_26:.2f}, close={close:.2f}")

                # Señales RSI
                if rsi > self.rsi_overbought:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_rsi_overbought",
                        strength=-0.7,
                        confidence=0.7,
                        side="sell",
                        source="technical",
                        features={"rsi": rsi},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "rsi"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: sell (RSI overbought, rsi={rsi:.2f})")
                elif rsi < self.rsi_oversold:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_rsi_oversold",
                        strength=0.7,
                        confidence=0.7,
                        side="buy",
                        source="technical",
                        features={"rsi": rsi},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "rsi"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: buy (RSI oversold, rsi={rsi:.2f})")
                else:
                    logger.debug(f"[DEBUG] {symbol} - No señal RSI: rsi={rsi:.2f}")

                # Señales MACD
                if macd > macd_signal and macd > 0:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_macd_bullish",
                        strength=0.7,
                        confidence=0.7,
                        side="buy",
                        source="technical",
                        features={"macd": macd, "macd_signal": macd_signal},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "macd"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: buy (MACD bullish, macd={macd:.2f})")
                elif macd < macd_signal and macd < 0:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_macd_bearish",
                        strength=-0.7,
                        confidence=0.7,
                        side="sell",
                        source="technical",
                        features={"macd": macd, "macd_signal": macd_signal},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "macd"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: sell (MACD bearish, macd={macd:.2f})")
                else:
                    logger.debug(f"[DEBUG] {symbol} - No señal MACD: macd={macd:.2f}, signal={macd_signal:.2f}")

                # Señales Bollinger
                if close > bollinger_upper:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_bb_overbought",
                        strength=-0.7,
                        confidence=0.7,
                        side="sell",
                        source="technical",
                        features={"close": close, "bollinger_upper": bollinger_upper},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "bollinger"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: sell (Bollinger overbought, close={close:.2f})")
                elif close < bollinger_lower:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        signal_type="technical_bb_oversold",
                        strength=0.7,
                        confidence=0.7,
                        side="buy",
                        source="technical",
                        features={"close": close, "bollinger_lower": bollinger_lower},
                        timestamp=pd.Timestamp.now(),
                        metadata={"indicator": "bollinger"}
                    ))
                    logger.info(f"📈 Señal técnica para {symbol}: buy (Bollinger oversold, close={close:.2f})")
                else:
                    logger.debug(f"[DEBUG] {symbol} - No señal Bollinger: close={close:.2f}, bb_lower={bollinger_lower:.2f}, bb_upper={bollinger_upper:.2f}")

            except Exception as e:
                logger.error(f"❌ Error generando señales técnicas para {symbol}: {e}")
        logger.info(f"📊 Señales técnicas multi-timeframe generadas: {len(signals)}")
        return signals
    
    async def _analyze_timeframes(self, symbol: str, data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Analiza múltiples timeframes para un símbolo
        """
        signals = []
        
        try:
            # Obtener datos OHLCV e indicadores
            ohlcv = data.get('ohlcv', {})
            indicators = data.get('indicators', {})
            logger.debug(f"📊 Datos para {symbol}: OHLCV={ohlcv}, Indicadores={indicators}")
            if not indicators:
                logger.warning(f"⚠️ No hay indicadores para {symbol}, no se generarán señales técnicas")
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
            
            if rsi < self.rsi_oversold:
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='rsi_oversold',
                    strength=min((self.rsi_oversold - rsi) / 15, 1.0),
                    confidence=0.7,
                    side='buy',
                    source="technical",
                    features={'rsi': rsi, 'condition': 'oversold'},
                    timestamp=pd.Timestamp.now(),
                    metadata={'indicator': 'RSI', 'threshold': self.rsi_oversold}
                )
            elif rsi > self.rsi_overbought:
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='rsi_overbought',
                    strength=min((rsi - self.rsi_overbought) / 15, 1.0),
                    confidence=0.7,
                    side='sell',
                    source="technical",
                    features={'rsi': rsi, 'condition': 'overbought'},
                    timestamp=pd.Timestamp.now(),
                    metadata={'indicator': 'RSI', 'threshold': self.rsi_overbought}
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
            if None in [macd, macd_signal]:
                return None
            
            if macd > macd_signal:
                strength = min(abs(macd - macd_signal) * self.macd_strength_factor, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='macd_bullish',
                    strength=strength,
                    confidence=0.7,
                    side='buy',
                    source="technical",
                    features={'macd': macd, 'macd_signal': macd_signal},
                    timestamp=pd.Timestamp.now(),
                    metadata={'indicator': 'MACD', 'type': 'bullish_crossover'}
                )
            elif macd < macd_signal:
                strength = min(abs(macd - macd_signal) * self.macd_strength_factor, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='macd_bearish',
                    strength=strength,
                    confidence=0.7,
                    side='sell',
                    source="technical",
                    features={'macd': macd, 'macd_signal': macd_signal},
                    timestamp=pd.Timestamp.now(),
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
            close = ohlcv.get('close')
            bb_upper = indicators.get('bollinger_upper')
            bb_lower = indicators.get('bollinger_lower')
            bb_middle = indicators.get('bb_middle')
            if None in [close, bb_upper, bb_lower]:
                return None
            
            if close < bb_lower:
                strength = min((bb_lower - close) / close * self.bb_strength_factor, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='bb_oversold',
                    strength=strength,
                    confidence=0.5,
                    side='buy',
                    source="technical",
                    features={'close': close, 'bb_lower': bb_lower, 'bb_middle': bb_middle},
                    timestamp=pd.Timestamp.now(),
                    metadata={'indicator': 'Bollinger', 'type': 'lower_touch'}
                )
            elif close >= bb_upper:
                strength = min((close - bb_upper) / close * self.bb_strength_factor, 1.0)
                return TacticalSignal(
                    symbol=symbol,
                    signal_type='bb_overbought',
                    strength=strength,
                    confidence=0.5,
                    side='sell',
                    source="technical",
                    features={'close': close, 'bb_upper': bb_upper, 'bb_middle': bb_middle},
                    timestamp=pd.Timestamp.now(),
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
            
            if sma_20 and sma_50:
                if sma_20 > sma_50 and close > sma_20:
                    strength = min((close - sma_20) / sma_20 * self.sma_strength_factor, 1.0)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='trend_bullish',
                        strength=strength,
                        confidence=0.6,
                        side='buy',
                        source="technical",
                        features={'close': close, 'sma_20': sma_20, 'sma_50': sma_50},
                        timestamp=pd.Timestamp.now(),
                        metadata={'indicator': 'SMA_Cross', 'type': 'golden_cross'}
                    )
                elif sma_20 < sma_50 and close < sma_20:
                    strength = min((sma_20 - close) / sma_20 * self.sma_strength_factor, 1.0)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='trend_bearish',
                        strength=strength,
                        confidence=0.6,
                        side='sell',
                        source="technical",
                        features={'close': close, 'sma_20': sma_20, 'sma_50': sma_50},
                        timestamp=pd.Timestamp.now(),
                        metadata={'indicator': 'SMA_Cross', 'type': 'death_cross'}
                    )
            
            if ema_12 and ema_26:
                if ema_12 > ema_26 and close > ema_12:
                    strength = min((close - ema_12) / ema_12 * self.sma_strength_factor, 0.8)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='ema_bullish',
                        strength=strength,
                        confidence=0.5,
                        side='buy',
                        source="technical",
                        features={'close': close, 'ema_12': ema_12, 'ema_26': ema_26},
                        timestamp=pd.Timestamp.now(),
                        metadata={'indicator': 'EMA_Cross', 'type': 'bullish'}
                    )
                elif ema_12 < ema_26 and close < ema_12:
                    strength = min((ema_12 - close) / ema_12 * self.sma_strength_factor, 0.8)
                    return TacticalSignal(
                        symbol=symbol,
                        signal_type='ema_bearish',
                        strength=strength,
                        confidence=0.5,
                        side='sell',
                        source="technical",
                        features={'close': close, 'ema_12': ema_12, 'ema_26': ema_26},
                        timestamp=pd.Timestamp.now(),
                        metadata={'indicator': 'EMA_Cross', 'type': 'bearish'}
                    )
            return None
        except Exception as e:
            logger.error(f"❌ Error en análisis de tendencia: {e}")
            return None

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
        
        for tf in timeframes:
            tf_data = data.get(f'tf_{tf}', {})
            if tf_data:
                if tf_data.get('trend', 'neutral') == 'bullish':
                    consensus['bullish_signals'] += 1
                elif tf_data.get('trend', 'neutral') == 'bearish':
                    consensus['bearish_signals'] += 1
                else:
                    consensus['neutral_signals'] += 1
        
        total_signals = sum([consensus['bullish_signals'], consensus['bearish_signals'], consensus['neutral_signals']])
        if total_signals > 0:
            max_signals = max(consensus['bullish_signals'], consensus['bearish_signals'], consensus['neutral_signals'])
            consensus['confidence'] = max_signals / total_signals
        
        return consensus
    except Exception as e:
        logger.error(f"❌ Error en consenso multi-timeframe: {e}")
        return {'confidence': 0.0, 'timeframes_analyzed': 0}