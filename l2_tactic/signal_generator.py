import asyncio
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from core.logging import logger
from .models import TacticalSignal, L2State
from .technical.multi_timeframe import MultiTimeframeTechnical
from .risk_overlay import RiskOverlay
from .signal_composer import SignalComposer
from .finrl_integration import FinRLProcessor
from .signal_validator import validate_signal_list, validate_tactical_signal, create_fallback_signal

class L2TacticProcessor:
    """
    Generador de señales tácticas para L2.
    """

    def __init__(self, config):
        """Inicializa el procesador táctico."""
        self.config = config
        self.multi_timeframe = MultiTimeframeTechnical(config)
        self.risk_overlay = RiskOverlay(config)
        # ✅ FIXED: Pass the correct config object to SignalComposer
        signal_config = getattr(config, 'signals', config) if hasattr(config, 'signals') else config
        self.signal_composer = SignalComposer(signal_config)

        # Usar el atributo model_path de la configuración L2Config
        model_path = getattr(config.ai_model, "model_path", None) if hasattr(config, 'ai_model') else None
        logger.info(f"🔍 Intentando cargar modelo desde: {model_path}")
        if model_path:
            logger.info(f"🔍 Ruta absoluta del modelo: {os.path.abspath(model_path)}")
            logger.info(f"🔍 Existe archivo?: {os.path.exists(model_path)}")
        if not model_path:
            logger.warning("⚠️ No se encontró model_path en la configuración L2. Usando fallback.")
            model_path = "models/L2/deepseek.zip"
            logger.info(f"🔍 Usando ruta fallback: {model_path}")
            logger.info(f"🔍 Ruta absoluta fallback: {os.path.abspath(model_path)}")
        
        try:
            self.finrl_processor = FinRLProcessor(model_path)
        except Exception as e:
            logger.error(f"❌ Error cargando FinRL processor: {e}")
            self.finrl_processor = None

    async def process_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Procesa datos de mercado para generar señales tácticas.
        """
        try:
            # Try different market data sources in order of preference
            market_data = state.get("market_data_simple") or state.get("market_data", {})
            if not market_data:
                logger.warning("⚠️ L2: No hay datos de mercado disponibles")
                return []

            signals = []
            
            for symbol, data in market_data.items():
                # Handle different data formats from backtesting
                if isinstance(data, dict):
                    # Backtesting format: dict with 'historical_data' key
                    if 'historical_data' in data and isinstance(data['historical_data'], pd.DataFrame):
                        df = data['historical_data']
                        if len(df) >= 200:
                            logger.info(f"✅ L2: Usando datos históricos para {symbol}: {len(df)} puntos")
                        else:
                            logger.warning(f"⚠️ L2: Datos históricos insuficientes para {symbol}: {len(df)} < 200 puntos requeridos")
                            continue
                    else:
                        # Single data point format - skip during warmup
                        logger.warning(f"⚠️ L2: Datos insuficientes para {symbol}: formato single point (warmup)")
                        continue
                elif isinstance(data, pd.DataFrame):
                    # Direct DataFrame format
                    df = data
                    if df.empty or len(df) < 200:
                        logger.warning(f"⚠️ L2: Datos insuficientes para {symbol}: {len(df)} < 200 puntos requeridos")
                        continue
                else:
                    logger.warning(f"⚠️ L2: Formato de datos inválido para {symbol}: {type(data)}")
                    continue

                # Calcular indicadores técnicos multi-timeframe
                indicators = self.multi_timeframe.calculate_technical_indicators(df)
                
                finrl_signal = await self.finrl_processor.get_action(
                    state, 
                    symbol,
                    indicators
                )
                if not finrl_signal:
                    logger.warning(f"⚠️ No hay señal FinRL para {symbol}")
                    continue

                # ✅ CRITICAL FIX: Improved L3→L2 synchronization with proper context sharing
                l3_context = self._check_l3_context_freshness(state, symbol, df)

                # CRITICAL FIX: Always try to use L3 context from cache first, fallback to l3_output
                l3_context_cache = state.get("l3_context_cache", {})
                l3_output = l3_context_cache.get("last_output", {})

                # If no L3 data in cache, try fallback to l3_output
                if not l3_output:
                    l3_output = state.get("l3_output", {})
                    logger.debug(f"Using l3_output fallback for {symbol}")

                if l3_output and l3_context['is_fresh']:
                    logger.info(f"✅ L3 context fresh for {symbol} - using FinRL + L3 context")
                    # Use FinRL signal with L3 context available
                    risk_filtered = finrl_signal

                    # Apply risk overlay with L3 context
                    risk_market_data = {symbol: df}
                    risk_signals = await self.risk_overlay.generate_risk_signals(
                        market_data=risk_market_data,
                        portfolio_data=state.get("portfolio", {}),
                        l3_context=l3_output  # Pass L3 context to risk overlay
                    )

                    if risk_signals:
                        logger.info(f"⚠️ Risk signals detected for {symbol}: {len(risk_signals)}")
                        # Apply risk-adjusted logic
                        risk_filtered = self._apply_risk_adjustment(finrl_signal, risk_signals, l3_output)

                elif l3_output and not l3_context['is_fresh']:
                    logger.warning(f"⚠️ L3 context exists but stale for {symbol} - using tactical fallback")
                    # L3 exists but is stale - use tactical analysis
                    tactical_signal = self._generate_tactical_fallback_signal(symbol, df, indicators)
                    if tactical_signal:
                        logger.info(f"🎯 Tactical fallback signal for {symbol}: {tactical_signal.side} (conf={tactical_signal.confidence:.3f})")
                        risk_filtered = tactical_signal
                    else:
                        risk_filtered = finrl_signal
                else:
                    logger.warning(f"❌ No L3 context available for {symbol} - using FinRL only")
                    # No L3 context at all - use FinRL signal as-is
                    risk_filtered = finrl_signal

                tactical_signal = self.signal_composer.compose_signal(
                    symbol=symbol,
                    base_signal=risk_filtered,
                    indicators=indicators,
                    state=state
                )

                if tactical_signal:
                    validated_signal = validate_tactical_signal(tactical_signal)
                    if validated_signal:
                        signals.append(validated_signal)
                        logger.info(f"✅ L2: Señal generada para {symbol}: {tactical_signal.side} (conf={tactical_signal.confidence:.3f}, strength={tactical_signal.strength:.3f})")
                    else:
                        logger.warning(f"⚠️ Invalid signal for {symbol}, creating fallback")
                        fallback = create_fallback_signal(symbol, "invalid_tactical")
                        signals.append(fallback)
                else:
                    logger.warning(f"⚠️ No tactical signal composed for {symbol}, creating fallback")
                    fallback = create_fallback_signal(symbol, "no_composition")
                    signals.append(fallback)

            # Validate signals before returning
            validated_signals = validate_signal_list(signals)
            return validated_signals
            
        except Exception as e:
            logger.error(f"❌ Error procesando señales L2: {e}", exc_info=True)
            return []

    FINRL_OBS_DIM = 257
    BASE_FEATURES_DIM = 246
    CROSS_FEATURES_DIM = 11
    PREFERRED_BASE_COLS: Optional[List[str]] = None

    def _select_base_features_row(self, features_df: pd.DataFrame) -> List[float]:
        """Selects up to 52 numeric features from the last row of the features DataFrame."""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * self.BASE_FEATURES_DIM

        last_row = features_df.iloc[-1]

        if self.PREFERRED_BASE_COLS:
            cols = [c for c in self.PREFERRED_BASE_COLS if c in features_df.columns]
        else:
            numeric_cols = [
                c for c in features_df.columns
                if pd.api.types.is_numeric_dtype(features_df[c]) or features_df[c].dtype in ['int64', 'float64', 'int32', 'float32']
            ]
            cols = sorted(numeric_cols)

        values = []
        for c in cols[:self.BASE_FEATURES_DIM]:
            try:
                v = last_row[c]
                values.append(float(v) if np.isfinite(v) else 0.0)
            except Exception:
                values.append(0.0)

        if len(values) < self.BASE_FEATURES_DIM:
            values.extend([0.0] * (self.BASE_FEATURES_DIM - len(values)))

        return values[:self.BASE_FEATURES_DIM]

    def _get_key_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Get key technical indicators and metrics from features"""
        last_row = features_df.iloc[-1]
        return {
            'rsi': float(last_row.get('rsi', 50.0)),
            'macd': float(last_row.get('macd', 0.0)),
            'macd_signal': float(last_row.get('macd_signal', 0.0)),
            'sma_20': float(last_row.get('sma_20', 0.0)),
            'sma_50': float(last_row.get('sma_50', 0.0)),
            'bollinger_upper': float(last_row.get('bollinger_upper', 0.0)),
            'bollinger_lower': float(last_row.get('bollinger_lower', 0.0)),
            'vol_zscore': float(last_row.get('vol_zscore', 0.0))
        }

    def _compute_eth_btc_ratio(self, market_data: Dict[str, pd.DataFrame],
                              features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute ETH/BTC close ratio with fallback"""
        try:
            for src in ("features", "market"):
                if src == "features":
                    eth = features_by_symbol.get("ETHUSDT")
                    btc = features_by_symbol.get("BTCUSDT")
                    eth_close = float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and "close" in eth.columns and not eth.empty else None
                    btc_close = float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and "close" in btc.columns and not btc.empty else None
                else:
                    eth = market_data.get("ETHUSDT")
                    btc = market_data.get("BTCUSDT")
                    eth_close = float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and not eth.empty else None
                    btc_close = float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and not btc.empty else None

                if eth_close is not None and btc_close not in (None, 0.0):
                    return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Compute 30-sample correlation between BTC and ETH close returns"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0
            if eth.empty or btc.empty or "close" not in eth.columns or "close" not in btc.columns:
                return 0.0
            n = 30
            eth_close = eth["close"].astype(float).tail(n)
            btc_close = btc["close"].astype(float).tail(n)
            common_idx = eth_close.index.intersection(btc_close.index)
            eth_close = eth_close.loc[common_idx]
            btc_close = btc_close.loc[common_idx]
            if len(eth_close) < 3:
                return 0.0
            eth_ret = eth_close.pct_change().dropna()
            btc_ret = btc_close.pct_change().dropna()
            common_idx = eth_ret.index.intersection(btc_ret.index)
            if len(common_idx) < 3:
                return 0.0
            corr = float(np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])[0, 1])
            if np.isfinite(corr):
                return corr
        except Exception:
            pass
        return 0.0

    def _compute_spread_pct(self, market_data: Dict[str, pd.DataFrame],
                           features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """Compute (BTC - ETH)/BTC as simple spread proxy in %"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = float(eth["close"].iloc[-1])
                btc_close = float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0

    def _build_cross_l3_features(self,
                                market_data: Dict[str, pd.DataFrame],
                                features_by_symbol: Dict[str, pd.DataFrame]) -> List[float]:
        """Build 11 cross/L3 features"""
        feats = []

        feats.append(self._compute_eth_btc_ratio(market_data, features_by_symbol))
        feats.append(self._compute_btc_eth_corr30(market_data))
        feats.append(self._compute_spread_pct(market_data, features_by_symbol))

        def pick_feature(df_map: Dict[str, pd.DataFrame], key: str, default: float) -> float:
            try:
                btc_df = df_map.get("BTCUSDT")
                if isinstance(btc_df, pd.DataFrame) and key in btc_df.columns and not btc_df.empty:
                    v = float(btc_df[key].iloc[-1])
                    return v if np.isfinite(v) else default
            except Exception:
                pass
            return default

        feats.append(pick_feature(features_by_symbol, "l3_regime", 0.0))
        feats.append(pick_feature(features_by_symbol, "l3_risk_appetite", 0.5))
        feats.append(pick_feature(features_by_symbol, "l3_alloc_BTC", 0.0))
        feats.append(pick_feature(features_by_symbol, "l3_alloc_ETH", 0.0))
        feats.append(pick_feature(features_by_symbol, "l3_alloc_CASH", 0.0))

        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame):
                n = 20
                v_eth = eth["volume"].astype(float).tail(n)
                v_btc = btc["volume"].astype(float).tail(n)
                common_idx = v_eth.index.intersection(v_btc.index)
                v_eth = v_eth.loc[common_idx]
                v_btc = v_btc.loc[common_idx]
                ratio = float(v_eth.mean() / v_btc.mean()) if v_btc.mean() != 0 else 0.0
                feats.append(ratio if np.isfinite(ratio) else 0.0)
                v_eth_ret = v_eth.pct_change().dropna()
                v_btc_ret = v_btc.pct_change().dropna()
                common_idx = v_eth_ret.index.intersection(v_btc_ret.index)
                if len(common_idx) >= 3:
                    corr = float(np.corrcoef(v_eth_ret.loc[common_idx], v_btc_ret.loc[common_idx])[0, 1])
                    feats.append(corr if np.isfinite(corr) else 0.0)
                else:
                    feats.append(0.0)
            else:
                feats.extend([0.0, 0.0])
        except Exception:
            feats.extend([0.0, 0.0])

        try:
            btc_f = features_by_symbol.get("BTCUSDT")
            eth_f = features_by_symbol.get("ETHUSDT")
            if (isinstance(btc_f, pd.DataFrame) and "macd" in btc_f.columns and not btc_f.empty and
                isinstance(eth_f, pd.DataFrame) and "macd" in eth_f.columns and not eth_f.empty):
                val = float(btc_f["macd"].iloc[-1]) - float(eth_f["macd"].iloc[-1])
                feats.append(val if np.isfinite(val) else 0.0)
            else:
                feats.append(0.0)
        except Exception:
            feats.append(0.0)

        if len(feats) < self.CROSS_FEATURES_DIM:
            feats.extend([0.0] * (self.CROSS_FEATURES_DIM - len(feats)))
        elif len(feats) > self.CROSS_FEATURES_DIM:
            feats = feats[:self.CROSS_FEATURES_DIM]

        return [float(x) for x in feats]

    async def ai_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de IA usando FinRL
        """
        signals = []
        try:
            if not self.finrl_processor:
                logger.warning("⚠️ FinRL processor no disponible")
                return []
                
            market_data = state.get("mercado", state.get("market_data", {}))
            if not market_data:
                logger.warning("⚠️ No hay datos de mercado para señales AI")
                return []
                
            for symbol, df in market_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"⚠️ Datos inválidos para {symbol}")
                    continue

                # Verificar que hay suficientes datos históricos (>200 puntos)
                if len(df) < 200:
                    logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(df)} < 200 puntos requeridos")
                    continue
                    
                # Calcular indicadores técnicos
                indicators = self.multi_timeframe.calculate_technical_indicators(df)
                
                # Obtener señal de FinRL
                finrl_signal = await self.finrl_processor.get_action(state, symbol, indicators)
                if finrl_signal:
                    # Asegurar que la señal tenga source='ai'
                    finrl_signal.source = 'ai'
                    signals.append(finrl_signal)
                    logger.debug(f"✅ Señal AI generada para {symbol}: {finrl_signal.side}")
                    
        except Exception as e:
            logger.error(f"❌ Error generando señales AI: {e}", exc_info=True)
            
        logger.info(f"🤖 Señales AI generadas: {len(signals)}")
        # Validate signals before returning
        validated_signals = validate_signal_list(signals)
        return validated_signals

    async def technical_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales técnicas basadas en indicadores
        """
        signals = []
        try:
            market_data = state.get("mercado", state.get("market_data", {}))
            if not market_data:
                logger.warning("⚠️ No hay datos de mercado para señales técnicas")
                return []
                
            for symbol, df in market_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"⚠️ Datos inválidos para {symbol}")
                    continue
                    
                # Calcular indicadores técnicos
                indicators = self.multi_timeframe.calculate_technical_indicators(df)
                
                # Generar señal técnica basada en indicadores
                tech_signal = self._generate_technical_signal(symbol, indicators, df)
                if tech_signal:
                    signals.append(tech_signal)
                    logger.debug(f"✅ Señal técnica generada para {symbol}: {tech_signal.side}")
                    
        except Exception as e:
            logger.error(f"❌ Error generando señales técnicas: {e}", exc_info=True)
            
        logger.info(f"📊 Señales técnicas generadas: {len(signals)}")
        # Validate signals before returning
        validated_signals = validate_signal_list(signals)
        return validated_signals

    def _generate_technical_signal(self, symbol: str, indicators: Dict, df: pd.DataFrame) -> Optional[TacticalSignal]:
        """
        Genera una señal técnica basada en indicadores
        """
        try:
            if not indicators:
                return None
                
            # Obtener valores actuales de indicadores
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            
            if not all(hasattr(ind, 'iloc') for ind in [rsi, macd, macd_signal] if ind is not None):
                logger.warning(f"⚠️ Indicadores inválidos para {symbol}")
                return None
                
            current_price = float(df['close'].iloc[-1])
            rsi_val = float(rsi.iloc[-1]) if rsi is not None else 50.0
            macd_val = float(macd.iloc[-1]) if macd is not None else 0.0
            macd_signal_val = float(macd_signal.iloc[-1]) if macd_signal is not None else 0.0
            
            # Lógica de señales técnicas
            signals_count = 0
            buy_signals = 0
            sell_signals = 0
            
            # RSI signals
            if rsi_val < 30:  # Oversold
                buy_signals += 1
                signals_count += 1
            elif rsi_val > 70:  # Overbought
                sell_signals += 1
                signals_count += 1
                
            # MACD signals
            if macd_val > macd_signal_val:  # MACD above signal
                buy_signals += 1
                signals_count += 1
            elif macd_val < macd_signal_val:  # MACD below signal
                sell_signals += 1
                signals_count += 1
                
            # Bollinger Bands signals
            if bb_upper is not None and bb_lower is not None:
                bb_upper_val = float(bb_upper.iloc[-1])
                bb_lower_val = float(bb_lower.iloc[-1])
                
                if current_price <= bb_lower_val:  # Price at lower band
                    buy_signals += 1
                    signals_count += 1
                elif current_price >= bb_upper_val:  # Price at upper band
                    sell_signals += 1
                    signals_count += 1
                    
            # Determinar señal final
            if signals_count == 0:
                return None
                
            if buy_signals > sell_signals:
                side = 'buy'
                strength = buy_signals / signals_count
            elif sell_signals > buy_signals:
                side = 'sell'
                strength = sell_signals / signals_count
            else:
                side = 'hold'
                strength = 0.5
                
            # Calcular confianza basada en la fuerza de los indicadores
            confidence = min(0.9, strength * 0.8 + 0.2)  # Entre 0.2 y 0.9
            
            # Crear features dict
            features = {
                'rsi': rsi_val,
                'macd': macd_val,
                'macd_signal': macd_signal_val,
                'close': current_price,
                'signals_count': signals_count,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals
            }
            
            if bb_upper is not None and bb_lower is not None:
                features.update({
                    'bb_upper': bb_upper_val,
                    'bb_lower': bb_lower_val
                })
                
            return TacticalSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                confidence=confidence,
                signal_type='technical',
                source='technical',
                features=features,
                timestamp=pd.Timestamp.now(),
                metadata={
                    'indicators_used': list(indicators.keys()),
                    'signals_breakdown': {
                        'buy': buy_signals,
                        'sell': sell_signals,
                        'total': signals_count
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error generando señal técnica para {symbol}: {e}")
            return None

    def _check_l3_context_freshness(self, state: Dict[str, Any], symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if L3 context is fresh or stale with improved timestamp logic
        """
        try:
            # CRITICAL FIX: Read from the synced L3 context cache instead of l3_output
            l3_context_cache = state.get("l3_context_cache", {})
            l3_data = l3_context_cache.get("last_output", {})

            # If no L3 data in cache, try fallback to l3_output
            if not l3_data:
                l3_data = state.get("l3_output", {})
                logger.debug(f"Using l3_output fallback for {symbol}")

            # If still no L3 data, consider it stale
            if not l3_data:
                logger.debug(f"No L3 data available for {symbol}")
                return {
                    'is_fresh': False,
                    'reason': 'no_l3_data',
                    'regime': 'unknown',
                    'price_change_pct': 0.0
                }

            # Get current timestamp for comparison
            current_time = pd.Timestamp.now()

            # Check L3 timestamp with robust parsing
            l3_timestamp_str = l3_data.get('timestamp')
            if l3_timestamp_str:
                try:
                    # Handle different timestamp formats with timezone consistency
                    if isinstance(l3_timestamp_str, str):
                        if l3_timestamp_str.endswith('Z'):
                            # Parse as UTC timezone-aware - handle both formats
                            try:
                                # Try direct parsing first
                                l3_timestamp = pd.Timestamp(l3_timestamp_str)
                            except:
                                # Fallback to manual parsing
                                l3_timestamp = pd.Timestamp(l3_timestamp_str.replace('Z', '+00:00'), tz='UTC')
                        else:
                            # Handle timestamps without Z suffix (from cached L3)
                            try:
                                # Try parsing as ISO format first
                                l3_timestamp = pd.Timestamp(l3_timestamp_str)
                                if l3_timestamp.tz is None:
                                    # Make it timezone-aware UTC if it's naive
                                    l3_timestamp = l3_timestamp.tz_localize('UTC')
                            except:
                                # Fallback: assume it's UTC and add timezone
                                try:
                                    l3_timestamp = pd.Timestamp(l3_timestamp_str, tz='UTC')
                                except:
                                    # Last resort: use current time minus small offset
                                    l3_timestamp = current_time - pd.Timedelta(seconds=60)
                                    logger.warning(f"Could not parse L3 timestamp '{l3_timestamp_str}', using current_time - 60s")
                    else:
                        l3_timestamp = pd.Timestamp(l3_timestamp_str)
                        if l3_timestamp.tz is None:
                            l3_timestamp = l3_timestamp.tz_localize('UTC')

                    # Ensure current_time is also timezone-aware for comparison
                    if current_time.tz is None:
                        current_time = current_time.tz_localize('UTC')

                    # Calculate time difference in seconds
                    time_diff_seconds = (current_time - l3_timestamp).total_seconds()

                    # For backtesting, use more lenient time thresholds to match L3 (increased to 45 minutes)
                    max_age_seconds = 2700  # 45 minutes for backtesting (increased from 30)

                    if time_diff_seconds > max_age_seconds:
                        logger.info(f"L3 context stale for {symbol}: {time_diff_seconds:.1f}s > {max_age_seconds}s")
                        return {
                            'is_fresh': False,
                            'reason': 'timestamp_too_old',
                            'regime': l3_data.get('regime', 'unknown'),
                            'price_change_pct': 0.0,
                            'time_diff_seconds': time_diff_seconds
                        }
                    else:
                        logger.debug(f"L3 context fresh for {symbol}: {time_diff_seconds:.1f}s old")

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing L3 timestamp for {symbol}: {e}")
                    # Don't fail completely - assume fresh if we can't parse
                    logger.info(f"⚠️ Could not parse L3 timestamp for {symbol}, assuming fresh")
                    # Continue to regime check instead of returning stale

            # Check regime and market conditions
            regime = l3_data.get('regime', 'unknown')

            # Only check price movements for volatile/range regimes
            if regime in ['volatile', 'range'] and len(df) >= 5:
                try:
                    recent_prices = df['close'].tail(5).astype(float)
                    price_change_pct = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100

                    # More lenient threshold for price movements (5% instead of 1%)
                    if abs(price_change_pct) > 5.0:
                        logger.info(f"Large price movement detected for {symbol}: {price_change_pct:.2f}%")
                        return {
                            'is_fresh': False,
                            'reason': 'large_price_movement',
                            'regime': regime,
                            'price_change_pct': price_change_pct
                        }
                except Exception as e:
                    logger.warning(f"Error checking price movement for {symbol}: {e}")

            # If we get here, L3 context is considered fresh
            return {
                'is_fresh': True,
                'reason': 'context_fresh',
                'regime': regime,
                'price_change_pct': 0.0
            }

        except Exception as e:
            logger.warning(f"Error checking L3 freshness for {symbol}: {e}")
            return {
                'is_fresh': False,
                'reason': 'error_checking',
                'regime': 'unknown',
                'price_change_pct': 0.0
            }

    def _generate_tactical_fallback_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """
        Generate tactical signal based on recent price action when L3 context is stale
        """
        try:
            if len(df) < 20:
                return None

            # Get recent price data
            recent_prices = df['close'].tail(20).astype(float)
            current_price = recent_prices.iloc[-1]
            prev_price = recent_prices.iloc[-2]

            # Calculate short-term trend
            short_trend = (current_price - recent_prices.iloc[-5]) / recent_prices.iloc[-5] * 100
            medium_trend = (current_price - recent_prices.iloc[-10]) / recent_prices.iloc[-10] * 100

            # Calculate momentum
            returns = recent_prices.pct_change().dropna()
            momentum = returns.tail(5).mean() * 100  # 5-period average return

            # Get technical indicators
            rsi_val = indicators.get('rsi')
            if hasattr(rsi_val, 'iloc'):
                rsi_val = float(rsi_val.iloc[-1])
            else:
                rsi_val = 50.0

            macd_val = indicators.get('macd')
            macd_signal_val = indicators.get('macd_signal')
            if hasattr(macd_val, 'iloc') and hasattr(macd_signal_val, 'iloc'):
                macd_val = float(macd_val.iloc[-1])
                macd_signal_val = float(macd_signal_val.iloc[-1])
                macd_diff = macd_val - macd_signal_val
            else:
                macd_diff = 0.0

            # Tactical decision logic
            confidence = 0.6  # Base confidence for tactical signals
            strength = 0.5    # Base strength

            # Strong momentum signals
            if abs(momentum) > 0.5:  # >0.5% average daily return
                if momentum > 0:
                    side = 'buy'
                    confidence = min(0.8, 0.6 + abs(momentum) * 0.5)
                else:
                    side = 'sell'
                    confidence = min(0.8, 0.6 + abs(momentum) * 0.5)
                strength = min(0.8, 0.5 + abs(momentum) * 2.0)

            # RSI-based signals
            elif rsi_val < 35:  # Oversold
                side = 'buy'
                confidence = 0.7
                strength = 0.6
            elif rsi_val > 65:  # Overbought
                side = 'sell'
                confidence = 0.7
                strength = 0.6

            # MACD-based signals
            elif abs(macd_diff) > 10:  # Significant MACD divergence
                if macd_diff > 0:
                    side = 'buy'
                    confidence = 0.65
                else:
                    side = 'sell'
                    confidence = 0.65
                strength = 0.55

            # Trend-following signals
            elif abs(short_trend) > 1.0:  # >1% move in 5 periods
                if short_trend > 0:
                    side = 'buy'
                    confidence = 0.6
                else:
                    side = 'sell'
                    confidence = 0.6
                strength = 0.5

            else:
                # No clear tactical signal
                side = 'hold'
                confidence = 0.4
                strength = 0.3

            # Create features dict
            features = {
                'close': current_price,
                'price_change_pct': (current_price - prev_price) / prev_price * 100,
                'short_trend_pct': short_trend,
                'medium_trend_pct': medium_trend,
                'momentum_pct': momentum,
                'rsi': rsi_val,
                'macd_diff': macd_diff,
                'tactical_reason': 'l3_stale_fallback'
            }

            return TacticalSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                confidence=confidence,
                signal_type='tactical_fallback',
                source='tactical_fallback',
                features=features,
                timestamp=pd.Timestamp.now(),
                metadata={
                    'l3_stale': True,
                    'tactical_indicators': {
                        'momentum': momentum,
                        'rsi': rsi_val,
                        'macd_diff': macd_diff,
                        'short_trend': short_trend
                    }
                }
            )

        except Exception as e:
            logger.error(f"❌ Error generating tactical fallback signal for {symbol}: {e}")
            return None

    def _apply_risk_adjustment(self, finrl_signal, risk_signals: List, l3_context: Dict) -> Any:
        """
        Apply risk-adjusted logic to FinRL signals based on L3 context and risk signals
        """
        try:
            # Get risk appetite from L3 context
            risk_appetite = l3_context.get('risk_appetite', 0.5)
            regime = l3_context.get('regime', 'neutral')

            # Analyze risk signals
            high_risk_signals = [s for s in risk_signals if getattr(s, 'severity', 'medium') == 'high']
            medium_risk_signals = [s for s in risk_signals if getattr(s, 'severity', 'medium') == 'medium']

            # Apply risk adjustments based on context
            if high_risk_signals:
                # High risk - reduce position size significantly or cancel
                if risk_appetite < 0.3:  # Conservative
                    logger.warning("High risk signals + conservative appetite - canceling signal")
                    return None
                else:
                    # Reduce strength significantly
                    finrl_signal.strength *= 0.3
                    finrl_signal.confidence *= 0.5

            elif medium_risk_signals:
                # Medium risk - moderate adjustment
                if regime in ['volatile', 'bear']:
                    finrl_signal.strength *= 0.6
                    finrl_signal.confidence *= 0.7
                else:
                    finrl_signal.strength *= 0.8
                    finrl_signal.confidence *= 0.8

            # Log risk adjustment
            logger.info(f"🎛️ Risk adjustment applied: strength={finrl_signal.strength:.2f}, confidence={finrl_signal.confidence:.2f}")
            return finrl_signal

        except Exception as e:
            logger.error(f"Error applying risk adjustment: {e}")
            return finrl_signal
