# l2_tactic/signal_generator - Gestión del portfolio
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
from .finrl_integration import FinRLProcessor, FinRLProcessorWrapper
from .signal_validator import validate_signal_list, validate_tactical_signal, create_fallback_signal


def safe_float(x):
    """
    Convierte a float el último valor de un array, lista o Serie.
    Evita el error "only length-1 arrays can be converted to Python scalars".
    """
    if isinstance(x, (list, np.ndarray, pd.Series)):
        if len(x) == 0:
            return np.nan
        return float(x[-1])  # último valor
    try:
        return float(x)
    except Exception:
        return np.nan

class L2TacticProcessor:
    """
    Generador de señales tácticas para L2.
    Incluye:
    - Señales AI (FinRL)
    - Señales técnicas (multi-timeframe)
    - Risk overlay (ajustes por contexto L3)
    - Fallback táctico cuando L3 está stale
    - Construcción de features cross-L3
    """

    # --- Dimensiones de features ---
    FINRL_OBS_DIM = 257
    BASE_FEATURES_DIM = 246
    CROSS_FEATURES_DIM = 11
    PREFERRED_BASE_COLS: Optional[List[str]] = None

    def __init__(self, config):
        """Inicializa el procesador táctico."""
        self.config = config
        self.multi_timeframe = MultiTimeframeTechnical(config)
        self.risk_overlay = RiskOverlay(config)

        # Pasar configuración correcta a SignalComposer
        signal_config = getattr(config, 'signals', config) if hasattr(config, 'signals') else config
        self.signal_composer = SignalComposer(signal_config)

        # Cargar modelo FinRL desde config.ai_model.model_path
        model_path = getattr(config.ai_model, "model_path", None) if hasattr(config, 'ai_model') else None
        logger.info(f"🔍 Intentando cargar modelo desde: {model_path}")
        if model_path:
            logger.info(f"🔍 Ruta absoluta del modelo: {os.path.abspath(model_path)}")
            logger.info(f"🔍 Existe archivo?: {os.path.exists(model_path)}")
        if not model_path:
            logger.warning("⚠️ No se encontró model_path en la configuración L2. Usando fallback.")
            model_path = "models/L2/gemini.zip"
            logger.info(f"🔍 Usando ruta fallback: {model_path}")
            logger.info(f"🔍 Ruta absoluta fallback: {os.path.abspath(model_path)}")

        # Store model path for potential reloading
        self.current_model_path = model_path

        try:
            self.finrl_processor = FinRLProcessor(model_path)

            # Detectar modelo y crear wrapper inteligente
            model_name = self._detect_model_name(model_path)
            self.finrl_wrapper = FinRLProcessorWrapper(self.finrl_processor, model_name)
            logger.info(f"✅ FinRLProcessorWrapper creado para modelo: {model_name}")

        except Exception as e:
            logger.error(f"❌ Error cargando FinRL processor: {e}")
            self.finrl_processor = None
            self.finrl_wrapper = None

    def _detect_model_name(self, model_path: str) -> str:
        """Detecta el nombre del modelo desde la ruta del archivo"""
        if not model_path:
            return "unknown"

        model_path_lower = model_path.lower()

        # Detectar modelos específicos
        if "gemini" in model_path_lower:
            return "gemini"
        elif "deepseek" in model_path_lower:
            return "deepseek"
        elif "claude" in model_path_lower:
            return "claude"
        elif "kimi" in model_path_lower:
            return "kimi"
        elif "gpt" in model_path_lower:
            return "gpt"
        else:
            # Modelo genérico - intentar inferir desde dimensiones esperadas
            if self.finrl_processor and hasattr(self.finrl_processor, 'observation_space_info'):
                expected_dims = self.finrl_processor.observation_space_info.get('expected_dims', 257)
                if expected_dims == 13:
                    return "gemini"  # Legacy 13 features
                elif expected_dims == 971:
                    return "claude"  # Risk-aware 971 features
                elif expected_dims == 257:
                    return "finrl_standard"  # Standard FinRL 257 features
                else:
                    return f"custom_{expected_dims}"

            return "unknown"

    def switch_model(self, model_key: str) -> bool:
        """Switch to a different L2 model dynamically"""
        try:
            if not hasattr(self.config, 'ai_model'):
                logger.error("❌ No ai_model config available for switching")
                return False

            # Try to switch model in config
            if self.config.ai_model.switch_model(model_key):
                new_model_path = self.config.ai_model.model_path
                logger.info(f"🔄 Switching L2 model to: {model_key} -> {new_model_path}")

                # Check if file exists
                if not os.path.exists(new_model_path):
                    logger.error(f"❌ Model file does not exist: {new_model_path}")
                    return False

                # Create new processor with new model
                try:
                    new_processor = FinRLProcessor(new_model_path)
                    self.finrl_processor = new_processor
                    self.current_model_path = new_model_path
                    logger.info(f"✅ Successfully switched to model: {model_key}")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to load new model {model_key}: {e}")
                    return False
            else:
                logger.error(f"❌ Config switch_model failed for: {model_key}")
                return False
        except Exception as e:
            logger.error(f"❌ Error switching model to {model_key}: {e}")
            return False

    async def process_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Procesa datos de mercado para generar señales tácticas.
        Combina:
        - Señales AI
        - Señales técnicas
        - Ajustes de riesgo
        - Fallback cuando L3 stale
        """
        try:
            market_data = state.get("market_data_simple") or state.get("market_data", {})
            if not market_data:
                logger.warning("⚠️ L2: No hay datos de mercado disponibles")
                return []

            signals = []
            for symbol, data in market_data.items():
                df = self._extract_dataframe(data, symbol)
                if df is None:
                    continue

                # Calcular indicadores técnicos multi-timeframe
                indicators = self.multi_timeframe.calculate_technical_indicators(df)

                # Señal AI
                finrl_signal = await self._get_finrl_signal(state, symbol, indicators)
                if not finrl_signal:
                    continue

                # Evaluar contexto L3
                l3_context = self._check_l3_context_freshness(state, symbol, df)
                l3_context_cache = state.get("l3_context_cache", {})
                l3_output = l3_context_cache.get("last_output", {}) or state.get("l3_output", {})

                if l3_output and l3_context['is_fresh']:
                    risk_signals = await self.risk_overlay.generate_risk_signals(
                        market_data={symbol: df},
                        portfolio_data=state.get("portfolio", {}),
                        l3_context=l3_output
                    )
                    if risk_signals:
                        finrl_signal = self._apply_risk_adjustment(finrl_signal, risk_signals, l3_output)

                    risk_filtered = finrl_signal

                elif l3_output and not l3_context['is_fresh']:
                    tactical_signal = self._generate_tactical_fallback_signal(symbol, df, indicators)
                    risk_filtered = tactical_signal or finrl_signal

                else:
                    risk_filtered = finrl_signal

                # Componer señal final
                tactical_signal = self.signal_composer.compose_signal(
                    symbol=symbol,
                    base_signal=risk_filtered,
                    indicators=indicators,
                    state=state
                )

                validated_signal = validate_tactical_signal(tactical_signal) if tactical_signal else create_fallback_signal(symbol, "no_composition")
                signals.append(validated_signal)

            return validate_signal_list(signals)

        except Exception as e:
            logger.error(f"❌ Error procesando señales L2: {e}", exc_info=True)
            return []

    # -------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------

    def _extract_dataframe(self, data, symbol: str) -> Optional[pd.DataFrame]:
        """Extrae DataFrame válido del formato de backtesting o real"""
        try:
            if isinstance(data, dict):
                if 'historical_data' in data and isinstance(data['historical_data'], pd.DataFrame):
                    df = data['historical_data']
                    if len(df) < 200:
                        logger.warning(f"⚠️ Datos históricos insuficientes para {symbol}: {len(df)} < 200")
                        return None
                    return df
                else:
                    logger.warning(f"⚠️ Datos insuficientes para {symbol}: formato single point")
                    return None
            elif isinstance(data, pd.DataFrame):
                if data.empty or len(data) < 200:
                    logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(data)} < 200")
                    return None
                return data
            else:
                logger.warning(f"⚠️ Formato de datos inválido para {symbol}: {type(data)}")
                return None
        except Exception as e:
            logger.error(f"❌ Error extrayendo DataFrame para {symbol}: {e}")
            return None

    async def _get_finrl_signal(self, state: Dict[str, Any], symbol: str, indicators: Dict) -> Optional[TacticalSignal]:
        """Obtiene señal de FinRL y asegura source='ai'"""
        try:
            if not self.finrl_processor:
                return None

            # Detectar método disponible automáticamente
            signal = await self._get_finrl_signal_safe(state, symbol, indicators)
            if signal:
                signal.source = 'ai'
            return signal
        except Exception as e:
            logger.error(f"❌ Error obteniendo señal FinRL para {symbol}: {e}")
            return None

    async def _get_finrl_signal_safe(self, state: Dict[str, Any], symbol: str, indicators: Dict) -> Optional[TacticalSignal]:
        """
        Usa el FinRLProcessorWrapper para generar señales automáticamente
        """
        try:
            if not self.finrl_wrapper:
                logger.warning(f"⚠️ FinRL wrapper no disponible para {symbol}")
                return None

            # Preparar datos de mercado para el wrapper
            market_data = state.get("market_data_simple") or state.get("market_data", {})

            # Usar el wrapper inteligente que maneja automáticamente:
            # - Detección de modelo (Gemini, DeepSeek, Claude, Kimi)
            # - Ajuste de forma de observaciones
            # - Método correcto (predict/get_action)
            signal = await self.finrl_wrapper.generate_signal(market_data, symbol, indicators)

            if signal:
                logger.debug(f"✅ Señal FinRL generada para {symbol}: {signal.side} (conf: {signal.confidence:.2f})")
            else:
                logger.debug(f"⚠️ No se pudo generar señal FinRL para {symbol}")

            return signal

        except Exception as e:
            logger.error(f"❌ Error en _get_finrl_signal_safe para {symbol}: {e}")
            return None

    # -------------------------------------------
    # Features y métricas multiasset
    # -------------------------------------------

    def _select_base_features_row(self, features_df: pd.DataFrame) -> List[float]:
        """Select numeric features for AI model"""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * self.BASE_FEATURES_DIM
        last_row = features_df.iloc[-1]
        if self.PREFERRED_BASE_COLS:
            cols = [c for c in self.PREFERRED_BASE_COLS if c in features_df.columns]
        else:
            cols = sorted([c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])])
        values = []
        for c in cols[:self.BASE_FEATURES_DIM]:
            try:
                # Use safe_float for robust conversion
                val = last_row[c]
                v = safe_float(val)
                values.append(v if np.isfinite(v) else 0.0)
            except Exception as e:
                logger.debug(f"Error converting feature {c}: {e}, using 0.0")
                values.append(0.0)
        if len(values) < self.BASE_FEATURES_DIM:
            values.extend([0.0] * (self.BASE_FEATURES_DIM - len(values)))
        return values[:self.BASE_FEATURES_DIM]

    def _get_key_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        last_row = features_df.iloc[-1]
        metrics = {}
        for key, default in [
            ('rsi', 50.0),
            ('macd', 0.0),
            ('macd_signal', 0.0),
            ('sma_20', 0.0),
            ('sma_50', 0.0),
            ('bollinger_upper', 0.0),
            ('bollinger_lower', 0.0),
            ('vol_zscore', 0.0)
        ]:
            try:
                val = last_row.get(key, default)
                # Use safe_float for robust conversion
                converted_val = safe_float(val)
                metrics[key] = converted_val if np.isfinite(converted_val) else default
            except Exception as e:
                logger.debug(f"Error converting metric {key}: {e}, using default {default}")
                metrics[key] = default
        return metrics

    def _compute_eth_btc_ratio(self, market_data: Dict[str, pd.DataFrame],
                               features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """ETH/BTC ratio"""
        try:
            eth = features_by_symbol.get("ETHUSDT") or market_data.get("ETHUSDT")
            btc = features_by_symbol.get("BTCUSDT") or market_data.get("BTCUSDT")
            eth_close = float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and not eth.empty else None
            btc_close = float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and not btc.empty else None
            if eth_close is not None and btc_close not in (None, 0.0):
                return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """30-period correlation BTC vs ETH"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0
            eth_close = eth["close"].astype(float).tail(30)
            btc_close = btc["close"].astype(float).tail(30)
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
            return corr if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    def _compute_spread_pct(self, market_data: Dict[str, pd.DataFrame],
                            features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """(BTC - ETH)/BTC spread %"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame):
                eth_close = float(eth["close"].iloc[-1])
                btc_close = float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0

    # -------------------------------------------
    # L3 context checks & tactical fallback
    # -------------------------------------------

    def _check_l3_context_freshness(self, state: Dict[str, Any], symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica si L3 context está fresco o stale"""
        try:
            l3_context_cache = state.get("l3_context_cache", {})
            l3_data = l3_context_cache.get("last_output", {}) or state.get("l3_output", {})
            if not l3_data:
                return {'is_fresh': False, 'reason': 'no_l3_data', 'regime': 'unknown', 'price_change_pct': 0.0}

            current_time = pd.Timestamp.now()
            l3_timestamp_str = l3_data.get('timestamp')
            if l3_timestamp_str:
                try:
                    l3_timestamp = pd.Timestamp(l3_timestamp_str)
                    if l3_timestamp.tz is None:
                        l3_timestamp = l3_timestamp.tz_localize('UTC')
                    if current_time.tz is None:
                        current_time = current_time.tz_localize('UTC')
                    time_diff_seconds = (current_time - l3_timestamp).total_seconds()
                    max_age_seconds = 2700
                    if time_diff_seconds > max_age_seconds:
                        return {'is_fresh': False, 'reason': 'timestamp_too_old', 'regime': l3_data.get('regime', 'unknown'), 'price_change_pct': 0.0}
                except Exception:
                    logger.warning(f"Error parsing L3 timestamp for {symbol}, assuming fresh")
            regime = l3_data.get('regime', 'unknown')
            return {'is_fresh': True, 'reason': 'context_fresh', 'regime': regime, 'price_change_pct': 0.0}
        except Exception as e:
            logger.warning(f"Error checking L3 freshness for {symbol}: {e}")
            return {'is_fresh': False, 'reason': 'error_checking', 'regime': 'unknown', 'price_change_pct': 0.0}

    def _generate_tactical_fallback_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Señal táctica cuando L3 está stale"""
        try:
            if len(df) < 20:
                return None
            recent_prices = df['close'].tail(20).astype(float)
            current_price = recent_prices.iloc[-1]
            prev_price = recent_prices.iloc[-2]
            short_trend = (current_price - recent_prices.iloc[-5]) / recent_prices.iloc[-5] * 100
            medium_trend = (current_price - recent_prices.iloc[-10]) / recent_prices.iloc[-10] * 100
            returns = recent_prices.pct_change().dropna()
            momentum = returns.tail(5).mean() * 100
            # Safely extract indicator values
            rsi_val = 50.0
            macd_val = 0.0
            macd_signal_val = 0.0

            # Use safe_float for all indicator extractions
            rsi_val = safe_float(indicators.get('rsi', 50.0))
            if np.isnan(rsi_val):
                rsi_val = 50.0

            macd_val = safe_float(indicators.get('macd', 0.0))
            if np.isnan(macd_val):
                macd_val = 0.0

            macd_signal_val = safe_float(indicators.get('macd_signal', 0.0))
            if np.isnan(macd_signal_val):
                macd_signal_val = 0.0
            macd_diff = macd_val - macd_signal_val

            # Lógica de decisión
            confidence, strength, side = 0.6, 0.5, 'hold'
            if abs(momentum) > 0.5:
                side = 'buy' if momentum > 0 else 'sell'
                confidence = min(0.8, 0.6 + abs(momentum)*0.5)
                strength = min(0.8, 0.5 + abs(momentum)*2.0)
            elif rsi_val < 35: side, confidence, strength = 'buy', 0.7, 0.6
            elif rsi_val > 65: side, confidence, strength = 'sell', 0.7, 0.6
            elif abs(macd_diff) > 10: side, confidence, strength = ('buy' if macd_diff>0 else 'sell'), 0.65, 0.55
            elif abs(short_trend) > 1.0: side, confidence, strength = ('buy' if short_trend>0 else 'sell'), 0.6, 0.5

            features = {
                'close': current_price,
                'price_change_pct': (current_price - prev_price)/prev_price*100,
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
                metadata={'l3_stale': True}
            )
        except Exception as e:
            logger.error(f"❌ Error generating tactical fallback signal for {symbol}: {e}")
            return None

    def _apply_risk_adjustment(self, finrl_signal, risk_signals: List, l3_context: Dict) -> Any:
        """Aplicar ajustes de riesgo a la señal FinRL"""
        try:
            risk_appetite = l3_context.get('risk_appetite', 0.5)
            regime = l3_context.get('regime', 'neutral')
            high_risk = [s for s in risk_signals if getattr(s, 'severity', 'medium')=='high']
            med_risk = [s for s in risk_signals if getattr(s, 'severity', 'medium')=='medium']

            if high_risk and risk_appetite < 0.3:
                logger.warning("High risk + conservative - canceling signal")
                return None
            if high_risk:
                finrl_signal.strength *= 0.3
                finrl_signal.confidence *= 0.5
            elif med_risk:
                if regime in ['volatile','bear']:
                    finrl_signal.strength *= 0.6
                    finrl_signal.confidence *= 0.7
                else:
                    finrl_signal.strength *= 0.8
                    finrl_signal.confidence *= 0.8

            logger.info(f"🎛️ Risk adjustment applied: strength={finrl_signal.strength:.2f}, confidence={finrl_signal.confidence:.2f}")
            return finrl_signal
        except Exception as e:
            logger.error(f"Error applying risk adjustment: {e}")
            return finrl_signal
